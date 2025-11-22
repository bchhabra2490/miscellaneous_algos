import yt_dlp
import whisper
import os
import re
import tempfile

from openai import OpenAI

client = OpenAI()


def download_audio(youtube_url, output_path="audio.mp3"):
    ydl_opts = {
        "format": "bestaudio/best",  # best available audio
        "outtmpl": output_path,  # where to save the file
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",  # you can choose: mp3, m4a, wav, opus
                "preferredquality": "192",
            }
        ],
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])


def transcribe_audio(audio_path):
    """
    Transcribes audio using Whisper
    """
    print("Loading Whisper model...")
    model = whisper.load_model("small")  # or "small", "medium", "large"

    print("Transcribing...")
    result = model.transcribe(audio_path)

    return result["text"]


def identify_key_points(transcript):
    """
    STEP 1: Ask GPT to identify what important points to extract.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "From the transcript, identify the important points, themes, arguments, or insights that should be extracted. Output a clean bullet list of what to extract in step 2.",
            },
            {"role": "user", "content": transcript},
        ],
    )
    return response.choices[0].message.content


def extract_using_plan(transcript, plan):
    """
    STEP 2: Use GPT again, giving it the plan + transcript to extract only the important content.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Using the extraction plan provided, extract only the important content from the transcript. Be concise, factual, and structured.",
            },
            {"role": "user", "content": f"Extraction plan:\n{plan}\n\nTranscript:\n{transcript}"},
        ],
    )
    return response.choices[0].message.content


def generate_audio_script(extracted):
    """
    STEP 3: Use GPT to generate an audio script from the extracted content.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Generate an audio script from the extracted content. Be concise, factual, and structured.",
            },
            {"role": "user", "content": f"Extracted content:\n{extracted}"},
        ],
    )
    return response.choices[0].message.content


def split_text_into_chunks(text, max_chars=4096):
    """
    Split text into chunks of max_chars, trying to break at sentence boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    # Try to split at sentence boundaries (period, exclamation, question mark followed by space or newline)
    sentences = re.split(r"([.!?]\s+)", text)

    current_chunk = ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")

        # If adding this sentence would exceed the limit, save current chunk and start new one
        if len(current_chunk) + len(sentence) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # If any chunk is still too long (shouldn't happen, but safety check), split it hard
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            # Hard split if needed
            for i in range(0, len(chunk), max_chars):
                final_chunks.append(chunk[i : i + max_chars])

    return final_chunks


def generate_audio_file(audio_script):
    """
    Use OpenAI TTS to generate an audio file from the audio script.
    Handles long scripts by splitting into chunks and combining the audio.
    """
    # Split the script into chunks if it's too long
    chunks = split_text_into_chunks(audio_script, max_chars=4096)

    if len(chunks) == 1:
        # Single chunk - simple case
        response = client.audio.speech.create(
            model="tts-1",
            input=audio_script,
            voice="alloy",
        )
        with open("audio_summary.mp3", "wb") as f:
            f.write(response.content)
        print("Audio file saved to audio_summary.mp3")
    else:
        # Multiple chunks - generate audio for each and combine
        print(f"Script is too long ({len(audio_script)} chars). Splitting into {len(chunks)} chunks...")

        temp_files = []
        try:
            # Generate audio for each chunk
            for i, chunk in enumerate(chunks):
                print(f"Generating audio for chunk {i+1}/{len(chunks)}...")
                response = client.audio.speech.create(
                    model="tts-1",
                    input=chunk,
                    voice="alloy",
                )

                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                temp_file.write(response.content)
                temp_file.close()
                temp_files.append(temp_file.name)

            # Combine audio files using pydub if available, otherwise use ffmpeg directly
            try:
                from pydub import AudioSegment

                # Load and concatenate all audio segments
                combined = AudioSegment.empty()
                for temp_file in temp_files:
                    segment = AudioSegment.from_mp3(temp_file)
                    combined += segment

                # Export the combined audio
                combined.export("audio_summary.mp3", format="mp3")
                print("Audio file saved to audio_summary.mp3")

            except ImportError:
                # Fallback to ffmpeg if pydub is not available
                import subprocess

                # Create a file list for ffmpeg concat
                concat_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
                for temp_file in temp_files:
                    concat_file.write(f"file '{os.path.abspath(temp_file)}'\n")
                concat_file.close()

                # Use ffmpeg to concatenate
                subprocess.run(
                    [
                        "ffmpeg",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        concat_file.name,
                        "-c",
                        "copy",
                        "audio_summary.mp3",
                    ],
                    check=True,
                    capture_output=True,
                )

                os.unlink(concat_file.name)
                print("Audio file saved to audio_summary.mp3")

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass


if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    download_audio(url, "output.mp3")
    print("Audio saved to output.mp3")
    transcription = transcribe_audio("output.mp3.mp3")

    # Save the transcription to a file
    with open("transcription.txt", "w") as f:
        f.write(transcription)
    print("Transcription saved to transcription.txt")

    with open("transcription.txt", "r") as f:
        transcription = f.read()

    plan = identify_key_points(transcription)
    print("Plan:")
    print(plan)

    extracted = extract_using_plan(transcription, plan)
    print("Extracted:")
    print(extracted)

    with open("extracted_content.txt", "w") as f:
        f.write(extracted)

    with open("extracted_content.txt", "r") as f:
        extracted = f.read()

    audio_script = generate_audio_script(extracted)
    print("Audio Script:")
    print(audio_script)

    with open("audio_script.txt", "w") as f:
        f.write(audio_script)
    print("Audio script saved to audio_script.txt")

    # Generate an audio file from the audio script
    generate_audio_file(audio_script)
