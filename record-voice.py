import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # sample rate
duration = 5  # seconds

print("Recording...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
print("Done!")

write("original-voice.wav", fs, audio)
