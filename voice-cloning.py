# Workaround for torchcodec loading issue on macOS
# Patch torchaudio.load to use librosa/soundfile instead
import torchaudio
import librosa
import torch
import numpy as np

_original_torchaudio_load = torchaudio.load


def patched_torchaudio_load(filepath, **kwargs):
    """Patch torchaudio.load to use librosa when torchcodec fails"""
    try:
        # Try original method first
        return _original_torchaudio_load(filepath, **kwargs)
    except RuntimeError as e:
        if "libtorchcodec" in str(e) or "Could not load" in str(e):
            # Fallback to librosa
            print(f"Warning: torchcodec failed, using librosa fallback for {filepath}")
            audio, sr = librosa.load(filepath, sr=None, mono=False)
            # Convert to torch tensor format
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1)  # Add channel dimension
            audio_tensor = torch.from_numpy(audio).float()
            return audio_tensor, sr
        else:
            raise


torchaudio.load = patched_torchaudio_load

from f5_tts.api import F5TTS
import soundfile as sf

f5tts = F5TTS()
text_to_synthesize = "The sun was just beginning to dip behind the tallest treetops of Fernwood Forest, painting the sky in the gentlest shades of lavender, rose, and sleepy gold. ... Everything felt hushed and soft, as if the whole forest was preparing for a quiet, cozy night. ... A cool breeze rustled gently through the leaves, carrying the sweet scents of pine needles and wildflowers. ... Far above, a pair of fireflies twinkled, like tiny stars come down to dance among the branches. ... [slowly] On a cushy patch of moss near the old oak tree, Ollie the Owl stretched his wings, blinking slowly as dusk settled around him. ... His feathers caught the last golden light, making him look even more wise and gentle than usual. ... He turned his head and listened to the soft footsteps among the ferns—Sadie the Squirrel was hopping in little zigzags, her tail bouncing behind her, and Tammy the Turtle ambled quietly beside her, each careful step pressing tiny patterns into the soft earth. ... “Good evening, dear friends,” Ollie called in a warm, low voice. ... [warmly] “The air feels perfect for a gentle stroll, don’t you think?”  ... ... ... Sadie grinned and did a little twirl. ... “Oh! ... Yes, yes! ... Let’s go on an adventure!” she chirped, her eyes sparkling. ... “We could see how fast we can run down the path!”  ... ... ... Tammy the Turtle looked up with a calm, soft smile. ... [softly] “A walk sounds lovely, Sadie, but maybe tonight… we could try going slow? ... Sometimes, you see the most wonderful things when you’re not in a hurry.”  ... ... ... Sadie wriggled her nose, thinking. ... “Hmm… Slow? ... But fast is so fun!”  ... ... ... Ollie nodded thoughtfully. ... “Tammy has a gentle wisdom. ... What if, just for tonight, we all try walking at Tammy’s pace? ... It could be a new kind of adventure.”  ... ... ... Sadie giggled, “Okay! ... I’ll try, but you might have to remind me!”  ... ... ... With that, the three friends set out, the forest path bathed in the softest, shadowy light. ... Tammy led the way, her shell gleaming faintly in the glow of the eager fireflies. ... Sadie bounced once, then carefully matched her steps to Tammy’s slow, steady rhythm. ... Ollie fluttered gently above, landing on low branches to keep company. ... [whispering] As they moved, the world seemed even quieter and more magical. ... The hush of the forest was filled with tiny sounds they’d never noticed before: the creak of tree branches, the brush of wings, the hush-hush of the wind through the ferns. ... Sadie noticed a little family of snails making their own slow way across a log, their shiny trails glowing in the firefly light. ... [softly] “Look!” Sadie whispered, crouching to watch. ... “I never saw that before!”  ... ... ... Tammy chuckled gently, “When you go slow, you see things you miss when you’re in a hurry.”  ... ... ... Ollie perched above, his big eyes twinkling. ... “There’s a whole quiet world here, just waiting for patient friends to find it.”  ... ... ... They continued on, passing a patch of silver mushrooms and a ring of tiny bluebells that only opened in the moon’s gentle glow. ... Sadie noticed how beautiful Tammy’s careful steps looked, each one careful and full of thought. ... She smiled and followed along, her own feet learning the quiet rhythm. ... Soon, they reached the old willow tree by the little stream. ... The water flowed past with the softest gurgle, and the air was full of the gentle hum of crickets. ... Ollie settled onto a low branch, fluffing his feathers, while Tammy and Sadie sat together on the moss. ... [softly] “Thank you for going slow with me,” Tammy said, her eyes kind. ... “It feels nice to share my favorite pace with you.”  ... ... ... Sadie stretched and let out a tiny yawn. ... “I liked it! ... I saw so many things I never saw before. ... And it felt… cozy. ... Like the forest was giving us a hug.”  ... ... ... Ollie nodded, his voice like a warm blanket. ... “Sometimes, sharing a slower moment helps us notice the best things, and helps us be gentle with each other—and ourselves.”  ... ... ... [whispers] The friends stayed close, listening to the stream and the cricket song. ... The sky turned deeper blue, and a few bright stars began to peek out overhead. ... Sadie’s eyelids drooped, and Tammy curled quietly into the moss, her shell shining in the moonlight. ... Ollie let out a quiet hoot, low and comforting, as a gentle wind rocked the branches above them. ... Tammy whispered, “I hope we can walk together like this again.”  ... ... ... “Me too,” Sadie murmured, snuggling close. ... [slowly] Ollie watched over his friends, feeling peaceful as the night grew soft and quiet. ... Fireflies drifted lazily through the air, their glow growing dim as they, too, grew sleepy. ... The gentle hush of the forest wrapped all around, like a lullaby. ... Soon, Sadie’s little breaths grew slow and even. ... Tammy’s eyes closed, her face peaceful in the cool night air. ... Ollie tucked his head under his wing, letting out one last, sleepy sigh. ... The moon rose higher, and the soft world of Fernwood Forest drifted into deep, gentle rest. ... The friends slept quietly, their hearts full of the kindness and patience they’d shared. ... Outside, the world was calm and still, and all around, the night grew peaceful and slow, inviting every sleepy creature to rest and dream. ... [whispers] Goodnight, Ollie. ... Goodnight, Sadie. ... Goodnight, Tammy. ... The forest watched over them as they drifted to sleep, wrapped in the gentle magic of their slower walk."
generated_audio, sample_rate, spec = f5tts.infer(
    gen_text=text_to_synthesize,
    ref_file="original-voice.wav",
    ref_text="I am recording my voice so that I can have it cloned.",
)

sf.write("generated_audio.wav", generated_audio, sample_rate)
print("Audio saved to generated_audio.wav")
