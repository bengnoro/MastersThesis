import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import os

OUTPUT_DIR = "baseline_audiogen"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Initializing AudioGen configuration...")
model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=5.94)

test_prompts = [
    "footstep",
    "keyboard",
    "dog bark",
    "gunshot",
    "moving motor vehicle",
    "rain",
    "sneeze"
]

formatted_prompts = [f"The sound of {p.lower()}." for p in test_prompts]

print("Processing generation queue...")
wavs = model.generate(formatted_prompts)

for idx, (prompt, wav) in enumerate(zip(test_prompts, wavs)):
    file_path = os.path.join(OUTPUT_DIR, f"sample_{idx}_audiogen")
    audio_write(file_path, wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    print(f"Stored: {file_path}.wav")

print("Finished.")