import torch
import scipy.io.wavfile
import os
from diffusers import AudioLDMPipeline

OUTPUT_DIR = "baseline_audioldm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Initializing AudioLDM configuration...")
repo_id = "cvssp/audioldm-m-full"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

test_prompts = [
    "footstep",
    "keyboard",
    "dog bark",
    "gunshot",
    "moving motor vehicle",
    "rain",
    "sneeze"
]

print("Processing generation queue...")
for idx, prompt in enumerate(test_prompts):
    formatted_prompt = f"The sound of {prompt.lower()}."
    print(f"Generating sequence for: {formatted_prompt}")

    audio = pipe(
        formatted_prompt,
        num_inference_steps=50,
        audio_length_in_s=5.94,
        num_waveforms_per_prompt=1
    ).audios[0]

    file_path = os.path.join(OUTPUT_DIR, f"sample_{idx}_audioldm.wav")
    scipy.io.wavfile.write(file_path, rate=16000, data=audio)

print("Finished.")