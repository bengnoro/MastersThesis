import torch
import torchaudio
import numpy as np
from PIL import Image
import scipy.io.wavfile as wavfile
import os
import warnings

from diffusers import StableDiffusionPipeline

warnings.filterwarnings("ignore")

OUTPUT_DIR = "baseline_riffusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 512
F_MIN = 0.0
F_MAX = 10000.0
MAX_DB = 80.0
MIN_DB = -100.0


def image_to_spectrogram(image: Image.Image):
    """
    Decodes the generated image back into an acoustic amplitude format.
    """
    image = image.convert("L")
    img_data = np.array(image).astype(np.float32)
    img_data = np.flipud(img_data)

    db_spec = (img_data / 255.0) * (MAX_DB - MIN_DB) + MIN_DB
    amp_spec = 10.0 ** (db_spec / 20.0)

    return torch.tensor(amp_spec, dtype=torch.float32)


def generate_audio(prompt, pipe, device):
    """
    Creates an image from a prompt and translates it into playable audio.
    """
    print(f"Generating layout for: {prompt}")
    image = pipe(prompt, num_inference_steps=50, width=512, height=512).images[0]

    amp_spec = image_to_spectrogram(image).unsqueeze(0).to(device)

    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
        f_min=F_MIN,
        f_max=F_MAX
    ).to(device)
    lin_spec = inverse_mel(amp_spec)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=1.0,
        n_iter=64
    ).to(device)

    waveform = griffin_lim(lin_spec)
    waveform = waveform / torch.max(torch.abs(waveform))

    return waveform.squeeze().cpu().numpy()


def main():
    """
    Executes the text-to-audio process across a list of test prompts.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading primary dependencies...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "riffusion/riffusion-model-v1",
        torch_dtype=torch.float16
    ).to(device)

    pipe.safety_checker = None

    test_prompts = [
        "footstep",
        "keyboard",
        "dog bark",
        "gunshot",
        "moving motor vehicle",
        "rain",
        "sneeze"
    ]

    for idx, prompt in enumerate(test_prompts):
        formatted_prompt = f"The sound of {prompt.lower()}."
        audio_data = generate_audio(formatted_prompt, pipe, device)

        file_path = os.path.join(OUTPUT_DIR, f"sample_{idx}_riffusion.wav")
        wavfile.write(file_path, SAMPLE_RATE, audio_data)
        print(f"Stored: {file_path}")

    print("Finished.")


if __name__ == "__main__":
    main()