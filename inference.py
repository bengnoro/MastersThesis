import torch
import torchaudio
import os
import warnings

try:
    import bigvgan
except ImportError:
    print("Please install NVIDIA BigVGAN: `pip install bigvgan`")
    exit()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)

from models import Generator, NOISE_DIM
from data_pipeline import (
    TEXT_ENCODER_MODEL, EMBEDDING_DIM, SAMPLE_RATE,
    ClapTextEncoder, DATASET_MEAN, DATASET_STD
)

CHECKPOINT_PATH = "checkpoints/gen_epoch_995.pth.tar"
OUTPUT_FOLDER = "generated_samples_bigvgan"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_GEN = 512


def get_bigvgan_pipeline():
    print("Loading NVIDIA BigVGAN v2 (22kHz, 80 Band)...")
    vocoder = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x', use_cuda_kernel=False).to(DEVICE)
    vocoder.remove_weight_norm()
    vocoder.eval()
    return vocoder


def denormalize_to_log_amplitude(spectrogram):
    """
    1. Reverses Z-score back to Decibels (dB)
    2. Converts dB back to physical amplitude
    3. Translates amplitude to natural log space expected by BigVGAN
    """
    S_db = (spectrogram * (3.0 * DATASET_STD)) + DATASET_MEAN
    S_amp = 10.0 ** (S_db / 20.0)
    S_log = torch.log(torch.clamp(S_amp, min=1e-5))
    return S_log


def generate_sound(text_prompt, generator, text_encoder, vocoder):
    print(f"Generating BigVGAN sound for: '{text_prompt}'...")

    with torch.no_grad():
        formatted_prompt = f"The sound of {text_prompt.lower()}."
        text_emb = text_encoder.encode(formatted_prompt, convert_to_tensor=True).to(DEVICE)
        noise = torch.randn(1, NOISE_DIM).to(DEVICE)

        generator.eval()
        generated_spec = generator(noise, text_emb)

        S_log = denormalize_to_log_amplitude(generated_spec.squeeze(1))
        audio_waveform = vocoder(S_log).squeeze(1)
        max_val = torch.max(torch.abs(audio_waveform))
        if max_val > 0:
            audio_waveform = audio_waveform / max_val

        return audio_waveform.cpu()


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Loading CLAP Semantic-Audio Text Encoder...")
    text_encoder = ClapTextEncoder(TEXT_ENCODER_MODEL, device=DEVICE)

    print("Loading SA-ResGAN Generator...")
    gen = Generator(noise_dim=NOISE_DIM, text_dim=EMBEDDING_DIM, base_channels=FEATURES_GEN).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        gen.load_state_dict(checkpoint["state_dict"])
    else:
        print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found. Generator output will be random noise.")

    gen.eval()

    vocoder = get_bigvgan_pipeline()

    test_prompts = [
        "footstep",
        "keyboard",
        "dog bark",
        "gunshot",
        "moving motor vehicle",
        "rain",
        "sneeze"
    ]

    for i, prompt in enumerate(test_prompts):
        waveform = generate_sound(prompt, gen, text_encoder, vocoder)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        torchaudio.save(os.path.join(OUTPUT_FOLDER, f"sample_{i}_bigvgan.wav"), waveform, SAMPLE_RATE)

    print(f"\nBigVGAN Generation complete. Check the '{OUTPUT_FOLDER}' folder.")


if __name__ == "__main__":
    main()