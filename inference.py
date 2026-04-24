import os
import torch
import torchaudio
import warnings

try:
    import bigvgan
except ImportError:
    print("Please install NVIDIA BigVGAN: `pip install bigvgan`")
    exit()

warnings.filterwarnings("ignore", category=FutureWarning)

from models import Generator, NOISE_DIM
from data_pipeline import (
    TEXT_ENCODER_MODEL, EMBEDDING_DIM, SAMPLE_RATE,
    ClapTextEncoder, DATASET_MEAN, DATASET_STD
)

# --- HARDCODED DIAGNOSTIC PATHS ---
CHECKPOINT_PATH = "/content/drive/MyDrive/diplomka/checkpoints/gen_epoch_525.pth.tar"
OUTPUT_DIR = "/content/drive/MyDrive/diplomka/debug_audio"

FEATURES_GEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bigvgan_pipeline():
    if hasattr(bigvgan.BigVGAN, '_from_pretrained'):
        orig_from_pretrained = bigvgan.BigVGAN._from_pretrained.__func__

        @classmethod
        def _patched_from_pretrained(cls, *args, **kwargs):
            kwargs.setdefault('proxies', None)
            kwargs.setdefault('resume_download', False)
            return orig_from_pretrained(cls, *args, **kwargs)

        bigvgan.BigVGAN._from_pretrained = _patched_from_pretrained

    vocoder = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x', use_cuda_kernel=False).to(DEVICE)
    vocoder.remove_weight_norm()
    vocoder.eval()
    return vocoder


def denormalize_to_log_amplitude(spectrogram):
    S_db = (spectrogram * (3.0 * DATASET_STD)) + DATASET_MEAN
    S_amp = 10.0 ** (S_db / 20.0)
    S_log = torch.log(torch.clamp(S_amp, min=1e-5))
    return S_log


def main():
    print("--- Initializing Diagnostic Inference ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[FATAL] Cannot find checkpoint at {CHECKPOINT_PATH}")
        exit(1)

    print("Loading Text Encoder, Generator, and BigVGAN...")
    text_encoder = ClapTextEncoder(TEXT_ENCODER_MODEL, device=DEVICE)
    gen = Generator(noise_dim=NOISE_DIM, text_dim=EMBEDDING_DIM, base_channels=FEATURES_GEN).to(DEVICE)
    vocoder = get_bigvgan_pipeline()

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()

    # Diagnostic test prompts mapping to DCASE classes
    test_prompts = [
        "The sound of footstep.",
        "The sound of keyboard.",
        "The sound of dog bark.",
        "The sound of gunshot.",
        "The sound of moving motor vehicle.",
        "The sound of rain.",
        "The sound of sneeze cough."
    ]

    print(f"Generating {len(test_prompts)} diagnostic samples...")
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f"Synthesizing: '{prompt}'")
            text_emb = text_encoder.encode(prompt, convert_to_tensor=True).to(DEVICE)
            noise = torch.randn(1, NOISE_DIM).to(DEVICE)

            with torch.amp.autocast('cuda'):
                generated_spec = gen(noise, text_emb)

            # SCIENTIFIC FIX: Exact precision matching to the FAD evaluation
            S_log = denormalize_to_log_amplitude(generated_spec.squeeze(1).float())
            audio_waveform = vocoder(S_log).squeeze(1).cpu()

            max_val = torch.max(torch.abs(audio_waveform))
            if max_val > 0:
                audio_waveform = audio_waveform / max_val

            # Convert 32-bit float tensor to 16-bit PCM integer
            audio_int16 = (audio_waveform * 32767.0).to(torch.int16)

            output_path = os.path.join(OUTPUT_DIR, f"epoch_525_test_{i}.wav")
            torchaudio.save(output_path, audio_int16, SAMPLE_RATE)

    print(f"\nDone. Open your Google Drive at {OUTPUT_DIR} and listen to the files.")


if __name__ == "__main__":
    main()