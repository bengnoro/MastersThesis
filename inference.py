import os
import warnings

import torch
import torchaudio
import librosa
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import bigvgan
except ImportError:
    print("Missing library: bigvgan. Please install it.")
    exit()

warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from models import Generator, NOISE_DIM
from data_pipeline import (
    TEXT_ENCODER_MODEL, EMBEDDING_DIM, SAMPLE_RATE,
    ClapTextEncoder, DATASET_MEAN, DATASET_STD
)

CHECKPOINT_PATH = "/content/drive/MyDrive/diplomka/checkpoints/gen_epoch_320.pth.tar"
OUTPUT_DIR = "/content/drive/MyDrive/diplomka/debug_audio_bigvganFinal"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRUNCATION_PSI = 0.7


def get_bigvgan_pipeline():
    """
    Initializes and returns the BigVGAN vocoder model.
    """
    if hasattr(bigvgan.BigVGAN, '_from_pretrained'):
        orig_from_pretrained = bigvgan.BigVGAN._from_pretrained.__func__

        @classmethod
        def _patched_from_pretrained(cls, *args, **kwargs):
            kwargs.setdefault('proxies', None)
            kwargs.setdefault('resume_download', False)
            return orig_from_pretrained(cls, *args, **kwargs)

        bigvgan.BigVGAN._from_pretrained = _patched_from_pretrained

    vocoder = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x', use_cuda_kernel=False).to(DEVICE)
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder


def denormalize_to_log_amplitude(s_norm):
    """
    Reverses the normalization process to retrieve the log-scaled amplitude spectrogram.
    """
    s_db = (s_norm * (3.0 * DATASET_STD)) + DATASET_MEAN
    s_amp = 10.0 ** (s_db / 20.0)
    s_log = torch.log(torch.clamp(s_amp, min=1e-5))
    return s_log


def trim_silence(waveform, top_db=40):
    """
    Removes silent segments from the beginning and end of the audio file.
    """
    wav_np = waveform.squeeze().cpu().numpy()
    trimmed, _ = librosa.effects.trim(wav_np, top_db=top_db)
    return torch.from_numpy(trimmed).unsqueeze(0).unsqueeze(0)


def save_spectrogram(tensor, output_path, title):
    """
    Converts the spectrogram tensor to an image and saves it to disk.
    """
    spec_np = tensor.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec_np, sr=SAMPLE_RATE, hop_length=256, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def apply_smooth_cleaning(s_norm, threshold=-0.80, contrast=1.05, hf_bins=15):
    """
    Adjusts contrast and applies a high-frequency roll-off to clean the generated spectrogram.
    """
    s_expanded = (s_norm - threshold) * contrast + threshold
    s_expanded = torch.clamp(s_expanded, min=-1.0, max=1.0)

    hf_floor = -1.0
    freq_mask = torch.ones(80, 1, device=s_norm.device)
    start_bin = 80 - hf_bins
    freq_mask[start_bin:80, 0] = torch.cos(torch.linspace(0, 1.5708, steps=hf_bins, device=s_norm.device))

    s_rolled = (s_expanded - hf_floor) * freq_mask.view(1, 1, 80, 1) + hf_floor
    s_final = torch.clamp(s_rolled, min=-0.98, max=1.0)

    return s_final


def apply_micro_dither(s_norm, noise_level=0.015):
    """
    Injects a small amount of random noise to mask grid-like artifacts in the spectrogram.
    """
    dither = torch.randn_like(s_norm) * noise_level
    s_dithered = s_norm + dither
    return torch.clamp(s_dithered, min=-1.0, max=1.0)


def apply_hpss_mastering(audio_tensor):
    """
    Separates harmonic and percussive elements to reduce phase issues, then recombines them.
    """
    y = audio_tensor.squeeze().cpu().numpy()
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=2.0)
    y_mastered = y_percussive + (0.85 * y_harmonic)
    return torch.from_numpy(y_mastered).unsqueeze(0)


def run_bigvgan_hybrid_inference():
    """
    Executes the full generation pipeline to synthesize audio from text prompts using BigVGAN.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading Generator model...")

    text_encoder = ClapTextEncoder(TEXT_ENCODER_MODEL, device=DEVICE)
    gen = Generator(noise_dim=NOISE_DIM, text_dim=EMBEDDING_DIM).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()

    print("Loading BigVGAN V2 Vocoder...")
    vocoder = get_bigvgan_pipeline()

    test_prompts = [
        "The sound of dog bark.",
        "The sound of footstep.",
        "The sound of gunshot.",
        "The sound of moving motor vehicle.",
        "The sound of keyboard.",
        "The sound of rain."
    ]

    print("Synthesizing audio samples...")
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f"Processing prompt: {prompt}")
            text_emb = text_encoder.encode([prompt], convert_to_tensor=True).to(DEVICE)

            noise = torch.randn(1, NOISE_DIM).to(DEVICE)
            noise = torch.clamp(noise, min=-TRUNCATION_PSI, max=TRUNCATION_PSI)

            generated_spec = gen(noise, text_emb)
            cleaned_spec = apply_smooth_cleaning(generated_spec)
            dithered_spec = apply_micro_dither(cleaned_spec, noise_level=0.015)

            s_log = denormalize_to_log_amplitude(dithered_spec.squeeze(1).float())

            spec_output_path = os.path.join(OUTPUT_DIR, f"test_sample_{i+1}_{prompt.replace(' ', '_')}_bigvgan_spec.png")
            save_spectrogram(s_log, spec_output_path, f"BigVGAN Input: {prompt}")

            with torch.amp.autocast('cuda'):
                audio_waveform = vocoder(s_log).cpu()

            audio_waveform = audio_waveform.float()
            audio_waveform = trim_silence(audio_waveform, top_db=40)
            audio_waveform = apply_hpss_mastering(audio_waveform)

            max_val = torch.max(torch.abs(audio_waveform))
            if max_val > 0:
                audio_waveform = audio_waveform / max_val

            headroom_factor = 0.85
            audio_waveform = audio_waveform * headroom_factor

            output_path = os.path.join(OUTPUT_DIR, f"test_sample_{i+1}_{prompt.replace(' ', '_')}_bigvgan_hybrid.wav")
            torchaudio.save(output_path, audio_waveform.squeeze(0), sample_rate=22050)

    print(f"Process complete. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_bigvgan_hybrid_inference()