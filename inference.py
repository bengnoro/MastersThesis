import torch
import torchaudio
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# --- Local Imports ---
from models import Generator, NOISE_DIM
from data_pipeline import TEXT_ENCODER_MODEL, EMBEDDING_DIM, N_MELS, TARGET_TIME_STEPS

# --- Configuration ---
CHECKPOINT_PATH = "checkpoints/gen_epoch_5.pth.tar"  # <--- UPDATE THIS if needed
OUTPUT_FOLDER = "generated_samples"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MUST match train.py settings
FEATURES_GEN = 64


def load_vocoder():
    """
    Loads a pre-trained HiFi-GAN vocoder from TorchAudio.
    This converts Mel-Spectrograms -> Audio Waveforms.
    The settings (22050Hz, 80 Mels) must match our training data exactly.
    """
    print("Loading HiFi-GAN Vocoder...")

    # Attempt to load the HiFiGAN bundle.
    # In newer versions of torchaudio, this is often in 'prototype'.
    bundle = None
    try:
        # Try standard location
        bundle = torchaudio.pipelines.HIFIGAN_VOCODER_V3_LJSPEECH
    except AttributeError:
        try:
            # Try prototype location (common in newer TorchAudio)
            from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH
            bundle = HIFIGAN_VOCODER_V3_LJSPEECH
        except ImportError:
            raise AttributeError(
                "Could not load HiFiGAN bundle. Ensure torchaudio is installed and supports prototype pipelines.")

    vocoder = bundle.get_vocoder().to(DEVICE)
    return vocoder


def generate_sound(text_prompt, generator, text_encoder, vocoder):
    """
    Runs the full inference pipeline: Text -> Embedding -> Spectrogram -> Audio.
    """
    print(f"Generating sound for prompt: '{text_prompt}'...")

    # 1. Encode Text
    with torch.no_grad():
        # Encode and reshape to [1, 384]
        text_emb = text_encoder.encode(text_prompt, convert_to_tensor=True)
        text_emb = text_emb.unsqueeze(0).to(DEVICE)

        # 2. Generate Noise
        # Shape: [1, 100]
        noise = torch.randn(1, NOISE_DIM).to(DEVICE)

        # 3. Generate Mel-Spectrogram
        generator.eval()
        # Shape: [1, 1, 80, 512]
        generated_spec = generator(noise, text_emb)

        # 4. Vocode (Spectrogram -> Waveform)
        # HiFi-GAN expects input shape [Batch, 80, Time]
        # Our generator outputs [Batch, 1, 80, Time], so we squeeze dimension 1
        spec_for_vocoder = generated_spec.squeeze(1)

        # Run Vocoder
        audio_waveform = vocoder(spec_for_vocoder)

        return audio_waveform.cpu(), generated_spec.cpu()


def save_audio(waveform, sample_rate, filename):
    # Vocoder outputs [Batch, Channels, Time] -> [1, 1, 10000]
    # torchaudio.save expects [Channels, Time] -> [1, 10000]
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)

    path = os.path.join(OUTPUT_FOLDER, filename)
    torchaudio.save(path, waveform, sample_rate)
    print(f"Saved audio: {path}")


def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 1. Load Text Encoder
    print("Loading Text Encoder...")
    text_encoder = SentenceTransformer(TEXT_ENCODER_MODEL)

    # 2. Load Generator
    print("Loading Generator...")
    # FIX: Added base_channels=FEATURES_GEN to match training configuration
    gen = Generator(noise_dim=NOISE_DIM, text_dim=EMBEDDING_DIM, base_channels=FEATURES_GEN).to(DEVICE)

    # Load Weights
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()  # Important! Switch to inference mode
    print("Checkpoint loaded.")

    # 3. Load Vocoder
    vocoder = load_vocoder()

    # 4. Define Prompts to Test
    test_prompts = [
        "footsteps on gravel",
        "knocking on a wooden door",
        "a computer fan humming"
    ]

    # 5. Run Inference
    for i, prompt in enumerate(test_prompts):
        waveform, spec = generate_sound(prompt, gen, text_encoder, vocoder)

        # Save .wav
        safe_filename = f"sample_{i}_{prompt.replace(' ', '_')}.wav"
        save_audio(waveform, 22050, safe_filename)

    print("\nDone! Check the 'generated_samples' folder.")
    print("Note: Since this model was trained on dummy data for few epochs,")
    print("the audio will likely sound like static noise. This confirms the PIPELINE works.")


if __name__ == "__main__":
    main()