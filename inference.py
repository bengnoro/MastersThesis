import torch
import torchaudio
import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)

from sentence_transformers import SentenceTransformer
from vocos import Vocos
from models import Generator, NOISE_DIM
from data_pipeline import TEXT_ENCODER_MODEL, EMBEDDING_DIM, SAMPLE_RATE

# config
CHECKPOINT_PATH = "checkpoints/gen_epoch_995.pth.tar"
OUTPUT_FOLDER = "generated_samples_vocos"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_GEN = 512


def denormalize_for_vocos(spectrogram, noise_floor=-11.5):
    """
    Direct linear un-mapping from GAN [-1, 1] back to Vocos
    native Log-Magnitude bounds [-11.5, 4.5].
    Must remain strictly linear to preserve the spectral envelope for the neural vocoder.
    """
    S_vocos_log_mel = (spectrogram * 8.0) - 3.5
    S_vocos_log_mel = torch.clamp(S_vocos_log_mel, min=noise_floor, max=4.5)
    return S_vocos_log_mel


def generate_sound(text_prompt, generator, text_encoder, vocoder):
    print(f"Generating high-fidelity sound for: '{text_prompt}'...")

    with torch.no_grad():
        text_emb = text_encoder.encode(text_prompt, convert_to_tensor=True).to(DEVICE)
        text_emb = text_emb.unsqueeze(0)
        noise = torch.randn(1, NOISE_DIM).to(DEVICE)

        generator.eval()
        generated_spec = generator(noise, text_emb)
        S_vocos = denormalize_for_vocos(generated_spec.squeeze(1))
        audio_waveform = vocoder.decode(S_vocos)
        max_val = torch.max(torch.abs(audio_waveform))
        if max_val > 0:
            audio_waveform = audio_waveform / max_val
        return audio_waveform.cpu()


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    text_encoder = SentenceTransformer(TEXT_ENCODER_MODEL).to(DEVICE)
    gen = Generator(noise_dim=NOISE_DIM, text_dim=EMBEDDING_DIM, base_channels=FEATURES_GEN).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        gen.load_state_dict(checkpoint["state_dict"])
    else:
        print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found.")
        return

    gen.eval()
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(DEVICE)
    test_prompts = [
        "footsteps",
        "door wood knock",
        "running on grass",
        "knocking",
        "glass breaking"
    ]

    for i, prompt in enumerate(test_prompts):
        waveform = generate_sound(prompt, gen, text_encoder, vocoder)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(0)

        torchaudio.save(os.path.join(OUTPUT_FOLDER, f"sample_{i}.wav"), waveform, SAMPLE_RATE)

    print(f"\nGeneration complete. Check the '{OUTPUT_FOLDER}' folder.")


if __name__ == "__main__":
    main()