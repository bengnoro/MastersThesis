import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import os

try:
    import bigvgan
except ImportError:
    print("Please install NVIDIA BigVGAN: `pip install bigvgan`")
    exit()

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
POWER = 1.0
CENTER = True
F_MIN = 0.0
F_MAX = 8000.0

DATASET_MEAN = -19.91
DATASET_STD = 21.04

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_bigvgan_reconstruction(input_audio_path, output_audio_path):
    print(f"Loading {input_audio_path}...")

    if not os.path.exists(input_audio_path):
        print(f"Error: File {input_audio_path} not found.")
        return

    waveform, sr = torchaudio.load(input_audio_path)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    mel_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=POWER, center=CENTER
    )
    to_db = AmplitudeToDB(stype='amplitude', top_db=80.0)

    S_amp = mel_transform(waveform)
    S_db = to_db(S_amp)

    S_norm = (S_db - DATASET_MEAN) / (3.0 * DATASET_STD)
    S_norm = torch.clamp(S_norm, min=-1.0, max=1.0)
    S_db_reconstructed = (S_norm * (3.0 * DATASET_STD)) + DATASET_MEAN
    S_amp_reconstructed = 10.0 ** (S_db_reconstructed / 20.0)
    S_log = torch.log(torch.clamp(S_amp_reconstructed, min=1e-5)).to(DEVICE)

    print("Loading vocoder")
    vocoder = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x', use_cuda_kernel=False).to(DEVICE)
    vocoder.remove_weight_norm()
    vocoder.eval()

    with torch.no_grad():
        reconstructed_waveform = vocoder(S_log).squeeze(1).cpu()

    max_val = torch.max(torch.abs(reconstructed_waveform))
    if max_val > 0: reconstructed_waveform = reconstructed_waveform / max_val

    if reconstructed_waveform.dim() == 1:
        reconstructed_waveform = reconstructed_waveform.unsqueeze(0)

    torchaudio.save(output_audio_path, reconstructed_waveform, SAMPLE_RATE)
    print(f"reconstruction saved to {output_audio_path}")


if __name__ == "__main__":
    # Point this to a single real DCASE audio file on your server for the test
    test_bigvgan_reconstruction(
        "/srv/large-data/hasan4/sounds/DCASE_2023_Challenge_Task_7_Dataset/dev/Footstep/001.wav",
        "test_reconstruction_bigvgan.wav")
