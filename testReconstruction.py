import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import os
import matplotlib.pyplot as plt

try:
    import bigvgan
except ImportError:
    print("Missing library: bigvgan. Please install it.")
    exit()

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
POWER = 1.0
F_MIN = 0.0
F_MAX = 8000.0

DATASET_MEAN = -19.91
DATASET_STD = 21.04

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_spectrograms(s_orig, s_gan_target, s_final_audio, filepath):
    """
    Saves a visual comparison of the original, targeted, and reconstructed spectrograms.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    im0 = axs[0].imshow(s_orig[0].cpu().numpy(), aspect='auto', origin='lower', cmap='inferno')
    axs[0].set_title("Original Audio")
    fig.colorbar(im0, ax=axs[0], format='%+2.0f dB')

    im1 = axs[1].imshow(s_gan_target[0].cpu().numpy(), aspect='auto', origin='lower', cmap='inferno')
    axs[1].set_title("Target Representation")
    fig.colorbar(im1, ax=axs[1], format='%+2.0f dB')

    im2 = axs[2].imshow(s_final_audio[0].cpu().numpy(), aspect='auto', origin='lower', cmap='inferno')
    axs[2].set_title("Final Reconstructed Output")
    fig.colorbar(im2, ax=axs[2], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def test_bigvgan_reconstruction(input_audio_path, output_audio_path):
    """
    Tests the accuracy of the spectrogram-to-audio reconstruction pipeline.
    """
    print(f"Processing input file: {input_audio_path}")

    if not os.path.exists(input_audio_path):
        print("Error: Input file missing.")
        return

    waveform, sr = torchaudio.load(input_audio_path)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    padding = int((N_FFT - HOP_LENGTH) / 2)
    waveform_padded = F.pad(waveform.unsqueeze(1), (padding, padding), mode='reflect').squeeze(1)

    mel_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        power=POWER,
        center=False,
        norm="slaney",
        mel_scale="slaney"
    ).to(DEVICE)

    to_db = AmplitudeToDB(stype='amplitude', top_db=100.0)

    waveform_padded = waveform_padded.to(DEVICE)
    s_amp = mel_transform(waveform_padded)
    s_db = to_db(s_amp)

    s_norm = (s_db - DATASET_MEAN) / (3.0 * DATASET_STD)
    s_norm = torch.clamp(s_norm, min=-1.0, max=1.0)

    s_db_reconstructed = (s_norm * (3.0 * DATASET_STD)) + DATASET_MEAN
    s_amp_reconstructed = 10.0 ** (s_db_reconstructed / 20.0)
    s_log = torch.log(torch.clamp(s_amp_reconstructed, min=1e-5))

    print("Loading vocoder...")

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

    with torch.no_grad():
        reconstructed_waveform = vocoder(s_log).squeeze(1).cpu()

    max_val = torch.max(torch.abs(reconstructed_waveform))
    if max_val > 0:
        reconstructed_waveform = reconstructed_waveform / max_val

    if reconstructed_waveform.dim() == 1:
        reconstructed_waveform = reconstructed_waveform.unsqueeze(0)

    torchaudio.save(output_audio_path, reconstructed_waveform, SAMPLE_RATE)
    print(f"Output saved: {output_audio_path}")

    reconstructed_padded = F.pad(reconstructed_waveform.unsqueeze(1), (padding, padding), mode='reflect').squeeze(1).to(DEVICE)
    s_amp_final = mel_transform(reconstructed_padded)
    s_db_final = to_db(s_amp_final)

    plot_path = output_audio_path.replace(".wav", ".png")
    plot_spectrograms(s_db.detach(), s_db_reconstructed.detach(), s_db_final.detach(), plot_path)
    print(f"Visual comparison saved: {plot_path}")


if __name__ == "__main__":
    test_bigvgan_reconstruction(
        "/srv/large-data/hasan4/sounds/DCASE2023_Task7/DCASE_2023_Challenge_Task_7_Dataset/dev/footstep/001.wav",
        "test_reconstruction_bigvgan1.wav"
    )
    test_bigvgan_reconstruction(
        "/srv/large-data/hasan4/sounds/DCASE2023_Task7/DCASE_2023_Challenge_Task_7_Dataset/dev/footstep/003.wav",
        "test_reconstruction_bigvgan3.wav"
    )