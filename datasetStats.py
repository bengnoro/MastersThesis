import torch
import torchaudio
import numpy as np
import glob
import os

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
POWER = 1.0
CENTER = True
F_MIN = 0.0
F_MAX = 8000.0

DUMMY_AUDIO_DIR = "/srv/large-data/hasan4/sounds/DCASE2023_Task7/DCASE_2023_Challenge_Task_7_Dataset/dev"


def calculate_stats(num_files=2000):
    """
    Computes the mean and standard deviation of decibel values across a subset of the audio dataset.
    """
    print(f"Scanning up to {num_files} files...")

    search_path = os.path.join(DUMMY_AUDIO_DIR, '**', '*.wav')
    files = glob.glob(search_path, recursive=True)[:num_files]

    if not files:
        print(f"Error: Directory is empty or path is incorrect ({DUMMY_AUDIO_DIR}).")
        return

    mel_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=POWER, center=CENTER
    )
    to_db = AmplitudeToDB(stype='amplitude', top_db=80.0)

    all_db_values = []

    for file_path in files:
        try:
            wav, sr = torchaudio.load(file_path)
            if sr != SAMPLE_RATE:
                wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            s_amp = mel_transform(wav)
            s_db = to_db(s_amp)

            all_db_values.append(s_db.flatten().numpy())
        except Exception:
            pass

    if not all_db_values:
        return

    global_db_array = np.concatenate(all_db_values)

    mean_val = np.mean(global_db_array)
    std_val = np.std(global_db_array)

    p1 = np.percentile(global_db_array, 1)
    p99 = np.percentile(global_db_array, 99)

    print("\nDataset Decibel Statistics:")
    print(f"Mean (µ):              {mean_val:.2f} dB")
    print(f"Standard Deviation (σ):{std_val:.2f} dB")
    print(f"1st Percentile:        {p1:.2f} dB")
    print(f"99th Percentile:       {p99:.2f} dB")
    print("\nSuggested updates for configuration:")
    print(f"DATASET_MEAN = {mean_val:.2f}")
    print(f"DATASET_STD = {std_val:.2f}")


if __name__ == "__main__":
    calculate_stats(3400)