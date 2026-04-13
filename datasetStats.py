import torch
import torchaudio
import numpy as np
import glob
import os

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

SAMPLE_RATE = 22050  # DCASE
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
POWER = 1.0
CENTER = True
F_MIN = 0.0
F_MAX = 8000.0

DUMMY_AUDIO_DIR = "/srv/large-data/hasan4/sounds/DCASE2023_Task7/DCASE_2023_Challenge_Task_7_Dataset/dev"


def calculate_stats(num_files=2000):
    print(f"Calculating dB dataset statistics (mean and std) for up to {num_files} DCASE files...")

    search_path = os.path.join(DUMMY_AUDIO_DIR, '**', '*.wav')
    files = glob.glob(search_path, recursive=True)[:num_files]

    if not files:
        print(f"Error: No .wav files found in {DUMMY_AUDIO_DIR} or its subdirectories. Check your extraction path.")
        return

    mel_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=POWER, center=CENTER
    )
    to_db = AmplitudeToDB(stype='amplitude', top_db=80.0)

    all_db_values = []

    for f in files:
        try:
            wav, sr = torchaudio.load(f)
            if sr != SAMPLE_RATE:
                wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            S_amp = mel_transform(wav)
            S_db = to_db(S_amp)

            all_db_values.append(S_db.flatten().numpy())
        except Exception:
            pass

    if not all_db_values:
        return

    global_db_array = np.concatenate(all_db_values)

    mu = np.mean(global_db_array)
    sigma = np.std(global_db_array)

    p1 = np.percentile(global_db_array, 1)
    p99 = np.percentile(global_db_array, 99)

    print("\nGlobal Decibel (dB) Spectrogram Stats")
    print(f"Mean (µ):              {mu:.2f} dB")
    print(f"Standard Deviation (σ):{sigma:.2f} dB")
    print(f"P1 (1st Percentile):   {p1:.2f} dB")
    print(f"P99 (99th Percentile): {p99:.2f} dB")
    print("-------------------------------------------------")
    print(f"DATASET_MEAN = {mu:.2f}")
    print(f"DATASET_STD = {sigma:.2f}")


if __name__ == "__main__":
    calculate_stats(2000)
