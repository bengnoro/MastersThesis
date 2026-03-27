import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

SAMPLE_RATE = 24000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 100
TARGET_TIME_STEPS = 512
TEXT_ENCODER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

DUMMY_AUDIO_DIR = "/srv/large-data/hasan4/sounds/FSD50K/FSD50K_dev_audio"
DUMMY_CSV_FILE = "/srv/large-data/hasan4/sounds/FSD50K/FSD50K_ground_truth/dev.csv"
TARGET_PATTERN = "Footsteps|Knock|Wood|Glass|Creak|Shatter" #regex to what to search for in fsdk sound library


class AudioTextDataset(Dataset):
    def __init__(self, csv_file, audio_dir, sample_rate, n_fft, hop_length, n_mels, target_time_steps):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_time_steps = target_time_steps

        if not os.path.exists(csv_file): raise FileNotFoundError(f"Missing CSV: {csv_file}")

        full_df = pd.read_csv(csv_file)
        self.captions_df = full_df[full_df['labels'].str.contains(TARGET_PATTERN, case=False, na=False)].reset_index(
            drop=True)
        print(f"Dataset Loaded. Filtered {len(self.captions_df)} samples.")

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=0, f_max=sample_rate // 2, power=1.0
        )

    def __len__(self):
        return len(self.captions_df)

    def normalize_for_gan(self, S_log_magnitude):
        S_norm = (S_log_magnitude + 3.5) / 8.0
        S_norm = torch.clamp(S_norm, min=-1.0, max=1.0)
        return S_norm

    def __getitem__(self, idx):
        try:
            row = self.captions_df.iloc[idx]
            audio_filename = f"{row['fname']}.wav"
            audio_path = os.path.join(self.audio_dir, audio_filename)

            if not os.path.exists(audio_path):
                return None, None

            waveform, original_sr = torchaudio.load(audio_path)

            if original_sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(original_sr, self.sample_rate)(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            max_abs_val = torch.max(torch.abs(waveform))
            if max_abs_val > 0: waveform = waveform / max_abs_val

            if torch.rand(1).item() < 0.5:
                gain = torch.empty(1).uniform_(0.5, 1.0).item()
                waveform = waveform * gain

            if torch.rand(1).item() < 0.5:
                shift_amt = torch.randint(0, waveform.shape[1], (1,)).item()
                waveform = torch.roll(waveform, shifts=shift_amt, dims=1)

            spectrogram = self.mel_spectrogram(waveform)
            S_log = torch.log(torch.clamp(spectrogram, min=1e-5))
            spectrogram = self.normalize_for_gan(S_log)

            n_time_steps = spectrogram.shape[2]
            if n_time_steps < self.target_time_steps:
                padding = self.target_time_steps - n_time_steps
                spectrogram = torch.nn.functional.pad(spectrogram, (0, padding), 'constant', -1.0)
            else:
                spectrogram = spectrogram[:, :, :self.target_time_steps]

            if spectrogram.dim() == 2: spectrogram = spectrogram.unsqueeze(0)

            caption = str(row['labels']).replace(',', ' ')
            return spectrogram, caption

        except Exception as e:
            return None, None


def collate_fn(batch):
    valid_batch = [item for item in batch if item[0] is not None]
    if len(valid_batch) == 0: return None, None
    spectrograms, captions = zip(*valid_batch)
    return torch.stack(spectrograms), list(captions)