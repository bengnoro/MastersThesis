import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

# config
SAMPLE_RATE = 24000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 100
TARGET_TIME_STEPS = 512
TEXT_ENCODER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
DUMMY_AUDIO_DIR = "/srv/large-data/hasan4/sounds/ESC-50-master/audio"
DUMMY_CSV_FILE = "/srv/large-data/hasan4/sounds/ESC-50-master/meta/esc50.csv"

# keywords to search for
TARGET_CATEGORIES = [
    "footsteps", "door_wood_knock", "door_wood_creaks",
    "glass_breaking", "keyboard_typing", "clock_tick",
    "water_drops", "wind"
]


class AudioTextDataset(Dataset):
    def __init__(self, csv_file, audio_dir, sample_rate, n_fft, hop_length, n_mels, target_time_steps):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_time_steps = target_time_steps

        if not os.path.exists(csv_file): raise FileNotFoundError(f"Missing CSV: {csv_file}")

        full_df = pd.read_csv(csv_file)
        self.captions_df = full_df[full_df['category'].isin(TARGET_CATEGORIES)].reset_index(drop=True)
        print(f"Dataset Loaded. {len(self.captions_df)} clean ESC-50 samples.")

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=0, f_max=sample_rate // 2
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def __len__(self):
        return len(self.captions_df)

    def normalize(self, S):
        S = torch.clamp(S, min=-80.0, max=0.0)
        S = S + 80.0
        S = S / 80.0
        S = (S * 2.0) - 1.0
        return S

    def __getitem__(self, idx):
        try:
            row = self.captions_df.iloc[idx]
            audio_path = os.path.join(self.audio_dir, str(row['filename']))

            if not os.path.exists(audio_path): return None, None
            waveform, original_sr = torchaudio.load(audio_path)
            if original_sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(original_sr, self.sample_rate)(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            max_abs_val = torch.max(torch.abs(waveform))
            if max_abs_val > 0: waveform = waveform / max_abs_val

            spectrogram = self.mel_spectrogram(waveform)
            spectrogram = self.amplitude_to_db(spectrogram)
            spectrogram = self.normalize(spectrogram)

            n_time_steps = spectrogram.shape[2]
            if n_time_steps < self.target_time_steps:
                padding = self.target_time_steps - n_time_steps
                spectrogram = torch.nn.functional.pad(spectrogram, (0, padding), 'constant', -1.0)
            else:
                spectrogram = spectrogram[:, :, :self.target_time_steps]

            if spectrogram.dim() == 2: spectrogram = spectrogram.unsqueeze(0)
            caption = str(row['category']).replace('_', ' ')
            return spectrogram, caption

        except Exception as e:
            return None, None


def collate_fn(batch):
    valid_batch = [item for item in batch if item[0] is not None]
    if len(valid_batch) == 0: return None, None
    spectrograms, captions = zip(*valid_batch)
    return torch.stack(spectrograms), list(captions)