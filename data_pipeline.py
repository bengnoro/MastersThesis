import torch
import torchaudio
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import os

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
TARGET_TIME_STEPS = 512

TEXT_ENCODER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

DUMMY_AUDIO_DIR = "sounds/audiocaps_raw_audio"
DUMMY_CSV_FILE = "sounds/train.csv"

KEYWORDS = [
    "footstep", "walk", "run", "jog", "sprint", "hike",
    "knock", "tap", "bang", "hit", "thud", "slam",
    "snow", "sand", "gravel", "grass", "concrete", "wood", "floor",
    "creak", "rustle", "slide", "scrape"
]


class AudioTextDataset(Dataset):
    def __init__(self, csv_file, audio_dir, text_encoder, sample_rate, n_fft, hop_length, n_mels, target_time_steps):
        self.audio_dir = audio_dir
        self.text_encoder = text_encoder
        self.sample_rate = sample_rate
        self.target_time_steps = target_time_steps
        self.n_mels = n_mels

        try:
            full_df = pd.read_csv(csv_file, sep=None, engine='python')
            full_df.columns = full_df.columns.str.strip().str.lower()
            pattern = '|'.join(KEYWORDS)
            self.captions_df = full_df[full_df['caption'].str.contains(pattern, case=False, na=False)].reset_index(
                drop=True)
            print(f"Dataset Loaded. Filtered from {len(full_df)} total to {len(self.captions_df)} relevant samples.")
        except Exception as e:
            print(f"CRITICAL ERROR loading CSV: {e}")
            raise e

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def __len__(self):
        return len(self.captions_df)

    def normalize(self, S):
        """
        Normalizes a dB spectrogram to [-1, 1].
        Assumes input is in range [-80, 0] (approx).
        """
        # 1. Clip values to be between -80 and 0
        S = torch.clamp(S, min=-80.0, max=0.0)
        # 2. Shift to [0, 80]
        S = S + 80.0
        # 3. Scale to [0, 1]
        S = S / 80.0
        # 4. Scale to [-1, 1]
        S = (S * 2.0) - 1.0
        return S

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            row = self.captions_df.iloc[idx]
            youtube_id = str(row['youtube_id']).strip()

            raw_start_time = str(row['start_time']).strip()

            try:
                start_time = str(int(float(raw_start_time)))
            except ValueError:
                start_time = raw_start_time
                if start_time.endswith('.0'):
                    start_time = start_time[:-2]

            audio_filename = f"{youtube_id}_{start_time}.wav"
            caption = str(row['caption'])
            audio_path = os.path.join(self.audio_dir, audio_filename)

            if not os.path.exists(audio_path):
                return None, None

            waveform, original_sr = torchaudio.load(audio_path)

            if original_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(original_sr, self.sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            max_abs_val = torch.max(torch.abs(waveform))
            if max_abs_val > 0:
                waveform = waveform / max_abs_val

            spectrogram = self.mel_spectrogram(waveform)
            spectrogram = self.amplitude_to_db(spectrogram)

            spectrogram = self.normalize(spectrogram)
            # ---------------------------------

            n_time_steps = spectrogram.shape[2]
            if n_time_steps < self.target_time_steps:
                padding = self.target_time_steps - n_time_steps
                spectrogram = torch.nn.functional.pad(spectrogram, (0, padding), 'constant', -1.0)
            else:
                spectrogram = spectrogram[:, :, :self.target_time_steps]

            if spectrogram.dim() == 2:
                spectrogram = spectrogram.unsqueeze(0)

            with torch.no_grad():
                text_embedding = self.text_encoder.encode(caption, convert_to_tensor=True)
                text_embedding = text_embedding.cpu()

            if len(text_embedding.shape) > 1:
                text_embedding = text_embedding.squeeze()

            return spectrogram, text_embedding

        except Exception as e:
            return None, None


def collate_fn(batch):
    valid_batch = [item for item in batch if item[0] is not None]
    if len(valid_batch) == 0:
        return None, None
    return torch.utils.data.dataloader.default_collate(valid_batch)