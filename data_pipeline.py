import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import glob
import re
import warnings

try:
    from transformers import AutoTokenizer, ClapTextModelWithProjection
except ImportError:
    print("Please install transformers: pip install transformers")

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
TARGET_TIME_STEPS = 512
CENTER = True
F_MIN = 0.0
F_MAX = 8000.0
# from test dataset stats
DATASET_MEAN = -19.91
DATASET_STD = 21.04
TEXT_ENCODER_MODEL = 'laion/clap-htsat-unfused'
EMBEDDING_DIM = 512

DUMMY_AUDIO_DIR = "/srv/large-data/hasan4/sounds/DCASE2023_Task7/DCASE_2023_Challenge_Task_7_Dataset/dev"
DUMMY_CSV_FILE = "/srv/large-data/hasan4/sounds/DCASE2023_Task7/DCASE_2023_Challenge_Task_7_Dataset/DevMeta.csv"


class ClapTextEncoder:
    def __init__(self, model_name=TEXT_ENCODER_MODEL, device="cpu"):
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ClapTextModelWithProjection.from_pretrained(model_name, use_safetensors=True).to(device)
        self.device = device
        self.model.eval()

    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.text_embeds


class AudioTextDataset(Dataset):
    def __init__(self, csv_file, audio_dir, sample_rate, n_fft, hop_length, n_mels, target_time_steps,
                 text_encoder=None):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_time_steps = target_time_steps

        if not os.path.exists(csv_file):
            print(f"Warning: Exact CSV path not found. Searching dynamically in {audio_dir}...")
            search_path = os.path.join(audio_dir, '**', '*.csv')
            found_csvs = glob.glob(search_path, recursive=True)

            metadata_csvs = [f for f in found_csvs if 'metadata' in f.lower() or 'devmeta' in f.lower()]
            if metadata_csvs:
                csv_file = metadata_csvs[0]
            elif found_csvs:
                csv_file = found_csvs[0]
            else:
                raise FileNotFoundError(f"Missing CSV: Could not locate any .csv files in {audio_dir}")

            print(f"Dynamically locked onto CSV: {csv_file}")

        self.captions_df = pd.read_csv(csv_file).reset_index(drop=True)
        print(f"DCASE Dataset Loaded. Processing {len(self.captions_df)} high-fidelity samples.")

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=F_MIN, f_max=F_MAX, power=1.0, center=CENTER
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80.0)

        self.embedding_cache = {}
        if text_encoder is not None:
            print("Pre-computing static text embeddings for ALL DCASE classes...")
            cat_col = 'category' if 'category' in self.captions_df.columns else 'class' if 'class' in self.captions_df.columns else None

            if cat_col:
                unique_classes = self.captions_df[cat_col].dropna().unique()
                for cat in unique_classes:
                    raw_labels = str(cat).replace('_', ' ').lower()
                    caption = f"The sound of {raw_labels}."
                    emb = text_encoder.encode(caption, convert_to_tensor=True).cpu()
                    self.embedding_cache[cat] = emb

    def __len__(self):
        return len(self.captions_df)

    def normalize_for_gan(self, S_db):
        S_norm = (S_db - DATASET_MEAN) / (3.0 * DATASET_STD)
        return torch.clamp(S_norm, min=-1.0, max=1.0)

    def __getitem__(self, idx):
        try:
            row = self.captions_df.iloc[idx]

            filename = row.get('filename', row.get('file', row.get('current_file_path', None)))
            category = row.get('category', row.get('class', None))

            if filename is None or category is None:
                return None, None, None

            base_filename = os.path.basename(str(filename))

            audio_path = os.path.join(self.audio_dir, str(category), base_filename)
            if not os.path.exists(audio_path):
                audio_path = os.path.join(self.audio_dir, base_filename)
                if not os.path.exists(audio_path):
                    search_pattern = os.path.join(self.audio_dir, '**', base_filename)
                    found_files = glob.glob(search_pattern, recursive=True)
                    if found_files:
                        audio_path = found_files[0]
                    else:
                        return None, None, None

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

            S_amp = self.mel_spectrogram(waveform)
            S_db = self.amplitude_to_db(S_amp)
            spectrogram = self.normalize_for_gan(S_db)

            if spectrogram.dim() == 2:
                spectrogram = spectrogram.unsqueeze(0)

            n_time_steps = spectrogram.shape[2]

            if n_time_steps < self.target_time_steps:
                padding = self.target_time_steps - n_time_steps
                pad_left = torch.randint(0, padding + 1, (1,)).item()
                pad_right = padding - pad_left

                if padding < n_time_steps:
                    spectrogram = torch.nn.functional.pad(spectrogram, (pad_left, pad_right), mode='reflect')
                else:
                    spectrogram = torch.nn.functional.pad(spectrogram, (pad_left, pad_right), mode='replicate')

            elif n_time_steps > self.target_time_steps:
                max_start = n_time_steps - self.target_time_steps
                start_idx = torch.randint(0, max_start + 1, (1,)).item()
                spectrogram = spectrogram[:, :, start_idx:start_idx + self.target_time_steps]

            raw_labels = str(category).replace('_', ' ').lower()
            caption = f"The sound of {raw_labels}."

            emb = self.embedding_cache.get(category, None)

            return spectrogram, caption, emb

        except Exception:
            return None, None, None


def collate_fn(batch):
    valid_batch = [item for item in batch if item[0] is not None]
    if len(valid_batch) == 0: return None, None, None
    spectrograms, captions, embs = zip(*valid_batch)

    if embs[0] is not None:
        return torch.stack(spectrograms), list(captions), torch.stack(embs).squeeze(1)
    return torch.stack(spectrograms), list(captions), None


if __name__ == "__main__":
    print("Testing DCASE 2023 Data Pipeline...")
    dataset = AudioTextDataset(
        csv_file=DUMMY_CSV_FILE,
        audio_dir=DUMMY_AUDIO_DIR,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        target_time_steps=TARGET_TIME_STEPS
    )

    total_files = len(dataset)
    print(f"\nTotal verified files loaded: {total_files}")

    if total_files > 0:
        spec, cap, emb = dataset[44]
        print(f"Sample 44 Caption: '{cap}'")
        if spec is not None:
            print(f"Sample 44 Spectrogram Shape: {spec.shape}")
            print(f"Sample 44 Value Range: [{spec.min():.2f}, {spec.max():.2f}]")