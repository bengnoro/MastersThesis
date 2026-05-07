import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import glob
import warnings
import random

try:
    from transformers import AutoTokenizer, ClapTextModelWithProjection
except ImportError:
    print("Missing library: transformers. Please install it.")

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
TARGET_TIME_STEPS = 512
CENTER = True
F_MIN = 0.0
F_MAX = 8000.0

DATASET_MEAN = -13.18
DATASET_STD = 20.96

TEXT_ENCODER_MODEL = 'laion/clap-htsat-unfused'
EMBEDDING_DIM = 512

DUMMY_AUDIO_DIR = "/content/DCASE_2023_Challenge_Task_7_Dataset/dev"
DUMMY_CSV_FILE = "/content/DCASE_2023_Challenge_Task_7_Dataset/DevMeta.csv"
TARGET_WAVE_LENGTH = TARGET_TIME_STEPS * HOP_LENGTH


class ClapTextEncoder:
    """
    Handles text tokenization and embedding generation using a pretrained CLAP model.
    """

    def __init__(self, model_name=TEXT_ENCODER_MODEL, device="cpu"):
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ClapTextModelWithProjection.from_pretrained(model_name, use_safetensors=True).to(device)
        self.device = device
        self.model.eval()

    def encode(self, texts, convert_to_tensor=True):
        """
        Converts raw text strings into numerical embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.text_embeds


class AudioTextDataset(Dataset):
    """
    Dataset loader for reading audio files and corresponding text captions.
    """

    def __init__(self, csv_file, audio_dir, sample_rate, n_fft, hop_length, n_mels, target_time_steps, text_encoder=None):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_time_steps = target_time_steps

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Missing CSV at {csv_file}.")

        raw_df = pd.read_csv(csv_file)

        print("Building file index...")
        all_wavs = glob.glob(os.path.join(self.audio_dir, '**', '*.wav'), recursive=True)
        all_wavs += glob.glob(os.path.join(self.audio_dir, '**', '*.WAV'), recursive=True)

        self.file_index = {}
        for file_path in all_wavs:
            parts = file_path.replace('\\', '/').split('/')
            if len(parts) >= 2:
                key = f"{parts[-2]}/{parts[-1]}".lower()
                self.file_index[key] = file_path

        print("Filtering missing files from metadata...")
        valid_rows = []
        for _, row in raw_df.iterrows():
            file_str = str(row.get('current_file_path', row.get('filename', ''))).replace('\\', '/')
            category = row.get('category', row.get('class', None))

            if not file_str or pd.isna(category):
                continue

            parts = file_str.split('/')
            search_key = f"{parts[-2]}/{parts[-1]}".lower() if len(parts) >= 2 else f"{category}/{parts[-1]}".lower()

            if search_key in self.file_index:
                valid_rows.append(row)

        self.captions_df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"Dataset ready. Valid pairs loaded: {len(self.captions_df)}")

        self.embedding_cache = {}
        if text_encoder is not None:
            cat_col = 'category' if 'category' in self.captions_df.columns else 'class'
            if cat_col in self.captions_df.columns:
                unique_classes = self.captions_df[cat_col].dropna().unique()
                for cat in unique_classes:
                    raw_labels = str(cat).replace('_', ' ').lower()
                    caption = f"The sound of {raw_labels}."
                    emb = text_encoder.encode(caption, convert_to_tensor=True).cpu()
                    self.embedding_cache[cat] = emb

        self.resampler_cache = {}

    def __len__(self):
        """
        Returns the total number of valid samples in the dataset.
        """
        return len(self.captions_df)

    def __getitem__(self, idx):
        """
        Retrieves an audio sample, standardizes its length, and fetches its text embedding.
        """
        while True:
            try:
                row = self.captions_df.iloc[idx]
                file_str = str(row.get('current_file_path', row.get('filename', ''))).replace('\\', '/')
                category = row.get('category', row.get('class', None))

                parts = file_str.split('/')
                search_key = f"{parts[-2]}/{parts[-1]}".lower() if len(parts) >= 2 else f"{category}/{parts[-1]}".lower()

                audio_path = self.file_index.get(search_key)
                if audio_path is None:
                    raise ValueError("File not found in index.")

                waveform, original_sr = torchaudio.load(audio_path)

                if original_sr != self.sample_rate:
                    if original_sr not in self.resampler_cache:
                        self.resampler_cache[original_sr] = torchaudio.transforms.Resample(original_sr, self.sample_rate)
                    waveform = self.resampler_cache[original_sr](waveform)

                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                max_abs_val = torch.max(torch.abs(waveform))
                if max_abs_val > 0:
                    waveform = waveform / max_abs_val

                if torch.rand(1).item() < 0.5:
                    gain = torch.empty(1).uniform_(0.5, 1.0).item()
                    waveform = waveform * gain

                n_samples = waveform.shape[1]
                if n_samples < TARGET_WAVE_LENGTH:
                    padding = TARGET_WAVE_LENGTH - n_samples
                    pad_left = torch.randint(0, padding + 1, (1,)).item()
                    pad_right = padding - pad_left
                    waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='constant', value=0.0)
                elif n_samples > TARGET_WAVE_LENGTH:
                    max_start = n_samples - TARGET_WAVE_LENGTH
                    start_idx = torch.randint(0, max_start + 1, (1,)).item()
                    waveform = waveform[:, start_idx:start_idx + TARGET_WAVE_LENGTH]

                raw_labels = str(category).replace('_', ' ').lower()
                caption = f"The sound of {raw_labels}."
                emb = self.embedding_cache.get(category, None)

                return waveform, caption, emb

            except Exception:
                idx = random.randint(0, len(self.captions_df) - 1)


def collate_fn(batch):
    """
    Groups individual samples into a batch for training.
    """
    waveforms, captions, embs = zip(*batch)
    if embs[0] is not None:
        return torch.stack(waveforms), list(captions), torch.stack(embs).squeeze(1)
    return torch.stack(waveforms), list(captions), None