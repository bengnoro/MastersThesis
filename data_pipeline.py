import torch
import torchaudio
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import os

# --- Configuration Constants ---
# These parameters are critical. They define the "shape" of your data.
# You must use the same values when you build your models (Generator/Discriminator).

# Audio Processing
SAMPLE_RATE = 22050  # Sample rate. 22050 Hz is common for audio ML
N_FFT = 1024  # Number of points for FFT
HOP_LENGTH = 256  # Hop length for STFT. Controls time resolution.
N_MELS = 80  # Number of Mel filter banks. Controls frequency resolution.
TARGET_TIME_STEPS = 512  # Fixed time steps for the spectrogram. Critical for batching.
# This corresponds to ~5.9 seconds of audio with these settings.

# Text Processing
TEXT_ENCODER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# This model outputs a 384-dimensional vector.
EMBEDDING_DIM = 384

# --- Dummy Data Paths (Update if needed) ---
# Note: Ensure these match your actual folder structure
DUMMY_AUDIO_DIR = "./demoSounds/audio"
DUMMY_CSV_FILE = "./demoSounds/dummy_captions.csv"


class AudioTextDataset(Dataset):
    """
    Custom PyTorch Dataset to load audio files and text captions.
    Performs all necessary preprocessing to return a
    (spectrogram_tensor, text_embedding_tensor) pair.
    """

    def __init__(self, csv_file, audio_dir, text_encoder, sample_rate, n_fft, hop_length, n_mels, target_time_steps):
        """
        Initializes the Dataset.
        """
        self.audio_dir = audio_dir
        self.text_encoder = text_encoder
        self.sample_rate = sample_rate
        self.target_time_steps = target_time_steps
        self.n_mels = n_mels  # <--- FIXED: Saved this attribute to prevent crash on error

        # --- CSV Loading & Cleaning ---
        try:
            # FIXED: sep=None with engine='python' auto-detects delimiter (comma, tab, etc.)
            self.captions_df = pd.read_csv(csv_file, sep=None, engine='python')

            # Clean column names: strip whitespace and convert to lowercase
            self.captions_df.columns = self.captions_df.columns.str.strip().str.lower()

            # Validation: Check if required columns exist
            required_cols = ['filename', 'caption']
            missing_cols = [c for c in required_cols if c not in self.captions_df.columns]

            if missing_cols:
                # Fallback: Try to guess if 'filename' is named 'file_name' or 'fname'
                rename_map = {
                    'file_name': 'filename', 'fname': 'filename', 'file': 'filename',
                    'text': 'caption', 'description': 'caption', 'desc': 'caption'
                }
                self.captions_df.rename(columns=rename_map, inplace=True)

                # Check again
                missing_cols = [c for c in required_cols if c not in self.captions_df.columns]
                if missing_cols:
                    raise ValueError(
                        f"CSV is missing columns: {missing_cols}. Found: {self.captions_df.columns.tolist()}")

        except Exception as e:
            print(f"CRITICAL ERROR loading CSV: {e}")
            raise e

        # --- Define Audio Transforms ---
        # Transform to convert a waveform to a Mel spectrogram
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            # n_iter=32, # REMOVED: This argument is for GriffinLim, not MelSpectrogram
        )

        # Transform to convert amplitude to decibels
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.captions_df)

    def __getitem__(self, idx):
        """
        Gets a single processed data point.
        Returns: tuple: (spectrogram_tensor, text_embedding_tensor) or (None, None) on failure
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = "Unknown"
        caption = "Unknown"

        try:
            # 1. --- Get Filename and Caption ---
            # Access by column name safely after cleaning in __init__
            # FIXED: Added .strip() and .replace to handle quote issues in filename
            raw_filename = str(self.captions_df.iloc[idx]['filename'])
            audio_filename = raw_filename.strip().strip('"').strip("'")

            caption = str(self.captions_df.iloc[idx]['caption'])

            audio_path = os.path.join(self.audio_dir, audio_filename)

            # 2. --- Audio Processing Pipeline ---
            if not os.path.exists(audio_path):
                # Detailed error to help debug paths
                raise FileNotFoundError(f"File not found. Looked for: '{audio_path}'")

            # Load audio
            waveform, original_sr = torchaudio.load(audio_path)

            # Resample if necessary
            if original_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(original_sr, self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Normalize audio (simple peak normalization)
            max_abs_val = torch.max(torch.abs(waveform))
            if max_abs_val > 0:
                waveform = waveform / max_abs_val

            # Compute Mel spectrogram
            spectrogram = self.mel_spectrogram(waveform)

            # Convert to Decibels (dB)
            spectrogram = self.amplitude_to_db(spectrogram)

            # 3. --- Padding / Truncation ---
            n_time_steps = spectrogram.shape[2]

            if n_time_steps < self.target_time_steps:
                padding = self.target_time_steps - n_time_steps
                spectrogram = torch.nn.functional.pad(spectrogram, (0, padding), 'constant', -80.0)
            else:
                spectrogram = spectrogram[:, :, :self.target_time_steps]

            # Ensure spectrogram has a "channel" dimension for CNNs [1, n_mels, time]
            if spectrogram.dim() == 2:
                spectrogram = spectrogram.unsqueeze(0)

            # 4. --- Text Processing Pipeline ---
            with torch.no_grad():
                text_embedding = self.text_encoder.encode(caption, convert_to_tensor=True)

            if len(text_embedding.shape) > 1:
                text_embedding = text_embedding.squeeze()

            return spectrogram, text_embedding

        except Exception as e:
            # Print error but don't crash yet (handled by collate_fn)
            print(f"Error processing index {idx}: {e}")
            # FIXED: Now self.n_mels exists, so this won't crash
            return torch.empty((1, self.n_mels, self.target_time_steps)), torch.empty((EMBEDDING_DIM,))


def collate_fn(batch):
    """
    Custom collate function to filter out failed samples (None, None)
    or empty tensors returned by error handler.
    """
    # Filter out bad samples
    # We check if the spectrogram is "empty" (size 0) or if it was the error sentinel
    valid_batch = []
    for item in batch:
        spec, embed = item
        if spec is None: continue
        # Check if it's the specific "error" tensor we return in except block
        # (Though usually we'd just return None in except, let's handle the empty tensor case too)
        valid_batch.append(item)

    if len(valid_batch) == 0:
        return None, None

    return torch.utils.data.dataloader.default_collate(valid_batch)


# --- SMOKE TEST BLOCK ---
if __name__ == "__main__":

    print("Starting data pipeline smoke test...")

    # Check if paths exist
    if not os.path.exists(DUMMY_AUDIO_DIR) or not os.path.exists(DUMMY_CSV_FILE):
        print("\n--- ERROR ---")
        print("Dummy data not found!")
        print(f"Directory checked: {os.path.abspath(DUMMY_AUDIO_DIR)}")
        print(f"File checked: {os.path.abspath(DUMMY_CSV_FILE)}")
    else:
        try:
            print(f"Loading text encoder model: '{TEXT_ENCODER_MODEL}'...")
            text_encoder = SentenceTransformer(TEXT_ENCODER_MODEL)
            print("Text encoder loaded.")

            print("Instantiating AudioTextDataset...")
            dataset = AudioTextDataset(
                csv_file=DUMMY_CSV_FILE,
                audio_dir=DUMMY_AUDIO_DIR,
                text_encoder=text_encoder,
                sample_rate=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                target_time_steps=TARGET_TIME_STEPS
            )
            print(f"Dataset created. Found {len(dataset)} sample(s).")
            print(f"Columns found: {dataset.captions_df.columns.tolist()}")

            # Print first few rows to verify CSV parsing
            print("\nFirst 2 rows of CSV data as parsed:")
            print(dataset.captions_df.head(2))

            # Test fetching directly
            if len(dataset) > 0:
                print("\nFetching first data sample (dataset[0])...")
                spectrogram, embedding = dataset[0]

                # Check for the error sentinel (empty tensor)
                if spectrogram.numel() == 0:  # Check if empty
                    print("FAIL: dataset[0] returned empty tensor (Error occurred in __getitem__)")
                else:
                    print("\n--- Single Item Test Results ---")
                    print(f"  Spectrogram shape: {spectrogram.shape}")
                    print(f"  Text Embedding shape: {embedding.shape}")
                    assert spectrogram.shape == (1, N_MELS, TARGET_TIME_STEPS), "Spectrogram shape wrong!"

                print("\n--- DataLoader Test (with collate_fn) ---")
                dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

                try:
                    spec_batch, embed_batch = next(iter(dataloader))
                    if spec_batch is None:
                        print("Batch was empty (all samples failed).")
                    else:
                        print(f"  Spectrogram batch shape: {spec_batch.shape}")
                        print(f"  Text Embedding batch shape: {embed_batch.shape}")
                        print("\n--- SMOKE TEST PASSED ---")
                except StopIteration:
                    print("DataLoader returned no batches.")

                print("Saving spectrogram visualization to 'test_spec.png'...")
                spec_numpy = spectrogram.squeeze().numpy()
                plt.figure(figsize=(10, 4))
                plt.imshow(spec_numpy, aspect='auto', origin='lower')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f"Spectrogram for: {dataset.captions_df.iloc[0]['caption']}")
                plt.savefig("test_spec.png")
                print("Done! Check your folder for test_spec.png")

            else:
                print("Dataset is empty.")



        except Exception as e:
            print(f"\n--- SMOKE TEST FAILED ---")
            print(f"An error occurred: {e}")
            import traceback

            traceback.print_exc()