import os
import shutil
import torch
import torchaudio
import warnings
import glob
import csv
import pandas as pd
import scipy.io.wavfile as wavfile
from tqdm import tqdm

try:
    from frechet_audio_distance import FrechetAudioDistance
except ImportError:
    print("CRITICAL: Please install the FAD library: `pip install frechet_audio_distance`")
    exit()

try:
    import bigvgan
except ImportError:
    print("Please install NVIDIA BigVGAN: `pip install bigvgan`")
    exit()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)

from models import Generator, NOISE_DIM
from data_pipeline import (
    TEXT_ENCODER_MODEL, EMBEDDING_DIM, SAMPLE_RATE,
    ClapTextEncoder, DATASET_MEAN, DATASET_STD
)

# --- STRUCTURED COLAB PATHS ---
CHECKPOINT_DIR = "/content/drive/MyDrive/diplomka/checkpoints"
RESULTS_CSV = "/content/drive/MyDrive/diplomka/fad_scores.csv"

FAKE_AUDIO_DIR = "/content/fad_eval/fake"
REAL_AUDIO_DIR = "/content/fad_eval/real"

LOCAL_DCASE_DIR = "/content/DCASE_2023_Challenge_Task_7_Dataset"
LOCAL_CSV_FILE = "/content/DCASE_2023_Challenge_Task_7_Dataset/DevMeta.csv"

NUM_SAMPLES_TO_GENERATE = 1000
FEATURES_GEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bigvgan_pipeline():
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
    return vocoder


def denormalize_to_log_amplitude(spectrogram):
    S_db = (spectrogram * (3.0 * DATASET_STD)) + DATASET_MEAN
    S_amp = 10.0 ** (S_db / 20.0)
    S_log = torch.log(torch.clamp(S_amp, min=1e-5))
    return S_log


def main():
    print("--- Initializing FAD Batch Evaluator ---")

    epochs_to_process = [525]
    print(f"Found {len(epochs_to_process)} milestones to evaluate: {epochs_to_process}")

    if not os.path.exists(LOCAL_CSV_FILE):
        print(f"\n[FATAL] Local dataset not found at {LOCAL_CSV_FILE}.")
        exit(1)

    print("Extracting exact file distribution from DevMeta.csv...")
    df = pd.read_csv(LOCAL_CSV_FILE)

    possible_file_cols = ['current_file_path', 'original_file_name', 'filename', 'file_name', 'fname', 'file', 'path']
    file_col = next((col for col in possible_file_cols if col in df.columns), None)
    cat_col = 'category' if 'category' in df.columns else 'class'

    print("Preparing flat ground-truth directory...")
    if os.path.exists(REAL_AUDIO_DIR):
        shutil.rmtree(REAL_AUDIO_DIR)
    os.makedirs(REAL_AUDIO_DIR, exist_ok=True)

    print(f"Scanning {LOCAL_DCASE_DIR} tree for .wav files...")
    all_wavs_on_disk = glob.glob(os.path.join(LOCAL_DCASE_DIR, "**", "*.wav"), recursive=True)
    all_wavs_on_disk += glob.glob(os.path.join(LOCAL_DCASE_DIR, "**", "*.WAV"), recursive=True)

    # SCIENTIFIC FIX: The Folder-Aware Dictionary Map
    wav_lookup = {}
    for p in all_wavs_on_disk:
        parts = p.replace('\\', '/').split('/')
        if len(parts) >= 2:
            key = f"{parts[-2]}/{parts[-1]}".lower()  # e.g., 'dog_bark/001.wav'
            wav_lookup[key] = p

    print(f"Registered {len(wav_lookup)} unique physical audio files on the NVMe drive.")

    valid_files_copied = 0
    corrupted_files_skipped = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Verifying & Flattening Real Audio", leave=False):
        file_str = str(row[file_col]).replace('\\', '/')
        parts = file_str.split('/')

        if len(parts) >= 2:
            search_key = f"{parts[-2]}/{parts[-1]}".lower()
        else:
            cat_str = str(row[cat_col]) if cat_col in df.columns and pd.notna(row[cat_col]) else ''
            search_key = f"{cat_str}/{parts[-1]}".lower()

        if not search_key.endswith('.wav'): search_key += '.wav'

        if search_key in wav_lookup:
            src_path = wav_lookup[search_key]
            target_path = os.path.join(REAL_AUDIO_DIR, f"real_gt_{i}.wav")

            # PRE-FLIGHT CHECK: Using FAD's native Scipy backend
            try:
                sr, data = wavfile.read(src_path)
                if len(data) > 0:
                    shutil.copy2(src_path, target_path)
                    valid_files_copied += 1
                else:
                    corrupted_files_skipped += 1
            except Exception as e:
                corrupted_files_skipped += 1
                print(f"\n[WARNING] Scipy rejected {search_key}: {e}")

    raw_classes = df[cat_col].dropna().unique().tolist()
    classes = [str(cat).replace('_', ' ').lower() for cat in raw_classes]
    samples_per_class = max(1, NUM_SAMPLES_TO_GENERATE // len(classes))

    text_encoder = ClapTextEncoder(TEXT_ENCODER_MODEL, device=DEVICE)
    gen = Generator(noise_dim=NOISE_DIM, text_dim=EMBEDDING_DIM, base_channels=FEATURES_GEN).to(DEVICE)
    vocoder = get_bigvgan_pipeline()

    print("Loading VGGish Feature Extractor...")
    frechet = FrechetAudioDistance(model_name="vggish", sample_rate=16000, use_pca=False, use_activation=False,
                                   verbose=True)

    with open(RESULTS_CSV, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not os.path.getsize(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else True:
            csv_writer.writerow(["Epoch", "FAD_Score"])

        for epoch in epochs_to_process:
            print(f"\n EVALUATING EPOCH {epoch}")
            ckpt_path = f"{CHECKPOINT_DIR}/gen_epoch_{epoch}.pth.tar"
            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            gen.load_state_dict(checkpoint["state_dict"])
            gen.eval()

            if os.path.exists(FAKE_AUDIO_DIR): shutil.rmtree(FAKE_AUDIO_DIR)
            os.makedirs(FAKE_AUDIO_DIR, exist_ok=True)

            fake_stats = {"min": float('inf'), "max": float('-inf'), "mean_sum": 0.0, "count": 0}

            print(f"Synthesizing {NUM_SAMPLES_TO_GENERATE} fake samples...")
            with torch.no_grad():
                file_idx = 0
                for cls in classes:
                    text_emb = text_encoder.encode(f"The sound of {cls}.", convert_to_tensor=True).to(DEVICE)
                    for _ in tqdm(range(samples_per_class), desc=f"'{cls}'", leave=False):
                        noise = torch.randn(1, NOISE_DIM).to(DEVICE)
                        with torch.amp.autocast('cuda'):
                            generated_spec = gen(noise, text_emb)
                        S_log = denormalize_to_log_amplitude(generated_spec.squeeze(1).float())
                        audio_waveform = vocoder(S_log).squeeze(1).cpu()

                        max_val = torch.max(torch.abs(audio_waveform))
                        if max_val > 0: audio_waveform = audio_waveform / max_val

                        val_min = audio_waveform.min().item()
                        val_max = audio_waveform.max().item()
                        val_mean = audio_waveform.mean().item()

                        fake_stats["min"] = min(fake_stats["min"], val_min)
                        fake_stats["max"] = max(fake_stats["max"], val_max)
                        fake_stats["mean_sum"] += val_mean
                        fake_stats["count"] += 1

                        audio_int16 = (audio_waveform * 32767.0).to(torch.int16)
                        torchaudio.save(os.path.join(FAKE_AUDIO_DIR, f"fake_{file_idx}.wav"), audio_int16, SAMPLE_RATE)
                        file_idx += 1

            print("\n" + "=" * 50)
            print(" DIAGNOSTIC TELEMETRY REPORT ")
            print("=" * 50)
            print(f"Total Dataset Files Scanned : {len(df)}")
            print(f"Real Files Verified (OK)    : {valid_files_copied}")
            print(f"Real Files Corrupted (SKIP) : {corrupted_files_skipped}")
            print("-" * 50)
            print(f"Fake Files Generated        : {fake_stats['count']}")
            if fake_stats['count'] > 0:
                print(f"Fake Audio Global Min       : {fake_stats['min']:.4f}")
                print(f"Fake Audio Global Max       : {fake_stats['max']:.4f}")
                print(f"Fake Audio Average Mean     : {(fake_stats['mean_sum'] / fake_stats['count']):.4f}")
            print("=" * 50 + "\n")

            print(f"Computing Fréchet Audio Distance for Epoch {epoch}...")
            fad_score = frechet.score(background_dir=REAL_AUDIO_DIR, eval_dir=FAKE_AUDIO_DIR, dtype="float32")
            print(f"-> FAD Score: {fad_score:.4f}")
            csv_writer.writerow([epoch, round(fad_score, 4)])
            csvfile.flush()

    if os.path.exists(FAKE_AUDIO_DIR): shutil.rmtree(FAKE_AUDIO_DIR)
    if os.path.exists(REAL_AUDIO_DIR): shutil.rmtree(REAL_AUDIO_DIR)
    print(f"\nAll FAD evaluations complete.")


if __name__ == "__main__":
    main()