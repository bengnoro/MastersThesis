import os
import shutil
import torch
import torchaudio
import torchaudio.transforms as T
import warnings
import glob
import csv
import pandas as pd
import scipy.io.wavfile as wavfile
import librosa
import numpy as np
from tqdm import tqdm

try:
    from frechet_audio_distance import FrechetAudioDistance
except ImportError:
    print("Missing library: frechet_audio_distance. Please install it.")
    exit()

try:
    import bigvgan
except ImportError:
    print("Missing library: bigvgan. Please install it.")
    exit()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)

from models import Generator, NOISE_DIM
from data_pipeline import (
    TEXT_ENCODER_MODEL, EMBEDDING_DIM, SAMPLE_RATE,
    ClapTextEncoder, DATASET_MEAN, DATASET_STD,
    N_FFT, HOP_LENGTH, N_MELS, F_MIN, F_MAX
)

CHECKPOINT_DIR = "/content/drive/MyDrive/diplomka/checkpoints"
RESULTS_CSV = "/content/drive/MyDrive/diplomka/fad_scores.csv"

FAKE_AUDIO_DIR_BVG = "/content/fad_eval/fake_bvg"
FAKE_AUDIO_DIR_GL = "/content/fad_eval/fake_gl"
REAL_AUDIO_DIR = "/content/fad_eval/real"

LOCAL_DCASE_DIR = "/content/DCASE_2023_Challenge_Task_7_Dataset"
LOCAL_CSV_FILE = "/content/DCASE_2023_Challenge_Task_7_Dataset/DevMeta.csv"

NUM_SAMPLES_TO_GENERATE = 1000
FEATURES_GEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEADROOM_FACTOR = 0.85


def get_bigvgan_pipeline():
    """
    Initializes and returns the BigVGAN vocoder model.
    """
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
    """
    Reverses the normalization process to retrieve the log-scaled amplitude spectrogram.
    """
    s_db = (spectrogram * (3.0 * DATASET_STD)) + DATASET_MEAN
    s_amp = 10.0 ** (s_db / 20.0)
    return torch.log(torch.clamp(s_amp, min=1e-5))


def denormalize_to_linear_amplitude(spectrogram):
    """
    Reverses the normalization process to retrieve the linear amplitude spectrogram.
    """
    s_db = (spectrogram * (3.0 * DATASET_STD)) + DATASET_MEAN
    return 10.0 ** (s_db / 20.0)


def trim_silence(audio_tensor, top_db=40, min_length_samples=22050):
    """
    Removes silent segments from the beginning and end of the audio file.
    """
    wav_np = audio_tensor.squeeze().numpy()
    trimmed_np, _ = librosa.effects.trim(wav_np, top_db=top_db)

    if len(trimmed_np) < min_length_samples:
        pad_length = min_length_samples - len(trimmed_np)
        trimmed_np = np.pad(trimmed_np, (0, pad_length), mode='constant', constant_values=0)

    return torch.from_numpy(trimmed_np).unsqueeze(0)


def main():
    """
    Runs the full FAD evaluation pipeline across multiple model checkpoints.
    """
    print("Initializing Evaluation Pipeline")
    epochs_to_process = [100, 150, 200, 250, 300, 320, 325, 350, 400, 450]
    print(f"Target Epochs: {epochs_to_process}")

    if not os.path.exists(LOCAL_CSV_FILE):
        print(f"File missing: {LOCAL_CSV_FILE}.")
        exit(1)

    print("Extracting metadata...")
    df = pd.read_csv(LOCAL_CSV_FILE)

    possible_file_cols = ['current_file_path', 'original_file_name', 'filename', 'file_name', 'fname', 'file', 'path']
    file_col = next((col for col in possible_file_cols if col in df.columns), None)
    cat_col = 'category' if 'category' in df.columns else 'class'

    if os.path.exists(REAL_AUDIO_DIR):
        shutil.rmtree(REAL_AUDIO_DIR)
    os.makedirs(REAL_AUDIO_DIR, exist_ok=True)

    print("Scanning audio files...")
    all_wavs_on_disk = glob.glob(os.path.join(LOCAL_DCASE_DIR, "**", "*.wav"), recursive=True)
    all_wavs_on_disk += glob.glob(os.path.join(LOCAL_DCASE_DIR, "**", "*.WAV"), recursive=True)

    wav_lookup = {}
    for p in all_wavs_on_disk:
        parts = p.replace('\\', '/').split('/')
        if len(parts) >= 2:
            key = f"{parts[-2]}/{parts[-1]}".lower()
            wav_lookup[key] = p

    valid_files_copied = 0
    corrupted_files_skipped = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Flattening Real Audio", leave=False):
        file_str = str(row[file_col]).replace('\\', '/')
        parts = file_str.split('/')

        if len(parts) >= 2:
            search_key = f"{parts[-2]}/{parts[-1]}".lower()
        else:
            cat_str = str(row[cat_col]) if cat_col in df.columns and pd.notna(row[cat_col]) else ''
            search_key = f"{cat_str}/{parts[-1]}".lower()

        if not search_key.endswith('.wav'):
            search_key += '.wav'

        if search_key in wav_lookup:
            src_path = wav_lookup[search_key]
            target_path = os.path.join(REAL_AUDIO_DIR, f"real_gt_{i}.wav")

            try:
                sr, data = wavfile.read(src_path)
                if len(data) > 0:
                    shutil.copy2(src_path, target_path)
                    valid_files_copied += 1
                else:
                    corrupted_files_skipped += 1
            except Exception:
                corrupted_files_skipped += 1

    raw_classes = df[cat_col].dropna().unique().tolist()
    classes = [str(cat).replace('_', ' ').lower() for cat in raw_classes]
    samples_per_class = max(1, NUM_SAMPLES_TO_GENERATE // len(classes))

    text_encoder = ClapTextEncoder(TEXT_ENCODER_MODEL, device=DEVICE)
    gen = Generator(noise_dim=NOISE_DIM, text_dim=EMBEDDING_DIM, base_channels=FEATURES_GEN).to(DEVICE)

    vocoder_bvg = get_bigvgan_pipeline()

    inverse_mel = T.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
        f_min=F_MIN,
        f_max=F_MAX
    ).to(DEVICE)

    griffin_lim = T.GriffinLim(
        n_fft=N_FFT,
        n_iter=64,
        win_length=N_FFT,
        hop_length=HOP_LENGTH,
        power=1.0
    ).to(DEVICE)

    print("Loading feature extractor...")
    frechet = FrechetAudioDistance(model_name="vggish", sample_rate=16000, use_pca=False, use_activation=False,
                                   verbose=False)

    with open(RESULTS_CSV, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not os.path.getsize(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else True:
            csv_writer.writerow(["Epoch", "BigVGAN_FAD", "GriffinLim_FAD"])

        for epoch in epochs_to_process:
            print(f"\nProcessing Epoch {epoch}")
            ckpt_path = f"{CHECKPOINT_DIR}/gen_epoch_{epoch}.pth.tar"

            if not os.path.exists(ckpt_path):
                print(f"Checkpoint not found: {ckpt_path}. Skipping.")
                continue

            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            gen.load_state_dict(checkpoint["state_dict"])
            gen.eval()

            if os.path.exists(FAKE_AUDIO_DIR_BVG):
                shutil.rmtree(FAKE_AUDIO_DIR_BVG)
            if os.path.exists(FAKE_AUDIO_DIR_GL):
                shutil.rmtree(FAKE_AUDIO_DIR_GL)

            os.makedirs(FAKE_AUDIO_DIR_BVG, exist_ok=True)
            os.makedirs(FAKE_AUDIO_DIR_GL, exist_ok=True)

            fake_stats = {"min": float('inf'), "max": float('-inf'), "mean_sum": 0.0, "count": 0}

            print("Generating synthesized audio files...")
            with torch.no_grad():
                file_idx = 0
                for cls in classes:
                    text_emb = text_encoder.encode(f"The sound of {cls}.", convert_to_tensor=True).to(DEVICE)
                    for _ in tqdm(range(samples_per_class), desc=f"Generating {cls}", leave=False):
                        noise = torch.randn(1, NOISE_DIM).to(DEVICE)

                        with torch.amp.autocast('cuda'):
                            generated_spec = gen(noise, text_emb)

                        s_log = denormalize_to_log_amplitude(generated_spec.squeeze(1).float())
                        audio_bvg = vocoder_bvg(s_log).squeeze(1).cpu()
                        audio_bvg = trim_silence(audio_bvg, top_db=40)
                        audio_bvg = torch.clamp(audio_bvg, min=-1.0, max=1.0) * HEADROOM_FACTOR

                        s_amp = denormalize_to_linear_amplitude(generated_spec.squeeze(1).float())
                        with torch.amp.autocast('cuda'):
                            linear_spec = inverse_mel(s_amp)
                            audio_gl = griffin_lim(linear_spec).cpu()

                        if audio_gl.dim() == 1:
                            audio_gl = audio_gl.unsqueeze(0)
                        elif audio_gl.dim() == 3:
                            audio_gl = audio_gl.squeeze(0)

                        audio_gl = trim_silence(audio_gl, top_db=40)
                        audio_gl = torch.clamp(audio_gl, min=-1.0, max=1.0) * HEADROOM_FACTOR

                        val_min = audio_bvg.min().item()
                        val_max = audio_bvg.max().item()
                        val_mean = audio_bvg.mean().item()

                        fake_stats["min"] = min(fake_stats["min"], val_min)
                        fake_stats["max"] = max(fake_stats["max"], val_max)
                        fake_stats["mean_sum"] += val_mean
                        fake_stats["count"] += 1

                        audio_bvg_int16 = (audio_bvg * 32767.0).to(torch.int16)
                        audio_gl_int16 = (audio_gl * 32767.0).to(torch.int16)

                        torchaudio.save(os.path.join(FAKE_AUDIO_DIR_BVG, f"fake_{file_idx}.wav"), audio_bvg_int16,
                                        SAMPLE_RATE)
                        torchaudio.save(os.path.join(FAKE_AUDIO_DIR_GL, f"fake_{file_idx}.wav"), audio_gl_int16,
                                        SAMPLE_RATE)
                        file_idx += 1

            print("\nEvaluation Data Metrics")
            print(f"Total Files Scanned: {len(df)}")
            print(f"Valid Source Files: {valid_files_copied}")
            print(f"Corrupted Source Files: {corrupted_files_skipped}")
            print(f"Total Generated Outputs: {fake_stats['count'] * 2}")

            if fake_stats['count'] > 0:
                print(f"Signal Min: {fake_stats['min']:.4f}")
                print(f"Signal Max: {fake_stats['max']:.4f}")
                print(f"Signal Average: {(fake_stats['mean_sum'] / fake_stats['count']):.4f}")

            print("Calculating FAD scores...")
            fad_score_bvg = frechet.score(background_dir=REAL_AUDIO_DIR, eval_dir=FAKE_AUDIO_DIR_BVG, dtype="float32")
            print(f"BigVGAN FAD: {fad_score_bvg:.4f}")

            fad_score_gl = frechet.score(background_dir=REAL_AUDIO_DIR, eval_dir=FAKE_AUDIO_DIR_GL, dtype="float32")
            print(f"Griffin-Lim FAD: {fad_score_gl:.4f}")

            csv_writer.writerow([epoch, round(fad_score_bvg, 4), round(fad_score_gl, 4)])
            csvfile.flush()

    if os.path.exists(FAKE_AUDIO_DIR_BVG):
        shutil.rmtree(FAKE_AUDIO_DIR_BVG)
    if os.path.exists(FAKE_AUDIO_DIR_GL):
        shutil.rmtree(FAKE_AUDIO_DIR_GL)
    if os.path.exists(REAL_AUDIO_DIR):
        shutil.rmtree(REAL_AUDIO_DIR)

    print("\nProcess finished.")


if __name__ == "__main__":
    main()