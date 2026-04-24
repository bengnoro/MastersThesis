import os
import sys
import glob
import re

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import matplotlib.pyplot as plt
import warnings
import time
import gc
import multiprocessing
import shutil

warnings.filterwarnings("ignore", category=FutureWarning)

from data_pipeline import (
    AudioTextDataset, collate_fn, N_MELS, TARGET_TIME_STEPS,
    EMBEDDING_DIM as TEXT_EMBEDDING_DIM, SAMPLE_RATE, N_FFT, HOP_LENGTH,
    DUMMY_CSV_FILE, DUMMY_AUDIO_DIR, TEXT_ENCODER_MODEL, ClapTextEncoder,
    DATASET_MEAN, DATASET_STD, F_MIN, F_MAX, CENTER
)
from models import Generator, Discriminator, initialize_weights, NOISE_DIM


# --- Strict Architectural Gatekeeper ---
def enforce_a100_and_get_batch_size():
    if not torch.cuda.is_available():
        print("\n[FATAL] No GPU detected. Terminating script.")
        sys.exit(1)

    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)

    print(f"\n[HARDWARE DETECTED] GPU has {vram_gb:.1f} GB VRAM.")

    if vram_gb >= 35.0:
        print("=> A100 Architecture Confirmed. Locking Batch Size to 64 for momentum stability.")
        return 64
    else:
        print(f"\n[FATAL] Assigned GPU only has {vram_gb:.1f} GB VRAM. You requested strict A100 execution.")
        print("Terminating script immediately to protect RNG continuity and checkpoint integrity.")
        sys.exit(1)


# --- Hyperparameters (TTUR Applied) ---
LEARNING_RATE_GEN = 5e-5
LEARNING_RATE_CRITIC = 2e-4
BATCH_SIZE = enforce_a100_and_get_batch_size()
Z_DIM = NOISE_DIM

FEATURES_CRITIC = 128
FEATURES_GEN = 256

CRITIC_ITERATIONS = 3
LAMBDA_FM = 0.1
LAMBDA_R1 = 1.0

NUM_EPOCHS = 1001
SAVE_INTERVAL = 5
KEEP_LAST_CHECKPOINTS = 3

# --- Structured Colab Paths ---
CHECKPOINT_DIR = "/content/drive/MyDrive/diplomka/checkpoints"
LOG_DIR = "/content/drive/MyDrive/diplomka/logs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def diff_spec_augment(x):
    x_aug = x.clone()
    B, C, F_dim, T_dim = x_aug.shape
    for i in range(B):
        f_len = torch.randint(0, F_dim // 5, (1,)).item()
        f_st = torch.randint(0, F_dim - f_len + 1, (1,)).item()
        x_aug[i, :, f_st:f_st + f_len, :] = 0.0

        t_len = torch.randint(0, T_dim // 10, (1,)).item()
        t_st = torch.randint(0, T_dim - t_len + 1, (1,)).item()
        x_aug[i, :, :, t_st:t_st + t_len] = 0.0
    return x_aug


def cleanup_old_checkpoints(current_epoch):
    epoch_to_delete = current_epoch - ((KEEP_LAST_CHECKPOINTS + 1) * SAVE_INTERVAL)
    if epoch_to_delete >= 0 and epoch_to_delete % 50 != 0:
        for prefix in ["gen", "critic"]:
            path = f"{CHECKPOINT_DIR}/{prefix}_epoch_{epoch_to_delete}.pth.tar"
            if os.path.exists(path): os.remove(path)


def save_spectrogram_image(gen_spec, epoch, batch_idx):
    os.makedirs(LOG_DIR, exist_ok=True)
    spec_cpu = torch.clamp(gen_spec[0].squeeze().detach().cpu().float(), min=-1.0, max=1.0).numpy()
    spec_db = (spec_cpu * (3.0 * DATASET_STD)) + DATASET_MEAN
    vmin = DATASET_MEAN - (3.0 * DATASET_STD)
    vmax = DATASET_MEAN + (3.0 * DATASET_STD)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spec_db, aspect='auto', origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, format='%+2.0f dB')
    fig.tight_layout()
    fig.savefig(f"{LOG_DIR}/epoch_{epoch}_batch_{batch_idx}.png")
    fig.clf()
    plt.close('all')
    gc.collect()


def plot_and_save_losses(g_losses, d_losses):
    os.makedirs(LOG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(g_losses, label="Generator Loss", color="blue", alpha=0.8)
    ax.plot(d_losses, label="Critic Loss (Hinge)", color="red", alpha=0.8)
    ax.set_title("GAN Training Loss over Time")
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{LOG_DIR}/training_loss_graph.png")
    fig.clf()
    plt.close('all')
    gc.collect()


def find_latest_checkpoint(ckpt_dir):
    gen_files = glob.glob(os.path.join(ckpt_dir, "gen_epoch_*.pth.tar"))
    if not gen_files:
        return 0
    epochs = [int(re.search(r"epoch_(\d+)", f).group(1)) for f in gen_files]
    max_epoch = max(epochs)

    if os.path.exists(os.path.join(ckpt_dir, f"critic_epoch_{max_epoch}.pth.tar")):
        return max_epoch
    return 0


def train():
    print(f"Using device: {device}")

    while True:
        try:
            print("Attempting to load models onto GPU VRAM...")
            text_encoder = ClapTextEncoder(TEXT_ENCODER_MODEL, device=device)
            gen = Generator(NOISE_DIM, TEXT_EMBEDDING_DIM, FEATURES_GEN).to(device)
            critic = Discriminator(TEXT_EMBEDDING_DIM, FEATURES_CRITIC).to(device)

            if torch.cuda.is_available():
                _test_tensor = torch.zeros((256, 1024, 1024), device=device)
                del _test_tensor
                torch.cuda.empty_cache()
            print("Initial VRAM successfully claimed.")
            break
        except Exception as e:
            if "out of memory" in str(e).lower():
                print("\n[STARTUP OOM ALERT] VRAM is currently full. Hibernating for 60 seconds...")
                text_encoder = gen = critic = None
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                time.sleep(60)
            else:
                raise e

    mel_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=1.0, center=CENTER
    ).to(device)

    to_db = AmplitudeToDB(stype='amplitude', top_db=80.0).to(device)

    dataset = AudioTextDataset(
        csv_file=DUMMY_CSV_FILE, audio_dir=DUMMY_AUDIO_DIR,
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, target_time_steps=TARGET_TIME_STEPS,
        text_encoder=text_encoder
    )

    optimal_workers = min(4, multiprocessing.cpu_count() or 1)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True,
                        num_workers=optimal_workers, pin_memory=True)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.5, 0.9), eps=1e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC, betas=(0.5, 0.9), eps=1e-4)

    scaler_gen = torch.amp.GradScaler('cuda')
    scaler_critic = torch.amp.GradScaler('cuda')

    def lr_lambda(epoch):
        decay_start = int(NUM_EPOCHS * 0.7)
        if epoch < decay_start:
            return 1.0
        else:
            decay_epochs = NUM_EPOCHS - decay_start
            return max(0.0, 1.0 - (epoch - decay_start) / decay_epochs)

    scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda)
    scheduler_critic = optim.lr_scheduler.LambdaLR(opt_critic, lr_lambda)

    g_losses = []
    d_losses = []

    latest_epoch = find_latest_checkpoint(CHECKPOINT_DIR)

    if latest_epoch > 0:
        print(f"Auto-resuming training from Epoch {latest_epoch}...")

        ckpt_gen = torch.load(f"{CHECKPOINT_DIR}/gen_epoch_{latest_epoch}.pth.tar", map_location=device,
                              weights_only=True)
        gen.load_state_dict(ckpt_gen["state_dict"])
        opt_gen.load_state_dict(ckpt_gen["optimizer"])
        if "scheduler" in ckpt_gen: scheduler_gen.load_state_dict(ckpt_gen["scheduler"])
        if "scaler" in ckpt_gen: scaler_gen.load_state_dict(ckpt_gen["scaler"])
        if "g_losses" in ckpt_gen: g_losses = ckpt_gen["g_losses"]
        if "d_losses" in ckpt_gen: d_losses = ckpt_gen["d_losses"]

        if "torch_rng" in ckpt_gen: torch.set_rng_state(ckpt_gen["torch_rng"].cpu())
        if "cuda_rng" in ckpt_gen: torch.cuda.set_rng_state_all([state.cpu() for state in ckpt_gen["cuda_rng"]])

        ckpt_critic = torch.load(f"{CHECKPOINT_DIR}/critic_epoch_{latest_epoch}.pth.tar", map_location=device,
                                 weights_only=True)
        critic.load_state_dict(ckpt_critic["state_dict"])
        opt_critic.load_state_dict(ckpt_critic["optimizer"])
        if "scheduler" in ckpt_critic: scheduler_critic.load_state_dict(ckpt_critic["scheduler"])
        if "scaler" in ckpt_critic: scaler_critic.load_state_dict(ckpt_critic["scaler"])

        START_EPOCH = latest_epoch + 1
    else:
        print("No valid checkpoints found. Starting fresh from Epoch 0...")
        initialize_weights(gen)
        initialize_weights(critic)
        START_EPOCH = 0

    test_rng = torch.Generator(device=device)
    test_rng.manual_seed(42)
    fixed_noise = torch.randn(1, Z_DIM, generator=test_rng, device=device)
    fixed_text_emb = text_encoder.encode(["footsteps"], convert_to_tensor=True).to(device).clone()

    print("\nStarting GPU-Accelerated SA-ResGAN Training with R1 Penalty & DiffAugment...")

    try:
        for epoch in range(START_EPOCH, NUM_EPOCHS):
            noise_std = max(0.0, 0.1 * (1.0 - epoch / 500.0))

            for batch_idx, (real_waveforms, captions, precomputed_embs) in enumerate(loader):
                if real_waveforms is None: continue

                real_waveforms = real_waveforms.to(device)
                cur_batch_size = real_waveforms.shape[0]
                if cur_batch_size < 2: continue

                with torch.no_grad():
                    S_amp = mel_transform(real_waveforms)
                    S_db = to_db(S_amp)
                    S_norm = (S_db - DATASET_MEAN) / (3.0 * DATASET_STD)
                    S_norm = torch.clamp(S_norm, min=-1.0, max=1.0)

                    if S_norm.shape[3] > TARGET_TIME_STEPS:
                        S_norm = S_norm[:, :, :, :TARGET_TIME_STEPS]
                    elif S_norm.shape[3] < TARGET_TIME_STEPS:
                        pad_amount = TARGET_TIME_STEPS - S_norm.shape[3]
                        S_norm = F.pad(S_norm, (0, pad_amount), mode='replicate')
                    real_spec = S_norm

                if precomputed_embs is not None:
                    text_emb = precomputed_embs.to(device)
                else:
                    with torch.no_grad():
                        text_emb = text_encoder.encode(captions, convert_to_tensor=True).to(device)

                text_emb = text_emb.clone()

                success = False
                while not success:
                    try:
                        # 1. Train the Critic
                        for _ in range(CRITIC_ITERATIONS):
                            opt_critic.zero_grad(set_to_none=True)
                            noise = torch.randn(cur_batch_size, Z_DIM).to(device)

                            real_spec_noisy = real_spec + torch.randn_like(real_spec) * noise_std
                            real_spec_noisy.requires_grad_(True)

                            with torch.no_grad():
                                with torch.amp.autocast('cuda'):
                                    fake_spec = gen(noise, text_emb).detach()

                            fake_spec_noisy = fake_spec + torch.randn_like(fake_spec) * noise_std

                            aug_real = diff_spec_augment(real_spec_noisy)
                            aug_fake = diff_spec_augment(fake_spec_noisy)

                            with torch.amp.autocast('cuda'):
                                critic_real, _ = critic(aug_real, text_emb)
                                critic_fake, _ = critic(aug_fake, text_emb)

                                critic_real_flat = critic_real.reshape(-1)
                                critic_fake_flat = critic_fake.reshape(-1)

                                loss_critic_adv = torch.mean(F.relu(1.0 - critic_real_flat.float())) + \
                                                  torch.mean(F.relu(1.0 + critic_fake_flat.float()))

                            grad_real, = torch.autograd.grad(
                                outputs=critic_real.sum(),
                                inputs=real_spec_noisy,
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True
                            )

                            # SCIENTIFIC FIX: Swapped .view() for .reshape() to handle non-contiguous grad outputs
                            grad_penalty = (grad_real.float() ** 2).reshape(grad_real.size(0), -1).sum(1).mean()
                            loss_critic = loss_critic_adv + (LAMBDA_R1 / 2) * grad_penalty

                            if torch.isnan(loss_critic) or torch.isinf(loss_critic):
                                raise ValueError("Loss Explosion")

                            scaler_critic.scale(loss_critic).backward()
                            scaler_critic.unscale_(opt_critic)
                            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
                            scaler_critic.step(opt_critic)
                            scaler_critic.update()

                        # 2. Train the Generator
                        noise = torch.randn(cur_batch_size, Z_DIM).to(device)
                        opt_gen.zero_grad(set_to_none=True)

                        with torch.amp.autocast('cuda'):
                            fake_spec = gen(noise, text_emb)

                            real_spec_noisy = real_spec + torch.randn_like(real_spec) * noise_std
                            fake_spec_noisy = fake_spec + torch.randn_like(fake_spec) * noise_std

                            aug_real = diff_spec_augment(real_spec_noisy)
                            aug_fake = diff_spec_augment(fake_spec_noisy)

                            with torch.no_grad():
                                _, real_features_fresh = critic(aug_real, text_emb)

                            critic_fake_out, fake_features = critic(aug_fake, text_emb)
                            loss_gen_adv = -torch.mean(critic_fake_out.float().reshape(-1))

                            loss_fm = 0.0
                            for f_real, f_fake in zip(real_features_fresh, fake_features):
                                loss_fm += F.l1_loss(f_fake.float(), f_real.float())

                            loss_gen = loss_gen_adv + (LAMBDA_FM * loss_fm)

                        if torch.isnan(loss_gen) or torch.isinf(loss_gen):
                            raise ValueError("Loss Explosion")

                        scaler_gen.scale(loss_gen).backward()
                        scaler_gen.unscale_(opt_gen)
                        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=10.0)
                        scaler_gen.step(opt_gen)
                        scaler_gen.update()

                        g_losses.append(loss_gen.item())
                        d_losses.append(loss_critic.item())

                        if batch_idx % 25 == 0:
                            current_lr = opt_gen.param_groups[0]['lr']
                            print(
                                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \t Loss D: {loss_critic.item():.4f}, Loss G: {loss_gen.item():.4f} \t LR: {current_lr:.6f}")

                        if batch_idx == 0:
                            with torch.no_grad():
                                gen.eval()
                                with torch.amp.autocast('cuda'):
                                    save_spectrogram_image(gen(fixed_noise, fixed_text_emb), epoch, batch_idx)
                                gen.train()

                        del fake_spec, real_spec_noisy, fake_spec_noisy, aug_real, aug_fake
                        del critic_real, critic_fake, grad_real, grad_penalty
                        del critic_fake_out, fake_features, real_features_fresh, loss_critic, loss_gen
                        success = True

                    except ValueError:
                        print(f"Detected NaN/Inf at Epoch {epoch}, Batch {batch_idx}. Skipping batch.")
                        opt_gen.zero_grad(set_to_none=True)
                        opt_critic.zero_grad(set_to_none=True)
                        success = True

                    except Exception as e:
                        err_msg = str(e)
                        if "out of memory" in err_msg.lower():
                            print(
                                f"\n[OOM ALERT] GPU Memory full. Waiting 60s to cleanly retry the exact same batch...")
                            del e
                            opt_critic.zero_grad(set_to_none=True)
                            opt_gen.zero_grad(set_to_none=True)
                            gc.collect()
                            torch.cuda.empty_cache()
                            time.sleep(60)
                        else:
                            raise e

            scheduler_gen.step()
            scheduler_critic.step()
            torch.cuda.empty_cache()

            if epoch % SAVE_INTERVAL == 0:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                tmp_gen_path = f"/content/tmp_gen_{epoch}.pth.tar"
                tmp_critic_path = f"/content/tmp_critic_{epoch}.pth.tar"

                torch.save({
                    "epoch": epoch,
                    "state_dict": gen.state_dict(),
                    "optimizer": opt_gen.state_dict(),
                    "scheduler": scheduler_gen.state_dict(),
                    "scaler": scaler_gen.state_dict(),
                    "g_losses": g_losses,
                    "d_losses": d_losses,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                }, tmp_gen_path)

                torch.save({
                    "epoch": epoch,
                    "state_dict": critic.state_dict(),
                    "optimizer": opt_critic.state_dict(),
                    "scheduler": scheduler_critic.state_dict(),
                    "scaler": scaler_critic.state_dict()
                }, tmp_critic_path)

                shutil.move(tmp_gen_path, f"{CHECKPOINT_DIR}/gen_epoch_{epoch}.pth.tar")
                shutil.move(tmp_critic_path, f"{CHECKPOINT_DIR}/critic_epoch_{epoch}.pth.tar")
                cleanup_old_checkpoints(epoch)
                plot_and_save_losses(g_losses, d_losses)

    except KeyboardInterrupt:
        print("\nTraining manually interrupted by user. Saving final loss graph...")
        plot_and_save_losses(g_losses, d_losses)


if __name__ == "__main__":
    train()