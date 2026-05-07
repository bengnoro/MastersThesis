import os
import sys
import glob
import re
import time
import csv

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import matplotlib.pyplot as plt
import warnings
import gc
import multiprocessing
import shutil

warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from data_pipeline import (
    AudioTextDataset, collate_fn, N_MELS, TARGET_TIME_STEPS,
    EMBEDDING_DIM as TEXT_EMBEDDING_DIM, SAMPLE_RATE, N_FFT, HOP_LENGTH,
    DUMMY_CSV_FILE, DUMMY_AUDIO_DIR, TEXT_ENCODER_MODEL, ClapTextEncoder,
    DATASET_MEAN, DATASET_STD, F_MIN, F_MAX, CENTER
)
from models import Generator, Critic, NOISE_DIM


class ExponentialMovingAverage:
    """
    Maintains moving averages of model parameters to stabilize training.
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """
        Stores the initial parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Applies decay to the moving averages based on the current parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        Temporarily swaps out active parameters for the moving averages.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """
        Restores the active parameters after evaluation.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def enforce_a100_and_get_batch_size():
    """
    Checks GPU capacity and enforces specific batch sizing based on memory limits.
    """
    if not torch.cuda.is_available():
        print("Fatal error: No GPU detected. Terminating script.")
        sys.exit(1)

    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)

    print(f"Hardware detected. GPU VRAM: {vram_gb:.1f} GB.")

    if vram_gb >= 35.0:
        print("A100 Architecture confirmed. Setting Batch Size to 64.")
        return 64
    else:
        print("Fatal error: Insufficient GPU VRAM.")
        sys.exit(1)


LEARNING_RATE_GEN = 5e-5
LEARNING_RATE_CRITIC = 5e-5
BATCH_SIZE = enforce_a100_and_get_batch_size()
Z_DIM = NOISE_DIM

CRITIC_ITERATIONS = 3
LAMBDA_R1 = 0.5

NUM_EPOCHS = 550
SAVE_INTERVAL = 5
KEEP_LAST_CHECKPOINTS = 3000

CHECKPOINT_DIR = "/content/drive/MyDrive/diplomka/checkpoints"
LOG_DIR = "/content/drive/MyDrive/diplomka/logs"
TELEMETRY_CSV = os.path.join(LOG_DIR, "training_telemetry.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def diff_spec_augment(x):
    """
    Applies random time and frequency masking to the spectrograms for data augmentation.
    """
    mask = torch.ones_like(x, requires_grad=False)
    batch_size, channels, freq_dim, time_dim = x.shape
    for i in range(batch_size):
        f_len = torch.randint(0, freq_dim // 5, (1,)).item()
        f_st = torch.randint(0, freq_dim - f_len + 1, (1,)).item()
        mask[i, :, f_st:f_st + f_len, :] = 0.0

        t_len = torch.randint(0, time_dim // 10, (1,)).item()
        t_st = torch.randint(0, time_dim - t_len + 1, (1,)).item()
        mask[i, :, :, t_st:t_st + t_len] = 0.0
    return x * mask


def cleanup_old_checkpoints(current_epoch):
    """
    Deletes older checkpoints to save disk space, preserving milestones.
    """
    epoch_to_delete = current_epoch - ((KEEP_LAST_CHECKPOINTS + 1) * SAVE_INTERVAL)
    if epoch_to_delete >= 0 and epoch_to_delete % 50 != 0:
        for prefix in ["gen", "critic"]:
            path = f"{CHECKPOINT_DIR}/{prefix}_epoch_{epoch_to_delete}.pth.tar"
            if os.path.exists(path):
                os.remove(path)


def save_spectrogram_image(gen_spec, epoch, label):
    """
    Converts a generated tensor into a visual representation and saves it to disk.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    spec_cpu = torch.clamp(gen_spec[0].squeeze().detach().cpu().float(), min=-1.0, max=1.0).numpy()
    spec_db = (spec_cpu * (3.0 * DATASET_STD)) + DATASET_MEAN
    vmin = DATASET_MEAN - (3.0 * DATASET_STD)
    vmax = DATASET_MEAN + (3.0 * DATASET_STD)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spec_db, aspect='auto', origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, format='%+2.0f dB')
    ax.set_title(f"Generated Image: {label}")
    fig.tight_layout()
    fig.savefig(f"{LOG_DIR}/epoch_{epoch}_{label}.png")
    fig.clf()
    plt.close('all')
    gc.collect()


def plot_and_save_losses(g_losses, d_losses):
    """
    Generates a line chart displaying generator and critic losses over time.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(g_losses, label="Generator Loss", color="blue", alpha=0.8)
    ax.plot(d_losses, label="Critic Loss", color="red", alpha=0.8)
    ax.set_title("Training Loss")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{LOG_DIR}/training_loss_graph.png")
    fig.clf()
    plt.close('all')
    gc.collect()


def calculate_gradient_norm(model):
    """
    Computes the overall norm of the model's gradients for monitoring stability.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def find_latest_checkpoint(ckpt_dir):
    """
    Locates the most recent training checkpoint in the given directory.
    """
    gen_files = glob.glob(os.path.join(ckpt_dir, "gen_epoch_*.pth.tar"))
    if not gen_files:
        return 0
    epochs = [int(re.search(r"epoch_(\d+)", f).group(1)) for f in gen_files]
    max_epoch = max(epochs)
    if os.path.exists(os.path.join(ckpt_dir, f"critic_epoch_{max_epoch}.pth.tar")):
        return max_epoch
    return 0


def train():
    """
    Main training loop for the GAN, handling data loading, optimization, and check-pointing.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Device set: {device}")

    while True:
        try:
            print("Loading models into VRAM...")
            text_encoder = ClapTextEncoder(TEXT_ENCODER_MODEL, device=device)

            gen = Generator(noise_dim=NOISE_DIM, text_dim=TEXT_EMBEDDING_DIM).to(device)
            critic = Critic().to(device)
            ema = ExponentialMovingAverage(gen, decay=0.999)

            if torch.cuda.is_available():
                _test_tensor = torch.zeros((256, 1024, 1024), device=device)
                del _test_tensor
                torch.cuda.empty_cache()
            print("VRAM loaded successfully.")
            break
        except Exception as e:
            if "out of memory" in str(e).lower():
                print("Warning: VRAM full. Retrying in 60 seconds...")
                text_encoder = gen = critic = ema = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                        num_workers=optimal_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.5, 0.9), eps=1e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC, betas=(0.5, 0.9), eps=1e-4)

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
        print(f"Resuming training from Epoch {latest_epoch}...")
        ckpt_gen = torch.load(f"{CHECKPOINT_DIR}/gen_epoch_{latest_epoch}.pth.tar", map_location=device, weights_only=True)

        if "training_state_dict" in ckpt_gen:
            gen.load_state_dict(ckpt_gen["training_state_dict"])
        else:
            gen.load_state_dict(ckpt_gen["state_dict"])

        opt_gen.load_state_dict(ckpt_gen["optimizer"])
        if "g_losses" in ckpt_gen: g_losses = ckpt_gen["g_losses"]
        if "d_losses" in ckpt_gen: d_losses = ckpt_gen["d_losses"]
        if "ema_shadow" in ckpt_gen: ema.shadow = ckpt_gen["ema_shadow"]
        if "torch_rng" in ckpt_gen: torch.set_rng_state(ckpt_gen["torch_rng"].cpu())
        if "cuda_rng" in ckpt_gen: torch.cuda.set_rng_state_all([state.cpu() for state in ckpt_gen["cuda_rng"]])

        ckpt_critic = torch.load(f"{CHECKPOINT_DIR}/critic_epoch_{latest_epoch}.pth.tar", map_location=device, weights_only=True)
        critic.load_state_dict(ckpt_critic["state_dict"])
        opt_critic.load_state_dict(ckpt_critic["optimizer"])

        for param_group in opt_gen.param_groups:
            param_group['lr'] = LEARNING_RATE_GEN
        for param_group in opt_critic.param_groups:
            param_group['lr'] = LEARNING_RATE_CRITIC

        scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda)
        scheduler_critic = optim.lr_scheduler.LambdaLR(opt_critic, lr_lambda)

        START_EPOCH = latest_epoch + 1
    else:
        print("Starting new training from Epoch 0.")
        ema.register()
        START_EPOCH = 0

    test_rng = torch.Generator(device=device)
    test_rng.manual_seed(42)
    fixed_noise_1 = torch.randn(1, Z_DIM, generator=test_rng, device=device)
    fixed_text_emb_1 = text_encoder.encode(["The sound of dog bark."], convert_to_tensor=True).to(device).clone()
    fixed_noise_2 = torch.randn(1, Z_DIM, generator=test_rng, device=device)
    fixed_text_emb_2 = text_encoder.encode(["The sound of footstep."], convert_to_tensor=True).to(device).clone()

    print("Training started.")

    try:
        for epoch in range(START_EPOCH, NUM_EPOCHS):
            noise_std = max(0.0, 0.1 * (1.0 - epoch / 500.0))
            epoch_start_time = time.time()

            for batch_idx, (real_waveforms, captions, precomputed_embs) in enumerate(loader):
                if real_waveforms is None:
                    continue

                real_waveforms = real_waveforms.to(device)
                cur_batch_size = real_waveforms.shape[0]
                if cur_batch_size < 2:
                    continue

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
                        for _ in range(CRITIC_ITERATIONS):
                            opt_critic.zero_grad(set_to_none=True)
                            noise = torch.randn(cur_batch_size, Z_DIM).to(device)

                            real_spec_noisy = real_spec + torch.randn_like(real_spec) * noise_std
                            real_spec_noisy.requires_grad_(True)

                            with torch.no_grad():
                                fake_spec = gen(noise, text_emb).detach()

                            fake_spec_noisy = fake_spec + torch.randn_like(fake_spec) * noise_std

                            aug_real = diff_spec_augment(real_spec_noisy)
                            aug_fake = diff_spec_augment(fake_spec_noisy)

                            critic_real = critic(aug_real, text_emb)
                            critic_fake = critic(aug_fake, text_emb)

                            critic_real_flat = critic_real.reshape(-1)
                            critic_fake_flat = critic_fake.reshape(-1)

                            loss_critic_adv = torch.mean(F.relu(1.0 - critic_real_flat)) + torch.mean(F.relu(1.0 + critic_fake_flat))

                            grad_real, = torch.autograd.grad(
                                outputs=critic_real.sum(),
                                inputs=real_spec_noisy,
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True
                            )

                            grad_penalty = (grad_real ** 2).reshape(grad_real.size(0), -1).sum(1).mean()
                            loss_critic = loss_critic_adv + (LAMBDA_R1 / 2) * grad_penalty

                            if torch.isnan(loss_critic) or torch.isinf(loss_critic):
                                raise ValueError("Loss calculation failed.")

                            loss_critic.backward()
                            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)

                            norm_critic = calculate_gradient_norm(critic)
                            opt_critic.step()

                        noise = torch.randn(cur_batch_size, Z_DIM).to(device)
                        opt_gen.zero_grad(set_to_none=True)

                        fake_spec = gen(noise, text_emb)

                        fake_spec_noisy = fake_spec + torch.randn_like(fake_spec) * noise_std
                        aug_fake = diff_spec_augment(fake_spec_noisy)

                        critic_fake_out = critic(aug_fake, text_emb)
                        loss_gen = -torch.mean(critic_fake_out.reshape(-1))

                        if torch.isnan(loss_gen) or torch.isinf(loss_gen):
                            raise ValueError("Loss calculation failed.")

                        loss_gen.backward()
                        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=10.0)

                        norm_gen = calculate_gradient_norm(gen)
                        opt_gen.step()

                        ema.update()

                        g_losses.append(loss_gen.item())
                        d_losses.append(loss_critic.item())

                        if batch_idx % 25 == 0:
                            elapsed = time.time() - epoch_start_time
                            batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                            print(f"Epoch: {epoch}/{NUM_EPOCHS} | Batch: {batch_idx}/{len(loader)} | "
                                  f"Loss D: {loss_critic.item():.3f} | Loss G: {loss_gen.item():.3f} | "
                                  f"Norm D: {norm_critic:.2f} | Norm G: {norm_gen:.2f} | "
                                  f"Speed: {batches_per_sec:.2f} it/s")

                            file_exists = os.path.isfile(TELEMETRY_CSV)
                            with open(TELEMETRY_CSV, mode='a', newline='') as f:
                                writer = csv.writer(f)
                                if not file_exists:
                                    writer.writerow(['Epoch', 'Batch', 'Loss_D', 'Loss_G', 'Norm_D', 'Norm_G', 'IOPS'])
                                writer.writerow([epoch, batch_idx, round(loss_critic.item(), 4), round(loss_gen.item(), 4),
                                                 round(norm_critic, 4), round(norm_gen, 4), round(batches_per_sec, 2)])

                        if batch_idx == 0:
                            with torch.no_grad():
                                ema.apply_shadow()
                                gen.eval()
                                save_spectrogram_image(gen(fixed_noise_1, fixed_text_emb_1), epoch, "dog_bark")
                                save_spectrogram_image(gen(fixed_noise_2, fixed_text_emb_2), epoch, "footstep")
                                gen.train()
                                ema.restore()

                        del fake_spec, real_spec_noisy, fake_spec_noisy, aug_real, aug_fake
                        del critic_real, critic_fake, grad_real, grad_penalty
                        del critic_fake_out, loss_critic, loss_gen
                        success = True

                    except ValueError:
                        print(f"Issue in Epoch {epoch}, Batch {batch_idx}. Retrying.")
                        opt_gen.zero_grad(set_to_none=True)
                        opt_critic.zero_grad(set_to_none=True)
                        continue

                    except Exception as e:
                        if "out of memory" in str(e).lower():
                            print("Warning: GPU Memory full. Retrying batch in 60s.")
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

                active_training_state = gen.state_dict()

                ema.apply_shadow()
                ema_inference_state = gen.state_dict()
                ema.restore()

                torch.save({
                    "epoch": epoch,
                    "state_dict": ema_inference_state,
                    "training_state_dict": active_training_state,
                    "optimizer": opt_gen.state_dict(),
                    "scheduler": scheduler_gen.state_dict(),
                    "g_losses": g_losses,
                    "d_losses": d_losses,
                    "ema_shadow": ema.shadow,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                }, tmp_gen_path)

                torch.save({
                    "epoch": epoch,
                    "state_dict": critic.state_dict(),
                    "optimizer": opt_critic.state_dict(),
                    "scheduler": scheduler_critic.state_dict(),
                }, tmp_critic_path)

                shutil.move(tmp_gen_path, f"{CHECKPOINT_DIR}/gen_epoch_{epoch}.pth.tar")
                shutil.move(tmp_critic_path, f"{CHECKPOINT_DIR}/critic_epoch_{epoch}.pth.tar")
                cleanup_old_checkpoints(epoch)
                plot_and_save_losses(g_losses, d_losses)

    except KeyboardInterrupt:
        print("Training manually stopped. Saving loss graph.")
        plot_and_save_losses(g_losses, d_losses)


if __name__ == "__main__":
    train()

    print("Process complete. Attempting to disconnect runtime.")
    try:
        from google.colab import runtime
        runtime.unassign()
    except ImportError:
        pass