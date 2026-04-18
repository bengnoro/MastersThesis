import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
import time
import gc

warnings.filterwarnings("ignore", category=FutureWarning)

from data_pipeline import (
    AudioTextDataset, collate_fn, N_MELS, TARGET_TIME_STEPS,
    EMBEDDING_DIM as TEXT_EMBEDDING_DIM, SAMPLE_RATE, N_FFT, HOP_LENGTH,
    DUMMY_CSV_FILE, DUMMY_AUDIO_DIR, TEXT_ENCODER_MODEL, ClapTextEncoder,
    DATASET_MEAN, DATASET_STD
)
from models import Generator, Discriminator, initialize_weights, NOISE_DIM

# hyperparameters
LEARNING_RATE_GEN = 5e-5
LEARNING_RATE_CRITIC = 2e-5
BATCH_SIZE = 12
Z_DIM = NOISE_DIM
FEATURES_CRITIC = 128
FEATURES_GEN = 256
CRITIC_ITERATIONS = 2
LAMBDA_FM = 0.1
START_EPOCH = 320
NUM_EPOCHS = 1001
SAVE_INTERVAL = 5
KEEP_LAST_CHECKPOINTS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def cleanup_old_checkpoints(current_epoch):
    epoch_to_delete = current_epoch - ((KEEP_LAST_CHECKPOINTS + 1) * SAVE_INTERVAL)
    if epoch_to_delete >= 0:
        for prefix in ["gen", "critic"]:
            path = f"checkpoints/{prefix}_epoch_{epoch_to_delete}.pth.tar"
            if os.path.exists(path): os.remove(path)


def save_spectrogram_image(gen_spec, epoch, batch_idx):
    os.makedirs("training_logs", exist_ok=True)

    spec_cpu = torch.clamp(gen_spec[0].squeeze().detach().cpu().float(), min=-1.0, max=1.0).numpy()

    spec_db = (spec_cpu * (3.0 * DATASET_STD)) + DATASET_MEAN

    vmin = DATASET_MEAN - (3.0 * DATASET_STD)
    vmax = DATASET_MEAN + (3.0 * DATASET_STD)

    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db, aspect='auto', origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(f"training_logs/epoch_{epoch}_batch_{batch_idx}.png")
    plt.close()


def plot_and_save_losses(g_losses, d_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(g_losses, label="Generator Loss", color="blue", alpha=0.8)
    plt.plot(d_losses, label="Critic Loss (Hinge)", color="red", alpha=0.8)
    plt.title("GAN Training Loss over Time")
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_graph.png")
    plt.close()


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
                print("\n[STARTUP OOM ALERT] Server is currently full. Hibernating for 60 seconds...")
                text_encoder = gen = critic = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(60)
            else:
                raise e

    dataset = AudioTextDataset(
        csv_file=DUMMY_CSV_FILE, audio_dir=DUMMY_AUDIO_DIR,
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, target_time_steps=TARGET_TIME_STEPS,
        text_encoder=text_encoder
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True,
                        num_workers=8, pin_memory=True)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.5, 0.9), eps=1e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC, betas=(0.5, 0.9), eps=1e-4)

    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_critic = torch.cuda.amp.GradScaler()

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

    if START_EPOCH > 0:
        print(f"Resuming training from Epoch {START_EPOCH}...")
        ckpt_gen = torch.load(f"checkpoints/gen_epoch_{START_EPOCH}.pth.tar", map_location=device, weights_only=True)

        gen_state = ckpt_gen["state_dict"]
        if "final_conv.conv.weight" in gen_state:
            gen_state["final_conv.0.conv.weight"] = gen_state.pop("final_conv.conv.weight")
        if "final_conv.conv.bias" in gen_state:
            gen_state["final_conv.0.conv.bias"] = gen_state.pop("final_conv.conv.bias")
        gen.load_state_dict(gen_state)
        opt_gen.load_state_dict(ckpt_gen["optimizer"])
        if "scheduler" in ckpt_gen:
            scheduler_gen.load_state_dict(ckpt_gen["scheduler"])
    else:
        initialize_weights(gen)
        initialize_weights(critic)

    fixed_noise = torch.randn(1, Z_DIM).to(device)
    fixed_text_emb = text_encoder.encode(["footsteps"], convert_to_tensor=True).to(device).clone()

    print("\nStarting SA-ResGAN Training...")

    try:
        for epoch in range(START_EPOCH, NUM_EPOCHS):
            noise_std = max(0.0, 0.1 * (1.0 - epoch / 500.0))

            for batch_idx, (real_spec, captions, precomputed_embs) in enumerate(loader):
                if real_spec is None: continue

                real_spec = real_spec.to(device)
                cur_batch_size = real_spec.shape[0]

                if cur_batch_size < 2:
                    continue

                if precomputed_embs is not None:
                    text_emb = precomputed_embs.to(device)
                else:
                    with torch.no_grad():
                        text_emb = text_encoder.encode(captions, convert_to_tensor=True).to(device)

                text_emb = text_emb.clone()

                while True:
                    try:
                        for step in range(CRITIC_ITERATIONS):
                            noise = torch.randn(cur_batch_size, Z_DIM).to(device)

                            with torch.no_grad():
                                with torch.cuda.amp.autocast():
                                    fake_spec = gen(noise, text_emb).detach()

                            opt_critic.zero_grad(set_to_none=True)

                            with torch.cuda.amp.autocast():
                                real_spec_noisy = real_spec + torch.randn_like(real_spec) * noise_std
                                fake_spec_noisy = fake_spec + torch.randn_like(fake_spec) * noise_std

                                critic_real, _ = critic(real_spec_noisy, text_emb)
                                critic_fake, _ = critic(fake_spec_noisy, text_emb)

                                critic_real = critic_real.reshape(-1)
                                critic_fake = critic_fake.reshape(-1)

                                loss_critic = torch.mean(F.relu(1.0 - critic_real.float())) + torch.mean(
                                    F.relu(1.0 + critic_fake.float()))

                            if torch.isnan(loss_critic) or torch.isinf(loss_critic):
                                print(
                                    f"Detected NaN/Inf in Critic Loss at Epoch {epoch}, Batch {batch_idx}. Skipping batch.")
                                opt_gen.zero_grad(set_to_none=True)
                                opt_critic.zero_grad(set_to_none=True)
                                raise ValueError("Loss Explosion")

                            scaler_critic.scale(loss_critic).backward()
                            scaler_critic.unscale_(opt_critic)
                            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
                            scaler_critic.step(opt_critic)
                            scaler_critic.update()

                        noise = torch.randn(cur_batch_size, Z_DIM).to(device)
                        opt_gen.zero_grad(set_to_none=True)

                        with torch.cuda.amp.autocast():
                            fake_spec = gen(noise, text_emb)

                            real_spec_noisy = real_spec + torch.randn_like(real_spec) * noise_std
                            fake_spec_noisy = fake_spec + torch.randn_like(fake_spec) * noise_std

                            with torch.no_grad():
                                _, real_features_fresh = critic(real_spec_noisy, text_emb)

                            critic_fake_out, fake_features = critic(fake_spec_noisy, text_emb)

                            loss_gen_adv = -torch.mean(critic_fake_out.float().reshape(-1))

                            loss_fm = 0.0
                            for f_real, f_fake in zip(real_features_fresh, fake_features):
                                loss_fm += F.l1_loss(f_fake.float(), f_real.float())

                            loss_gen = loss_gen_adv + (LAMBDA_FM * loss_fm)

                        if torch.isnan(loss_gen) or torch.isinf(loss_gen):
                            print(f"Detected NaN/Inf in Gen Loss at Epoch {epoch}, Batch {batch_idx}. Skipping batch.")
                            opt_gen.zero_grad(set_to_none=True)
                            opt_critic.zero_grad(set_to_none=True)
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
                                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \t Loss D: {loss_critic.item():.4f}, Loss G: {loss_gen.item():.4f} \t LR: {current_lr:.6f} \t (Batch Size: {cur_batch_size})")

                        if batch_idx == 0:
                            with torch.no_grad():
                                gen.eval()
                                with torch.cuda.amp.autocast():
                                    save_spectrogram_image(gen(fixed_noise, fixed_text_emb), epoch, batch_idx)
                                gen.train()

                        break

                    except ValueError:
                        break

                    except Exception as e:
                        err_msg = str(e)
                        if "out of memory" in err_msg.lower():
                            print(
                                f"\n[OOM ALERT] GPU Memory full. Waiting 60s to cleanly retry the exact same batch...")
                            del e
                            noise = fake_spec = critic_real = critic_fake = None
                            critic_fake_out = fake_features = real_features_fresh = None
                            loss_critic = loss_gen_adv = loss_fm = loss_gen = None
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
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    "state_dict": gen.state_dict(),
                    "optimizer": opt_gen.state_dict(),
                    "scheduler": scheduler_gen.state_dict()
                }, f"checkpoints/gen_epoch_{epoch}.pth.tar")

                torch.save({
                    "state_dict": critic.state_dict(),
                    "optimizer": opt_critic.state_dict(),
                    "scheduler": scheduler_critic.state_dict()
                }, f"checkpoints/critic_epoch_{epoch}.pth.tar")
                cleanup_old_checkpoints(epoch)

                plot_and_save_losses(g_losses, d_losses)

    except KeyboardInterrupt:
        print("\nTraining manually interrupted by user. Saving final loss graph...")
        plot_and_save_losses(g_losses, d_losses)


if __name__ == "__main__":
    train()