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

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
Z_DIM = NOISE_DIM
FEATURES_CRITIC = 64
FEATURES_GEN = 512
CRITIC_ITERATIONS = 2
LAMBDA_GP = 10.0
LAMBDA_FM = 2.0
START_EPOCH = 0
NUM_EPOCHS = 1001
SAVE_INTERVAL = 5
KEEP_LAST_CHECKPOINTS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_gradient_penalty(critic, real_data, fake_data, text_embeddings, device):
    cur_batch_size = real_data.shape[0]
    epsilon = torch.rand((cur_batch_size, 1, 1, 1)).to(device)

    interpolated = (epsilon * real_data + ((1 - epsilon) * fake_data)).requires_grad_(True)

    with torch.cuda.amp.autocast(enabled=False):
        critic_interpolated, _ = critic(interpolated.float(), text_embeddings.float())

        gradients = torch.autograd.grad(
            inputs=interpolated, outputs=critic_interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True, retain_graph=True,
        )[0]

    gradients = gradients.view(gradients.shape[0], -1)

    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = torch.mean((gradients_norm - 1) ** 2)
    return gradient_penalty


def cleanup_old_checkpoints(current_epoch):
    epoch_to_delete = current_epoch - ((KEEP_LAST_CHECKPOINTS + 1) * SAVE_INTERVAL)
    if epoch_to_delete >= 0:
        for prefix in ["gen", "critic"]:
            path = f"checkpoints/{prefix}_epoch_{epoch_to_delete}.pth.tar"
            if os.path.exists(path): os.remove(path)


def save_spectrogram_image(gen_spec, epoch, batch_idx):
    os.makedirs("training_logs", exist_ok=True)
    spec_cpu = gen_spec[0].squeeze().detach().cpu().float().numpy()

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
    plt.plot(d_losses, label="Critic Loss (Wasserstein)", color="red", alpha=0.8)
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
                print("\n[STARTUP OOM ALERT] Server is currently full. Hibernating for 5 minutes")
                text_encoder = gen = critic = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(60*5)
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

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9), eps=1e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9), eps=1e-4)

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
        gen.load_state_dict(ckpt_gen["state_dict"])
        opt_gen.load_state_dict(ckpt_gen["optimizer"])
        if "scheduler" in ckpt_gen:
            scheduler_gen.load_state_dict(ckpt_gen["scheduler"])

        ckpt_critic = torch.load(f"checkpoints/critic_epoch_{START_EPOCH}.pth.tar", map_location=device,
                                 weights_only=True)
        critic.load_state_dict(ckpt_critic["state_dict"])
        opt_critic.load_state_dict(ckpt_critic["optimizer"])
        if "scheduler" in ckpt_critic:
            scheduler_critic.load_state_dict(ckpt_critic["scheduler"])
    else:
        initialize_weights(gen)
        initialize_weights(critic)

    fixed_noise = torch.randn(1, Z_DIM).to(device)
    fixed_text_emb = text_encoder.encode(["footsteps"], convert_to_tensor=True).to(device).clone()
    fixed_text_emb = F.normalize(fixed_text_emb, p=2, dim=1)

    print("\nStarting SA-ResGAN Training...")

    try:
        for epoch in range(START_EPOCH, NUM_EPOCHS):
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
                text_emb = F.normalize(text_emb, p=2, dim=1)

                while True:
                    try:
                        # 1. Train the Critic
                        last_real_features = None
                        for step in range(CRITIC_ITERATIONS):
                            noise = torch.randn(cur_batch_size, Z_DIM).to(device)

                            with torch.no_grad():
                                with torch.cuda.amp.autocast():
                                    fake_spec = gen(noise, text_emb)

                            opt_critic.zero_grad(set_to_none=True)

                            with torch.cuda.amp.autocast():
                                critic_real, real_features = critic(real_spec, text_emb)
                                critic_fake, _ = critic(fake_spec, text_emb)

                                critic_real = critic_real.reshape(-1)
                                critic_fake = critic_fake.reshape(-1)

                            gp = compute_gradient_penalty(critic, real_spec, fake_spec, text_emb, device)
                            loss_critic = (-(torch.mean(critic_real.float()) - torch.mean(
                                critic_fake.float())) + LAMBDA_GP * gp)

                            scaler_critic.scale(loss_critic).backward()
                            scaler_critic.step(opt_critic)
                            scaler_critic.update()

                            if step == CRITIC_ITERATIONS - 1:
                                last_real_features = [f.detach() for f in real_features]

                        # 2. Train the Generator
                        noise = torch.randn(cur_batch_size, Z_DIM).to(device)
                        opt_gen.zero_grad(set_to_none=True)

                        with torch.cuda.amp.autocast():
                            fake_spec = gen(noise, text_emb)
                            critic_fake_out, fake_features = critic(fake_spec, text_emb)

                        loss_gen_adv = -torch.mean(critic_fake_out.float().reshape(-1))

                        loss_fm = 0.0
                        for f_real, f_fake in zip(last_real_features, fake_features):
                            loss_fm += F.l1_loss(f_fake.float(), f_real.float())

                        loss_gen = loss_gen_adv + (LAMBDA_FM * loss_fm)

                        scaler_gen.scale(loss_gen).backward()
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

                    except Exception as e:
                        err_msg = str(e)
                        if "out of memory" in err_msg.lower():
                            print(
                                f"\n[OOM ALERT] GPU Memory full. Waiting 120s to cleanly retry the exact same batch...")

                            del e

                            noise = fake_spec = critic_real = critic_fake = gp = None
                            loss_critic = critic_fake_out = fake_features = real_features = last_real_features = None
                            loss_gen_adv = loss_fm = loss_gen = None

                            opt_critic.zero_grad(set_to_none=True)
                            opt_gen.zero_grad(set_to_none=True)

                            gc.collect()
                            torch.cuda.empty_cache()

                            time.sleep(120)
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
