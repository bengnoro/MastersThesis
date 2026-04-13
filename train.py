import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import warnings

# Suppress harmless HuggingFace warnings and multiprocessing deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)

from data_pipeline import (
    AudioTextDataset, collate_fn, N_MELS, TARGET_TIME_STEPS,
    EMBEDDING_DIM as TEXT_EMBEDDING_DIM, SAMPLE_RATE, N_FFT, HOP_LENGTH,
    DUMMY_CSV_FILE, DUMMY_AUDIO_DIR, TEXT_ENCODER_MODEL
)
from models import Generator, Discriminator, initialize_weights, NOISE_DIM
from sentence_transformers import SentenceTransformer

# Hyperparameters
LEARNING_RATE = 1e-4
# Reduced batch size to safely fit 16GB VRAM with SA-ResGAN
BATCH_SIZE = 32
Z_DIM = NOISE_DIM
FEATURES_CRITIC = 64
FEATURES_GEN = 512
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
START_EPOCH = 0  # to resume, to start new use 0
NUM_EPOCHS = 1001
SAVE_INTERVAL = 5
KEEP_LAST_CHECKPOINTS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def compute_gradient_penalty(critic, real_data, fake_data, text_embeddings, device):
    """Calculates gradient penalty to enforce Lipschitz constraint (Gulrajani et al., 2017)."""
    BATCH_SIZE = real_data.shape[0]
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).to(device)
    interpolated = (epsilon * real_data + ((1 - epsilon) * fake_data)).requires_grad_(True)
    critic_interpolated = critic(interpolated, text_embeddings)

    gradients = torch.autograd.grad(
        inputs=interpolated, outputs=critic_interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True, retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty


def cleanup_old_checkpoints(current_epoch):
    epoch_to_delete = current_epoch - (KEEP_LAST_CHECKPOINTS * SAVE_INTERVAL)
    if epoch_to_delete >= 0:
        for prefix in ["gen", "critic"]:
            path = f"checkpoints/{prefix}_epoch_{epoch_to_delete}.pth.tar"
            if os.path.exists(path): os.remove(path)


def save_spectrogram_image(gen_spec, epoch, batch_idx):
    os.makedirs("training_logs", exist_ok=True)
    spec_cpu = gen_spec[0].squeeze().detach().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_cpu, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(f"training_logs/epoch_{epoch}_batch_{batch_idx}.png")
    plt.close()


def train():
    print("Loading Text Encoder directly to GPU...")
    text_encoder = SentenceTransformer(TEXT_ENCODER_MODEL).to(device)
    print("Initializing ESC-50 Dataset...")
    dataset = AudioTextDataset(
        csv_file=DUMMY_CSV_FILE, audio_dir=DUMMY_AUDIO_DIR,
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, target_time_steps=TARGET_TIME_STEPS
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True,
                        num_workers=16, pin_memory=True)
    gen = Generator(NOISE_DIM, TEXT_EMBEDDING_DIM, FEATURES_GEN).to(device)
    critic = Discriminator(TEXT_EMBEDDING_DIM, FEATURES_CRITIC).to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    if START_EPOCH > 0:
        print(f"Resuming training from Epoch {START_EPOCH}...")
        gen.load_state_dict(
            torch.load(f"checkpoints/gen_epoch_{START_EPOCH}.pth.tar", map_location=device, weights_only=True)[
                "state_dict"])
        critic.load_state_dict(
            torch.load(f"checkpoints/critic_epoch_{START_EPOCH}.pth.tar", map_location=device, weights_only=True)[
                "state_dict"])
    else:
        initialize_weights(gen)
        initialize_weights(critic)

    fixed_noise = torch.randn(1, Z_DIM).to(device)
    fixed_text_emb = text_encoder.encode(["footsteps"], convert_to_tensor=True).to(device)

    print("Starting SA-ResGAN Training...")
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        for batch_idx, (real_spec, captions) in enumerate(loader):
            if real_spec is None: continue
            real_spec = real_spec.to(device)
            cur_batch_size = real_spec.shape[0]
            with torch.no_grad():
                text_emb = text_encoder.encode(captions, convert_to_tensor=True).to(device)
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM).to(device)
                fake_spec = gen(noise, text_emb)

                critic_real = critic(real_spec, text_emb).reshape(-1)
                critic_fake = critic(fake_spec, text_emb).reshape(-1)

                gp = compute_gradient_penalty(critic, real_spec, fake_spec, text_emb, device)
                loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp)

                opt_critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            gen_fake = critic(fake_spec, text_emb).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 25 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \t Loss D: {loss_critic.item():.4f}, Loss G: {loss_gen.item():.4f}")

            if batch_idx == 0:
                with torch.no_grad():
                    gen.eval()
                    save_spectrogram_image(gen(fixed_noise, fixed_text_emb), epoch, batch_idx)
                    gen.train()

        if epoch % SAVE_INTERVAL == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"state_dict": gen.state_dict(), "optimizer": opt_gen.state_dict()},
                       f"checkpoints/gen_epoch_{epoch}.pth.tar")
            torch.save({"state_dict": critic.state_dict(), "optimizer": opt_critic.state_dict()},
                       f"checkpoints/critic_epoch_{epoch}.pth.tar")
            cleanup_old_checkpoints(epoch)


if __name__ == "__main__":
    train()