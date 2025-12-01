import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Local Imports ---
# Fixed Imports:
# 1. Renamed EMBEDDING_DIM to TEXT_EMBEDDING_DIM to match train.py usage
# 2. Removed NOISE_DIM from data_pipeline (it belongs to models or config)
from data_pipeline import (
    AudioTextDataset,
    collate_fn,
    N_MELS,
    TARGET_TIME_STEPS,
    EMBEDDING_DIM as TEXT_EMBEDDING_DIM,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH
)

# Imported NOISE_DIM from models where it is defined
from models import Generator, Discriminator, initialize_weights, NOISE_DIM

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
CHANNELS_IMG = 1
Z_DIM = NOISE_DIM
NUM_EPOCHS = 10
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5  # Train Discriminator 5 times for every 1 Generator step
LAMBDA_GP = 10  # Gradient penalty lambda

# --- Device Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def compute_gradient_penalty(critic, real_data, fake_data, text_embeddings, device):
    """
    Calculates the Gradient Penalty (GP) for WGAN-GP.
    This enforces the 1-Lipschitz constraint, allowing the GAN to learn stably.
    """
    BATCH_SIZE, C, H, W = real_data.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).to(device)

    # Interpolate between real and fake
    interpolated = (epsilon * real_data + ((1 - epsilon) * fake_data)).requires_grad_(True)

    # Get critic scores
    critic_interpolated = critic(interpolated, text_embeddings)

    # Calculate gradients of scores with respect to the interpolated examples
    gradients = torch.autograd.grad(
        inputs=interpolated,
        outputs=critic_interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten gradients
    gradients = gradients.view(gradients.shape[0], -1)

    # Calculate penalty: (norm(gradients) - 1)^2
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def save_spectrogram_image(gen_spec, epoch, batch_idx):
    """Saves a generated spectrogram as a PNG for visual monitoring."""
    if not os.path.exists("training_logs"):
        os.makedirs("training_logs")

    spec_cpu = gen_spec[0].squeeze().detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(spec_cpu, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Generated Epoch {epoch} Batch {batch_idx}")
    plt.tight_layout()
    plt.savefig(f"training_logs/epoch_{epoch}_batch_{batch_idx}.png")
    plt.close()


# --- Main Training Loop ---
def train():
    # 1. Setup Data
    # Import paths from data_pipeline config (ensure these are set correctly in data_pipeline.py)
    from data_pipeline import DUMMY_CSV_FILE, DUMMY_AUDIO_DIR, TEXT_ENCODER_MODEL
    from sentence_transformers import SentenceTransformer

    print("Loading Text Encoder...")
    text_encoder = SentenceTransformer(TEXT_ENCODER_MODEL)

    print("Initializing Dataset...")
    dataset = AudioTextDataset(
        csv_file=DUMMY_CSV_FILE,  # Make sure this points to your REAL csv in data_pipeline.py
        audio_dir=DUMMY_AUDIO_DIR,  # Make sure this points to your REAL audio folder
        text_encoder=text_encoder,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        target_time_steps=TARGET_TIME_STEPS
    )

    # Debug info
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False  # Changed to False so small datasets/last batches don't get skipped
    )

    print(f"Loader batches per epoch: {len(loader)}")

    # 2. Setup Models
    gen = Generator(NOISE_DIM, TEXT_EMBEDDING_DIM, FEATURES_GEN).to(device)
    critic = Discriminator(TEXT_EMBEDDING_DIM, FEATURES_CRITIC).to(device)

    initialize_weights(gen)
    initialize_weights(critic)

    # 3. Optimizers
    # WGAN uses Adam with Beta1=0.0 and Beta2=0.9 usually
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    # Fixed noise for visualization (to see how the SAME seed improves over time)
    fixed_noise = torch.randn(1, Z_DIM).to(device)
    # Grab a real text embedding for visualization
    try:
        # Just grab the first text embedding from the dataset for testing
        _, fixed_text_emb = dataset[0]
        fixed_text_emb = fixed_text_emb.unsqueeze(0).to(device)
    except:
        print("Warning: Could not grab text from dataset for fixed noise. Using random text.")
        fixed_text_emb = torch.randn(1, TEXT_EMBEDDING_DIM).to(device)

    gen.train()
    critic.train()

    print("Starting Training...")

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real_spec, text_emb) in enumerate(loader):

            # Move data to device
            real_spec = real_spec.to(device)
            text_emb = text_emb.to(device)
            cur_batch_size = real_spec.shape[0]

            # ---------------------
            # Train Discriminator (Critic)
            # ---------------------
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM).to(device)
                fake_spec = gen(noise, text_emb)

                # Critic scores (no sigmoid!)
                critic_real = critic(real_spec, text_emb).reshape(-1)
                critic_fake = critic(fake_spec, text_emb).reshape(-1)

                # Gradient Penalty
                gp = compute_gradient_penalty(critic, real_spec, fake_spec, text_emb, device)

                # WGAN Loss: D(fake) - D(real) + lambda * gp
                # We want to maximize D(real) - D(fake), which is minimizing -(D(real) - D(fake))
                loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )

                opt_critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # ---------------------
            # Train Generator
            # ---------------------
            # We want to maximize D(fake) -> minimize -D(fake)
            gen_fake = critic(fake_spec, text_emb).reshape(-1)
            loss_gen = -torch.mean(gen_fake)

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # ---------------------
            # Logging & Visualization
            # ---------------------
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \t "
                    f"Loss D: {loss_critic.item():.4f}, Loss G: {loss_gen.item():.4f}"
                )

            # SAVE IMAGE ON THE FIRST BATCH OF EVERY EPOCH
            if batch_idx == 0:
                with torch.no_grad():
                    gen.eval()  # Switch to eval mode to handle single-item batch (BatchNorm fix)
                    fake = gen(fixed_noise, fixed_text_emb)
                    # Save the spectrogram image
                    save_spectrogram_image(fake, epoch, batch_idx)
                    print(f"Saved monitoring image to training_logs/epoch_{epoch}_batch_{batch_idx}.png")
                    gen.train()  # Switch back to train mode

        # Save Checkpoint every 5 epochs
        if epoch % 5 == 0:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            save_checkpoint(gen, opt_gen, filename=f"checkpoints/gen_epoch_{epoch}.pth.tar")
            save_checkpoint(critic, opt_critic, filename=f"checkpoints/critic_epoch_{epoch}.pth.tar")


if __name__ == "__main__":
    train()