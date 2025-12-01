import torch
import torch.nn as nn

# --- Configuration (Must match data_pipeline.py) ---
N_MELS = 80
TARGET_TIME_STEPS = 512
TEXT_EMBEDDING_DIM = 384
NOISE_DIM = 100  # Size of the random noise vector (z)


class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM, text_dim=TEXT_EMBEDDING_DIM, base_channels=512):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.text_dim = text_dim

        # 1. Input: Concatenate Noise + Text
        input_dim = noise_dim + text_dim

        # 2. Initial Linear Layer to create the "seed" feature map
        # We want to start from a small resolution and upsample.
        # Target: 80 x 512
        # We will upsample 4 times by a factor of 2.
        # 80 / 16 = 5
        # 512 / 16 = 32
        # So our starting size (bottom of the hourglass) is (5, 32)
        self.h_dim = 5
        self.w_dim = 32
        self.initial_channels = base_channels

        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.h_dim * self.w_dim * self.initial_channels),
            nn.BatchNorm1d(self.h_dim * self.w_dim * self.initial_channels),
            nn.ReLU(True)
        )

        # 3. Transposed Convolutions (Upsampling)
        # Each layer doubles the width and height
        self.main = nn.Sequential(
            # Block 1: (5, 32) -> (10, 64)
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(True),

            # Block 2: (10, 64) -> (20, 128)
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(True),

            # Block 3: (20, 128) -> (40, 256)
            nn.ConvTranspose2d(base_channels // 4, base_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 8),
            nn.ReLU(True),

            # Block 4: (40, 256) -> (80, 512)
            nn.ConvTranspose2d(base_channels // 8, 1, kernel_size=4, stride=2, padding=1),
            # No BatchNorm on output
            # We use Tanh because spectrograms (in dB) are roughly centered/normalized
            # Adjust normalization in data_pipeline if using Tanh (-1 to 1) vs Sigmoid (0 to 1)
            # For now, we output raw values (linear) or Tanh if data is scaled to [-1, 1]
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        # Concatenate inputs: [Batch, Noise] + [Batch, Text] -> [Batch, Noise+Text]
        x = torch.cat((noise, text_embedding), dim=1)

        # Project and reshape
        x = self.fc(x)
        x = x.view(-1, self.initial_channels, self.h_dim, self.w_dim)

        # Upsample
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, text_dim=TEXT_EMBEDDING_DIM, base_channels=64):
        super(Discriminator, self).__init__()

        # 1. Image Processing (Downsampling)
        # Input: (1, 80, 512)
        self.main = nn.Sequential(
            # Block 1: (80, 512) -> (40, 256)
            nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: (40, 256) -> (20, 128)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: (20, 128) -> (10, 64)
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: (10, 64) -> (5, 32)
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Calculated flattened size
        self.flattened_size = (base_channels * 8) * 5 * 32

        # 2. Condition Processing (Concatenation)
        # We flatten the image features and concatenate the text embedding
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size + text_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
            # No Sigmoid here if using Wasserstein Loss (recommended)
            # If using standard BCELoss, add Sigmoid()
        )

    def forward(self, spectrogram, text_embedding):
        # Extract features from image
        features = self.main(spectrogram)
        features = features.view(features.size(0), -1)  # Flatten

        # Concatenate text info
        combined = torch.cat((features, text_embedding), dim=1)

        # Output score
        prediction = self.classifier(combined)
        return prediction


def initialize_weights(m):
    """
    Custom weights initialization called on netG and netD.
    Standard practice for GANs to help convergence.
    """
    classname = m.__class__.__name__
    if 'Conv' in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


# --- SMOKE TEST BLOCK ---
if __name__ == "__main__":
    print("Starting Model Architecture Smoke Test...")

    # 1. Define Dummy Inputs (Batch Size = 2)
    batch_size = 2
    dummy_noise = torch.randn(batch_size, NOISE_DIM)
    dummy_text = torch.randn(batch_size, TEXT_EMBEDDING_DIM)
    dummy_spectrogram = torch.randn(batch_size, 1, N_MELS, TARGET_TIME_STEPS)

    print(f"Inputs:")
    print(f"  Noise: {dummy_noise.shape}")
    print(f"  Text:  {dummy_text.shape}")
    print(f"  Spec:  {dummy_spectrogram.shape}")

    try:
        # 2. Initialize Models
        print("\nInitializing Generator...")
        netG = Generator()
        netG.apply(initialize_weights)
        print("Generator Initialized.")

        print("Initializing Discriminator...")
        netD = Discriminator()
        netD.apply(initialize_weights)
        print("Discriminator Initialized.")

        # 3. Test Generator
        print("\nTesting Generator Forward Pass...")
        fake_spectrogram = netG(dummy_noise, dummy_text)
        print(f"  Output Shape: {fake_spectrogram.shape}")

        expected_shape = (batch_size, 1, N_MELS, TARGET_TIME_STEPS)
        if fake_spectrogram.shape == expected_shape:
            print("  SUCCESS: Generator output matches target dimensions.")
        else:
            print(f"  FAIL: Expected {expected_shape}, got {fake_spectrogram.shape}")

        # 4. Test Discriminator
        print("\nTesting Discriminator Forward Pass...")
        # Test with Real Data
        pred_real = netD(dummy_spectrogram, dummy_text)
        # Test with Fake Data (detached to mimic training loop)
        pred_fake = netD(fake_spectrogram.detach(), dummy_text)

        print(f"  Prediction Real Shape: {pred_real.shape}")
        print(f"  Prediction Fake Shape: {pred_fake.shape}")

        if pred_real.shape == (batch_size, 1) and pred_fake.shape == (batch_size, 1):
            print("  SUCCESS: Discriminator outputs single score per item.")
        else:
            print("  FAIL: Discriminator output shape is wrong.")

        print("\n--- MODEL ARCHITECTURE TEST PASSED ---")

    except Exception as e:
        print("\n--- TEST FAILED ---")
        print(e)
        import traceback

        traceback.print_exc()