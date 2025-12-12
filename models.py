import torch
import torch.nn as nn

N_MELS = 80
TARGET_TIME_STEPS = 512
TEXT_EMBEDDING_DIM = 384
NOISE_DIM = 100


class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM, text_dim=TEXT_EMBEDDING_DIM, base_channels=512):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.text_dim = text_dim

        input_dim = noise_dim + text_dim

        self.h_dim = 5
        self.w_dim = 32
        self.initial_channels = base_channels

        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.h_dim * self.w_dim * self.initial_channels),
            nn.BatchNorm1d(self.h_dim * self.w_dim * self.initial_channels),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            # Block 1
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(True),

            # Block 2
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(True),

            # Block 3
            nn.ConvTranspose2d(base_channels // 4, base_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 8),
            nn.ReLU(True),

            # Block 4
            nn.ConvTranspose2d(base_channels // 8, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        x = torch.cat((noise, text_embedding), dim=1)
        x = self.fc(x)
        x = x.view(-1, self.initial_channels, self.h_dim, self.w_dim)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, text_dim=TEXT_EMBEDDING_DIM, base_channels=64):
        super(Discriminator, self).__init__()


        self.main = nn.Sequential(
            # Block 1
            nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2, affine=True),  # <--- CHANGED
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4, affine=True),  # <--- CHANGED
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 8, affine=True),  # <--- CHANGED
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flattened_size = (base_channels * 8) * 5 * 32

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size + text_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, spectrogram, text_embedding):
        features = self.main(spectrogram)
        features = features.view(features.size(0), -1)
        combined = torch.cat((features, text_embedding), dim=1)
        prediction = self.classifier(combined)
        return prediction


def initialize_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)
