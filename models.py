import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Conf
N_MELS = 100
TARGET_TIME_STEPS = 512
TEXT_EMBEDDING_DIM = 384
NOISE_DIM = 100


class SelfAttention(nn.Module):
    """Self-Attention Layer (Zhang et al., 2018) to learn long-range audio rhythms."""

    def __init__(self, in_channels):
        super().__init__()
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        q = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, width * height)
        v = self.value(x).view(batch_size, -1, width * height)
        attention = torch.bmm(q, k)
        attention = torch.softmax(attention, dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma * out + x


class GeneratorResBlock(nn.Module):
    """Residual Block utilizing bilinear upsampling to prevent high-frequency checkerboard artifacts."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        res = self.shortcut(x)
        x = self.upsample(x)
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return nn.functional.relu(x + res)


class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM, text_dim=TEXT_EMBEDDING_DIM, base_channels=512):
        super().__init__()
        input_dim = noise_dim + text_dim
        self.h_dim = 6
        self.w_dim = 32
        self.initial_channels = base_channels
        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.h_dim * self.w_dim * self.initial_channels),
            nn.BatchNorm1d(self.h_dim * self.w_dim * self.initial_channels),
            nn.ReLU(True))
        self.res1 = GeneratorResBlock(base_channels, base_channels // 2)
        self.attn = SelfAttention(base_channels // 2)
        self.res2 = GeneratorResBlock(base_channels // 2, base_channels // 4)
        self.res3 = GeneratorResBlock(base_channels // 4, base_channels // 8)
        self.res4 = GeneratorResBlock(base_channels // 8, base_channels // 16)

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels // 16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh())

    def forward(self, noise, text_embedding):
        x = torch.cat((noise, text_embedding), dim=1)
        x = self.fc(x)
        x = x.view(-1, self.initial_channels, self.h_dim, self.w_dim)
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.final_conv(x)

        padding = (0, 0, 2, 2)
        x = nn.functional.pad(x, padding, "constant", -1.0)
        return x


class Discriminator(nn.Module):
    def __init__(self, text_dim=TEXT_EMBEDDING_DIM, base_channels=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(1, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True))
        self.attn = SelfAttention(base_channels * 4)

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 8, base_channels * 16, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True))

        self.flattened_size = (base_channels * 16) * 3 * 16

        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(self.flattened_size + text_dim, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(512, 1)))

    def forward(self, spectrogram, text_embedding):
        x = self.conv1(spectrogram)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attn(x)
        x = self.conv4(x)
        x = self.conv5(x)

        features = x.view(x.size(0), -1)
        combined = torch.cat((features, text_embedding), dim=1)
        return self.classifier(combined)


def initialize_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname and 'SelfAttention' not in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)