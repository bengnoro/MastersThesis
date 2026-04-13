import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math

# Config
N_MELS = 80
TARGET_TIME_STEPS = 512
TEXT_EMBEDDING_DIM = 512
NOISE_DIM = 100


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        batch_size, _, h, w = x.shape
        y_coords = torch.linspace(-1, 1, steps=h, device=x.device).view(1, 1, h, 1).expand(batch_size, 1, h, w)
        x_coords = torch.linspace(-1, 1, steps=w, device=x.device).view(1, 1, 1, w).expand(batch_size, 1, h, w)
        x = torch.cat([x, y_coords, x_coords], dim=1)
        return self.conv(x)


class ConditionalGroupNorm2d(nn.Module):
    def __init__(self, num_features, num_groups=32, cond_dim=TEXT_EMBEDDING_DIM + NOISE_DIM):
        super().__init__()
        self.num_features = num_features

        self.num_groups = min(num_groups, num_features)
        if self.num_features % self.num_groups != 0:
            self.num_groups = 1

        self.norm = nn.GroupNorm(self.num_groups, num_features, affine=False)
        self.gamma_embed = nn.Linear(cond_dim, num_features)
        self.beta_embed = nn.Linear(cond_dim, num_features)

        nn.init.zeros_(self.gamma_embed.weight)
        nn.init.ones_(self.gamma_embed.bias)
        nn.init.zeros_(self.beta_embed.weight)
        nn.init.zeros_(self.beta_embed.bias)

    def forward(self, x, c):
        out = self.norm(x)
        gamma = self.gamma_embed(c).view(-1, self.num_features, 1, 1)
        beta = self.beta_embed(c).view(-1, self.num_features, 1, 1)
        return gamma * out + beta


class SelfAttention(nn.Module):
    def __init__(self, in_channels, use_sn=True):
        super().__init__()

        def sn_wrapper(module):
            return spectral_norm(module) if use_sn else module

        self.query = sn_wrapper(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = sn_wrapper(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = sn_wrapper(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        q = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, width * height)
        v = self.value(x).view(batch_size, -1, width * height)

        d = max(1, C // 8)
        attention = torch.bmm(q, k) / math.sqrt(d)
        attention = torch.softmax(attention, dim=-1)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma * out + x


class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cgn1 = ConditionalGroupNorm2d(in_channels)
        self.conv1 = CoordConv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

        self.cgn2 = ConditionalGroupNorm2d(out_channels)
        self.conv2 = CoordConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, c, target_size):
        res = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        res = self.shortcut_conv(res)

        x = self.cgn1(x, c)
        x = F.leaky_relu(x, 0.2)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.conv1(x)

        x = self.cgn2(x, c)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)

        return x + res


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        )

    def forward(self, x):
        return self.model(x) + self.shortcut(x)


class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM, text_dim=TEXT_EMBEDDING_DIM, base_channels=256):
        super().__init__()
        input_dim = noise_dim + text_dim

        self.h_dim = 5
        self.w_dim = 8
        self.initial_channels = base_channels

        hidden_dim = base_channels * 4
        fc_out_features = self.h_dim * self.w_dim * self.initial_channels

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, fc_out_features),
            nn.GroupNorm(min(32, fc_out_features), fc_out_features),
            nn.LeakyReLU(0.2)
        )

        self.res0 = ResBlockUp(base_channels, base_channels)
        self.res1 = ResBlockUp(base_channels, base_channels)
        self.res2 = ResBlockUp(base_channels, base_channels // 2)
        self.attn = SelfAttention(base_channels // 2, use_sn=True)
        self.res3 = ResBlockUp(base_channels // 2, base_channels // 4)
        self.res4 = ResBlockUp(base_channels // 4, base_channels // 8)
        self.res5 = ResBlockUp(base_channels // 8, base_channels // 16)

        self.final_cgn = ConditionalGroupNorm2d(base_channels // 16)
        self.final_conv = CoordConv2d(base_channels // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, noise, text_embedding):
        cond = torch.cat((noise, text_embedding), dim=1)

        x = self.fc(cond)
        x = x.view(-1, self.initial_channels, self.h_dim, self.w_dim)

        x = self.res0(x, cond, target_size=(10, 16))
        x = self.res1(x, cond, target_size=(20, 32))
        x = self.res2(x, cond, target_size=(40, 64))
        x = self.attn(x)
        x = self.res3(x, cond, target_size=(80, 128))
        x = self.res4(x, cond, target_size=(80, 256))
        x = self.res5(x, cond, target_size=(80, 512))

        x = self.final_cgn(x, cond)
        x = F.leaky_relu(x, 0.2)
        x = self.final_conv(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, text_dim=TEXT_EMBEDDING_DIM, base_channels=128):
        super().__init__()

        self.initial_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(1, base_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2)
        )

        self.res1 = ResBlockDown(base_channels, base_channels * 2)
        self.res2 = ResBlockDown(base_channels * 2, base_channels * 4)
        self.attn = SelfAttention(base_channels * 4, use_sn=True)
        self.res3 = ResBlockDown(base_channels * 4, base_channels * 8)
        self.res4 = ResBlockDown(base_channels * 8, base_channels * 16)
        self.res5 = ResBlockDown(base_channels * 16, base_channels * 16)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, N_MELS, TARGET_TIME_STEPS)
            x = self.initial_conv(dummy_input)
            x = self.res1(x)
            x = self.res2(x)
            x = self.attn(x)
            x = self.res3(x)
            x = self.res4(x)
            x = self.res5(x)
            self.flattened_size = x.numel() // x.shape[0]

        self.feature_extractor = nn.Sequential(
            spectral_norm(nn.Linear(self.flattened_size, 512)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.final_linear = spectral_norm(nn.Linear(512, 1))
        self.text_projection = spectral_norm(nn.Linear(text_dim, 512, bias=False))

    def forward(self, spectrogram, text_embedding):
        features = []

        x = self.initial_conv(spectrogram)
        features.append(x)
        x = self.res1(x)
        features.append(x)
        x = self.res2(x)
        features.append(x)
        x = self.attn(x)
        features.append(x)
        x = self.res3(x)
        features.append(x)
        x = self.res4(x)
        features.append(x)
        x = self.res5(x)
        features.append(x)

        x = F.leaky_relu(x, 0.2)
        flat_features = x.view(x.size(0), -1)

        phi = self.feature_extractor(flat_features)
        out = self.final_linear(phi)

        text_proj = self.text_projection(text_embedding)
        projection_score = torch.sum(phi * text_proj, dim=1, keepdim=True)

        out = out + projection_score

        return out, features


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if hasattr(m, 'weight_orig'):
            torch.nn.init.normal_(m.weight_orig.data, 0.0, 0.02)
        elif hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)