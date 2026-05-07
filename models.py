import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math

N_MELS = 80
TARGET_TIME_STEPS = 512
TEXT_EMBEDDING_DIM = 512
NOISE_DIM = 100


class PixelNorm(nn.Module):
    """
    Applies pixel normalization to prevent phase distortion and magnitude spikes.
    """

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        """
        Upcasts the input to float32 to prevent float16 overflow during calculation.
        """
        x_f32 = x.float()
        norm = torch.sqrt(torch.mean(x_f32 ** 2, dim=1, keepdim=True) + self.epsilon)
        return (x_f32 / norm).to(x.dtype)


class MinibatchStdDev(nn.Module):
    """
    Computes minibatch standard deviation to prevent mode collapse.
    """

    def __init__(self, group_size=4, epsilon=1e-8):
        super().__init__()
        self.group_size = group_size
        self.epsilon = epsilon

    def forward(self, x):
        """
        Calculates the standard deviation across the batch and appends it as a new channel.
        """
        batch_size, channels, height, width = x.shape
        group_size = min(self.group_size, batch_size)

        if batch_size % group_size != 0:
            group_size = batch_size

        y = x.float().view(group_size, -1, channels, height, width)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.mean(y ** 2, dim=0)
        y = torch.sqrt(y + self.epsilon)
        y = torch.mean(y, dim=[1, 2, 3], keepdim=True)
        y = y.repeat(group_size, 1, height, width).to(x.dtype)

        return torch.cat([x, y], dim=1)


class CoordConv2d(nn.Module):
    """
    2D Convolution that appends coordinate channels to the input tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding, padding_mode='reflect')
        )

    def forward(self, x):
        """
        Creates X and Y coordinate grids and concatenates them with the input feature map.
        """
        batch_size, _, height, width = x.shape
        y_coords = torch.linspace(-1, 1, steps=height, device=x.device).view(1, 1, height, 1).expand(batch_size, 1,
                                                                                                     height, width)
        x_coords = torch.linspace(-1, 1, steps=width, device=x.device).view(1, 1, 1, width).expand(batch_size, 1,
                                                                                                   height, width)
        x = torch.cat([x, y_coords, x_coords], dim=1)
        return self.conv(x)


class ConditionalGroupNorm2d(nn.Module):
    """
    Applies group normalization modulated by a conditional embedding.
    """

    def __init__(self, num_features, num_groups=32, cond_dim=TEXT_EMBEDDING_DIM + NOISE_DIM):
        super().__init__()
        self.num_features = num_features
        self.num_groups = min(num_groups, num_features)
        self.norm = nn.GroupNorm(self.num_groups, num_features, affine=False)

        self.gamma_embed = nn.Linear(cond_dim, num_features)
        self.beta_embed = nn.Linear(cond_dim, num_features)

        nn.init.ones_(self.gamma_embed.weight)
        nn.init.zeros_(self.beta_embed.weight)

    def forward(self, x, cond):
        """
        Normalizes the input and scales/shifts it based on the conditional text and noise.
        """
        out = self.norm(x)
        gamma = self.gamma_embed(cond).view(-1, self.num_features, 1, 1)
        beta = self.beta_embed(cond).view(-1, self.num_features, 1, 1)
        return out * gamma + beta


class SelfAttention(nn.Module):
    """
    Applies self-attention to spatial dimensions of the feature map.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Calculates attention weights using query and key projections, then applies them to the value projection.
        """
        batch_size, channels, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)
        return self.gamma * out + x


class ResBlockUp(nn.Module):
    """
    Generator residual block with upsampling and conditional normalization.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cgn1 = ConditionalGroupNorm2d(in_channels)
        self.conv1 = CoordConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cgn2 = ConditionalGroupNorm2d(out_channels)
        self.conv2 = CoordConv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.upsample_conv = CoordConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pixel_norm = PixelNorm()

    def forward(self, x, cond):
        """
        Processes the input through residual connections, upsampling both the shortcut and main path.
        """
        res = F.interpolate(x, scale_factor=2, mode='nearest')
        res = self.upsample_conv(res)
        res = self.pixel_norm(res)

        h = self.cgn1(x, cond)
        h = F.leaky_relu(h, 0.2)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = self.pixel_norm(h)

        h = self.cgn2(h, cond)
        h = F.leaky_relu(h, 0.2)
        h = self.conv2(h)
        h = self.pixel_norm(h)

        return h + res


class Generator(nn.Module):
    """
    Main generator architecture mapping text and noise into an audio spectrogram.
    """

    def __init__(self, noise_dim=NOISE_DIM, text_dim=TEXT_EMBEDDING_DIM, base_channels=512):
        super().__init__()
        self.cond_dim = noise_dim + text_dim
        self.base_channels = base_channels
        self.dense = spectral_norm(nn.Linear(self.cond_dim, base_channels * 4 * 4))

        self.res1 = ResBlockUp(base_channels, base_channels)
        self.res2 = ResBlockUp(base_channels, base_channels // 2)
        self.res3 = ResBlockUp(base_channels // 2, base_channels // 4)
        self.attn = SelfAttention(base_channels // 4)
        self.res4 = ResBlockUp(base_channels // 4, base_channels // 8)
        self.res5 = ResBlockUp(base_channels // 8, base_channels // 16)
        self.res6 = ResBlockUp(base_channels // 16, base_channels // 32)
        self.res7 = ResBlockUp(base_channels // 32, base_channels // 64)

        self.final_conv = nn.Sequential(
            PixelNorm(),
            CoordConv2d(base_channels // 64, 1, kernel_size=(433, 1), stride=(1, 1), padding=0)
        )

    def forward(self, z, text_embedding):
        """
        Concatenates noise and text, projects to spatial dimensions, and upsamples to generate the spectrogram.
        """
        cond = torch.cat([z, text_embedding], dim=1)
        x = self.dense(cond)
        x = x.view(-1, self.base_channels, 4, 4)

        x = self.res1(x, cond)
        x = self.res2(x, cond)
        x = self.res3(x, cond)
        x = self.attn(x)
        x = self.res4(x, cond)
        x = self.res5(x, cond)
        x = self.res6(x, cond)
        x = self.res7(x, cond)

        x = self.final_conv(x)
        x = torch.tanh(x)
        return x


class ResBlockDown(nn.Module):
    """
    Critic residual block with spatial downsampling.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = CoordConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = CoordConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        """
        Processes the input through residual connections, downsampling both the shortcut and main path.
        """
        res = F.avg_pool2d(x, 2)
        res = self.shortcut(res)

        h = F.leaky_relu(x, 0.2)
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)
        return h + res


class Critic(nn.Module):
    """
    Main critic architecture evaluating the realism of a spectrogram given a text condition.
    """

    def __init__(self):
        super().__init__()
        self.initial_conv = CoordConv2d(1, 32, kernel_size=3, stride=1, padding=1)

        self.res1 = ResBlockDown(32, 64)
        self.res2 = ResBlockDown(64, 128)
        self.attn = SelfAttention(128)
        self.res3 = ResBlockDown(128, 256)
        self.res4 = ResBlockDown(256, 512)
        self.res5 = ResBlockDown(512, 512)

        self.mbstd = MinibatchStdDev()
        self.final_conv = CoordConv2d(513, 1, kernel_size=(2, 16), stride=(1, 16), padding=0)
        self.text_projection = spectral_norm(nn.Linear(TEXT_EMBEDDING_DIM, 512))

    def forward(self, spectrogram, text_embedding):
        """
        Processes the spectrogram, calculates a spatial score, projects the text embedding, and sums them for a final realism score.
        """
        x = self.initial_conv(spectrogram)
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = F.leaky_relu(x, 0.2)
        x = self.mbstd(x)
        out = self.final_conv(x)

        text_proj = self.text_projection(text_embedding)
        text_proj = text_proj.view(text_proj.size(0), text_proj.size(1), 1, 1)
        projection_score = torch.sum(x[:, :512, :, :] * text_proj, dim=1, keepdim=True)

        spatial_score = out + projection_score
        final_score = torch.mean(spatial_score, dim=(2, 3))

        return final_score