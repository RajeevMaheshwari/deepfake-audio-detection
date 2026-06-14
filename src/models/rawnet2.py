import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SincConv(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30 + 30 * np.random.rand(out_channels)
        band_hz = 50 + 30 * np.random.rand(out_channels)

        self.low_hz_ = nn.Parameter(torch.tensor(low_hz, dtype=torch.float32).unsqueeze(1))
        self.band_hz_ = nn.Parameter(torch.tensor(band_hz, dtype=torch.float32).unsqueeze(1))

        n = (torch.arange(kernel_size) - (kernel_size - 1) / 2).unsqueeze(0)
        self.register_buffer('n_', n.float())

        hamming = 0.54 - 0.46 * torch.cos(2 * np.pi * torch.arange(kernel_size) / kernel_size)
        self.register_buffer('window_', hamming.float().unsqueeze(0))

    def forward(self, x):
        low = torch.abs(self.low_hz_)
        high = low + torch.abs(self.band_hz_)
        f_times_t = torch.matmul(low, self.n_) * 2 / self.sample_rate
        band = (torch.matmul(high, self.n_) * 2 / self.sample_rate)

        filt = (torch.sin(band) - torch.sin(f_times_t)) / (self.n_ / 2 + 1e-8)
        filt = filt * self.n_ / (self.n_ + 1e-8)
        filt = filt * self.window_

        filt = filt / (torch.abs(filt).sum(dim=1, keepdim=True) + 1e-8)
        filt = filt.unsqueeze(1)

        return F.conv1d(x, filt, padding=self.kernel_size // 2)


class FMSBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.scale(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.fms = FMSBlock(channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.3)
        x = self.bn2(self.conv2(x))
        x = self.fms(x)
        x = x * self.se(x)
        return F.leaky_relu(x + residual, 0.3)


class RawNet2(nn.Module):
    def __init__(self, num_classes=2, sinc_filters=128, sinc_kernel=251):
        super().__init__()
        self.sinc = SincConv(sinc_filters, sinc_kernel)

        self.conv1 = nn.Conv1d(sinc_filters, sinc_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(sinc_filters)

        self.res_blocks = nn.ModuleList([
            ResBlock(sinc_filters),
            ResBlock(sinc_filters),
            ResBlock(sinc_filters),
            ResBlock(sinc_filters),
            ResBlock(sinc_filters // 2),
        ])

        self.conv_reduce = nn.Conv1d(sinc_filters, sinc_filters // 2, 3, padding=1)
        self.bn_reduce = nn.BatchNorm1d(sinc_filters // 2)

        self.res_block6 = ResBlock(sinc_filters // 2)

        self.gru = nn.GRU(sinc_filters // 2, 512, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.sinc(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.3)
        x = F.max_pool1d(x, 3)

        for i, block in enumerate(self.res_blocks):
            x = block(x)
            if i == 3:
                x = F.leaky_relu(self.bn_reduce(self.conv_reduce(x)), 0.3)

        x = self.res_block6(x)
        x = F.avg_pool1d(x, 8)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
