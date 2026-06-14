import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxFeatureMap(nn.Module):
    def forward(self, x):
        half = x.size(1) // 2
        return torch.max(x[:, :half], x[:, half:])


class LCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride, padding)
        self.mfm = MaxFeatureMap()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.mfm(x)
        x = self.bn(x)
        return x


class LCNN(nn.Module):
    def __init__(self, input_dim=180, num_classes=2):
        super().__init__()
        self.blocks = nn.Sequential(
            LCNNBlock(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d((2, 2)),
            LCNNBlock(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d((2, 2)),
            LCNNBlock(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d((2, 2)),
            LCNNBlock(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            LCNNBlock(64, 64, kernel_size=(3, 3), padding=(1, 1)),
        )
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.blocks(x)
        x = x.mean(dim=2).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x
