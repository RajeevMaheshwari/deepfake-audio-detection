import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class ResNetSpec(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if ResNet34_Weights else None
        self.resnet = resnet34(weights=weights)

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
