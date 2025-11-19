import torch
import torch.nn as nn
from torchvision import models

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        # Load standard ResNet structure
        # explicitly setting weights=None ensures we train from scratch.
        self.net = models.resnet18(weights=None)

        # Replace the first 7x7 conv with 3x3, stride 1
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove MaxPool
        self.net.maxpool = nn.Identity()

        # Adjust the final Fully Connected layer for 200 classes
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)
