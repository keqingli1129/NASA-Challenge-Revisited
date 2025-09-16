import torch
import torch.nn as nn
from torchvision import models

class ExoplanetResNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ExoplanetResNet, self).__init__()
        # Load a pretrained ResNet18 model
        self.resnet = models.resnet18(pretrained=pretrained)
        # Change the final fully connected layer to output num_classes
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)