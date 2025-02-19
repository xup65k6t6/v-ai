# v_ai/models/cnn.py

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        # Use the new weights argument.
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
        # Remove the final fully connected layer.
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 512

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 3, H, W]
        Returns:
            features: Tensor of shape [B, 512]
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features
