# v_ai/models/cnn.py

import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        # Load a pretrained ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        # Remove the final fully connected layer. The output is of shape [B, 512, 1, 1].
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
