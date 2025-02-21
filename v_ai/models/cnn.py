# v_ai/models/cnn.py

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

class ResNetBackbone(nn.Module):
    def __init__(self, resnet_size="18", pretrained=True):
        super(ResNetBackbone, self).__init__()
        # Mapping from size to model and weights
        models = {
            "18": (resnet18, ResNet18_Weights),
            "34": (resnet34, ResNet34_Weights),
            "50": (resnet50, ResNet50_Weights),
            "101": (resnet101, ResNet101_Weights),
            "152": (resnet152, ResNet152_Weights),
        }
        if resnet_size not in models:
            raise ValueError(f"Unsupported ResNet size: {resnet_size}. Choose from {list(models.keys())}")
        
        model_fn, weights_cls = models[resnet_size]
        weights = weights_cls.DEFAULT if pretrained else None
        resnet = model_fn(weights=weights)
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Determine the output dimension based on ResNet size
        if resnet_size in ["18", "34"]:
            self.output_dim = 512
        else:
            self.output_dim = 2048
        print(f"ResNet{resnet_size} output dim: {self.output_dim}")

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 3, H, W]
        Returns:
            features: Tensor of shape [B, output_dim]
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features