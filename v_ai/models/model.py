# v_ai/models/model.py

import torch
import torch.nn as nn
from v_ai.models.cnn import ResNetBackbone
from v_ai.models.temporal import LSTMTemporalModel


class VideoClassificationModel(nn.Module):
    def __init__(self, cnn_backbone='resnet', lstm_hidden_dim=256, lstm_layers=1,
                 num_classes=2, bidirectional=False, pretrained=True):
        """
        Args:
            cnn_backbone (str): Currently supports 'resnet'. You can extend this to 'efficientnet'.
            lstm_hidden_dim (int): Hidden dimension for the LSTM.
            lstm_layers (int): Number of LSTM layers.
            num_classes (int): Number of classes for classification.
            bidirectional (bool): Use bidirectional LSTM.
            pretrained (bool): Whether to use pretrained weights for the CNN backbone.
        """
        super(VideoClassificationModel, self).__init__()

        if cnn_backbone == 'resnet':
            self.cnn = ResNetBackbone(pretrained=pretrained)
            cnn_feature_dim = self.cnn.output_dim
        else:
            # Placeholder for other backbones (e.g., EfficientNet)
            self.cnn = ResNetBackbone(pretrained=pretrained)
            cnn_feature_dim = self.cnn.output_dim

        self.temporal_model = LSTMTemporalModel(input_dim=cnn_feature_dim,
                                                hidden_dim=lstm_hidden_dim,
                                                num_layers=lstm_layers,
                                                num_classes=num_classes,
                                                bidirectional=bidirectional)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, C, H, W] where T is the sequence length.
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        B, T, C, H, W = x.shape
        cnn_features = []
        # Process each frame with the CNN backbone
        for t in range(T):
            frame = x[:, t, :, :, :]  # Shape: [B, C, H, W]
            feat = self.cnn(frame)     # Shape: [B, cnn_feature_dim]
            cnn_features.append(feat)
        # Stack along the time dimension -> shape: [B, T, cnn_feature_dim]
        features_seq = torch.stack(cnn_features, dim=1)
        logits = self.temporal_model(features_seq)
        return logits
