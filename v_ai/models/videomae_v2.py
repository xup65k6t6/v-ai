# v_ai/models/videomae_v2.py

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class VideoMAEV2ClassificationModel(nn.Module):
    def __init__(self, num_classes, num_frames=8, pretrained=True, model_name="OpenGVLab/VideoMAEv2-Base"):
        """
        Args:
            num_classes (int): Number of group activity classes.
            num_frames (int): Number of frames per clip.
            pretrained (bool): Whether to load pretrained weights.
            model_name (str): Hugging Face model identifier.
        """
        super(VideoMAEV2ClassificationModel, self).__init__()
        self.num_frames = num_frames
        # Load configuration and adjust for our input.
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_frames = num_frames  # Adjust temporal input; the model will interpolate positional embeddings.
        config.num_classes = 0  # Remove the pretrained head.

        if pretrained:
            self.videomae = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        else:
            self.videomae = AutoModel.from_config(config, trust_remote_code=True)
        
        hidden_size = config.hidden_size if hasattr(config, "hidden_size") else 768
        # Attach a new classification head.
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, C, T, H, W]
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        outputs = self.videomae(x)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output  # [B, hidden_size]
        else:
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits
