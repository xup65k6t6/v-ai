# v_ai/models/videomae_v2.py

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, VideoMAEForVideoClassification, VideoMAEConfig
from v_ai.data import GROUP_ACTIVITY_MAPPING

class VideoMAEV2ClassificationModel(nn.Module):
    def __init__(self, num_frames=8, image_size= 224,pretrained=True, model_name="OpenGVLab/VideoMAEv2-Base"):
        """
        Args:
            num_frames (int): Number of frames per clip.
            pretrained (bool): Whether to load pretrained weights.
            model_name (str): Hugging Face model identifier.
        """
        super(VideoMAEV2ClassificationModel, self).__init__()
        self.num_frames = num_frames
        # Load configuration and adjust for our input.
        # config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config = VideoMAEConfig.from_pretrained(model_name, trust_remote_code=True, image_size = image_size, num_frames = 8, )
        config.num_frames = num_frames  # Adjust temporal input; the model will interpolate positional embeddings.
        config.num_labels = len(GROUP_ACTIVITY_MAPPING)
        config.label2id = GROUP_ACTIVITY_MAPPING
        config.id2label = {v: k for k, v in GROUP_ACTIVITY_MAPPING.items()}

        if pretrained:
            # self.videomae = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
            self.videomae = VideoMAEForVideoClassification.from_pretrained(model_name, config=config, trust_remote_code=True, ignore_mismatched_sizes=True)
        else:
            self.videomae = AutoModel.from_config(config, trust_remote_code=True)


    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, C, T, H, W]
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        outputs = self.videomae(x)
        logits = outputs.logits
        return logits
