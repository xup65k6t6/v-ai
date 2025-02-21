# v_ai/models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from v_ai.models.cnn import ResNetBackbone
from v_ai.models.temporal import LSTMTemporalModel


class GroupActivityRecognitionModel(nn.Module):
    def __init__(self, num_classes, resnet_size="18", person_hidden_dim=256, group_hidden_dim=256,
                 person_lstm_layers=1, group_lstm_layers=1, bidirectional=True, pretrained=True,
                 use_scene_context=False):
        super(GroupActivityRecognitionModel, self).__init__()
        self.person_cnn = ResNetBackbone(resnet_size=resnet_size, pretrained=pretrained)
        cnn_feature_dim = self.person_cnn.output_dim

        self.person_lstm = LSTMTemporalModel(
            input_dim=cnn_feature_dim, hidden_dim=person_hidden_dim,
            num_layers=person_lstm_layers, num_classes=cnn_feature_dim,
            bidirectional=bidirectional
        )

        self.use_scene_context = use_scene_context
        if use_scene_context:
            self.scene_cnn = ResNetBackbone(resnet_size=resnet_size, pretrained=pretrained)
            self.scene_lstm = LSTMTemporalModel(
                input_dim=cnn_feature_dim, hidden_dim=group_hidden_dim,
                num_layers=group_lstm_layers, num_classes=group_hidden_dim,
                bidirectional=bidirectional
            )
            group_input_dim = cnn_feature_dim * 2  # Concat CNN and LSTM features
        else:
            group_input_dim = cnn_feature_dim * 2

        self.group_lstm = LSTMTemporalModel(
            input_dim=group_input_dim, hidden_dim=group_hidden_dim,
            num_layers=group_lstm_layers, num_classes=group_hidden_dim,
            bidirectional=bidirectional
        )

        self.classifier = nn.Linear(group_hidden_dim, num_classes)

    def forward(self, frames, player_annots):
        B, T, C, H, W = frames.shape
        device = frames.device

        # Step 1: Process person-level features over the temporal window
        batch_frame_features = []
        for b in range(B):
            sample_frames = frames[b]  # [T, C, H, W]
            sample_annots = player_annots[b]  # List of dicts
            person_seqs = []  # List of [T, cnn_feature_dim] for each person
            for annot in sample_annots:
                bbox = annot["bbox"]
                if bbox == (0, 0, 0, 0):
                    continue
                x, y, w, h = map(int, bbox)
                x = max(0, x)
                y = max(0, y)
                w = min(w, W - x)
                h = min(h, H - y)
                if w <= 0 or h <= 0:
                    continue
                # Extract crops for all frames
                crops = []
                for t in range(T):
                    frame = sample_frames[t]  # [C, H, W]
                    crop = frame[:, y:y+h, x:x+w]  # [C, h, w]
                    crop = F.interpolate(crop.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
                    crop = (crop / 255.0 - mean) / std
                    crops.append(crop)
                crops = torch.stack(crops)  # [T, C, 224, 224]
                cnn_feats = self.person_cnn(crops)  # [T, cnn_feature_dim]
                person_seqs.append(cnn_feats)

            # Apply person-level LSTM to each person's sequence
            frame_features = []
            for t in range(T):
                cnn_feats_t = [seq[t] for seq in person_seqs]  # CNN features at frame t
                if not cnn_feats_t:
                    frame_feat = torch.zeros(2 * self.person_cnn.output_dim, device=device)
                else:
                    cnn_feats_t = torch.stack(cnn_feats_t)  # [num_persons, cnn_feature_dim]
                    cnn_pooled = cnn_feats_t.max(dim=0)[0]  # [cnn_feature_dim]
                    # LSTM over all persons' sequences
                    persons_seq = torch.stack(person_seqs, dim=0)  # [num_persons, T, cnn_feature_dim]
                    lstm_feats = self.person_lstm(persons_seq.transpose(0, 1))  # [T, num_persons, cnn_feature_dim] -> [T, cnn_feature_dim]
                    frame_feat = torch.cat([cnn_pooled, lstm_feats[t]], dim=0)  # [2 * cnn_feature_dim]
                frame_features.append(frame_feat)
            batch_frame_features.append(torch.stack(frame_features))  # [T, 2 * cnn_feature_dim]

        # Step 2: Optionally process scene context
        if self.use_scene_context:
            scene_features = []
            for t in range(T):
                frame = frames[:, t]  # [B, C, H, W]
                scene_feat = self.scene_cnn(frame)  # [B, cnn_feature_dim]
                scene_features.append(scene_feat)
            scene_seq = torch.stack(scene_features, dim=1)  # [B, T, cnn_feature_dim]
            scene_output = self.scene_lstm(scene_seq)  # [B, group_hidden_dim]
            person_seq = torch.stack(batch_frame_features)  # [B, T, 2 * cnn_feature_dim]
            group_input = torch.cat([person_seq, scene_output.unsqueeze(1).expand(-1, T, -1)], dim=2)
        else:
            group_input = torch.stack(batch_frame_features)  # [B, T, 2 * cnn_feature_dim]

        # Step 3: Group-level LSTM
        group_output = self.group_lstm(group_input)  # [B, group_hidden_dim]

        # Step 4: Classification
        logits = self.classifier(group_output)  # [B, num_classes]
        return logits

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
