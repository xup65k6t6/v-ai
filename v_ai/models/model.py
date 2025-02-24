# v_ai/models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from v_ai.models.cnn import ResNetBackbone
from v_ai.models.temporal import LSTMTemporalModel
from torchvision.models.video import r3d_18, R3D_18_Weights

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

    def forward(self, frames, person_crops):
        """
        Args:
            frames: Tensor [B, T, C, H, W]
            person_crops: List of length B, each containing list of [T, C, 224, 224] tensors
        Returns:
            logits: Tensor [B, num_classes]
        """
        B, T, C, H, W = frames.shape
        device = frames.device

        # Step 1: Process person-level features
        batch_frame_features = []
        for b in range(B):
            sample_crops = person_crops[b]  # List of [T, C, 224, 224]
            person_seqs = []
            for crops in sample_crops:
                if crops.sum() == 0:  # Skip padded crops
                    continue
                cnn_feats = self.person_cnn(crops)  # [T, cnn_feature_dim]
                person_seqs.append(cnn_feats)

            frame_features = []
            for t in range(T):
                cnn_feats_t = [seq[t] for seq in person_seqs if seq.sum() != 0]
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
    def __init__(self, num_classes, resnet_size="18", hidden_dim=256, lstm_layers=1,
                 bidirectional=False, pretrained=True):
        super(VideoClassificationModel, self).__init__()
        self.cnn = ResNetBackbone(resnet_size=resnet_size, pretrained=pretrained)
        cnn_feature_dim = self.cnn.output_dim
        
        self.lstm = LSTMTemporalModel(
            input_dim=cnn_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            num_classes=hidden_dim,
            bidirectional=bidirectional
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, frames):
        """
        Args:
            frames: Tensor [B, T, C, H, W]
        Returns:
            logits: Tensor [B, num_classes]
        """
        B, T, C, H, W = frames.shape
        # Process each frame with CNN
        cnn_features = []
        for t in range(T):
            frame = frames[:, t]  # [B, C, H, W]
            feat = self.cnn(frame)  # [B, cnn_feature_dim]
            cnn_features.append(feat)
        features_seq = torch.stack(cnn_features, dim=1)  # [B, T, cnn_feature_dim]
        
        # Temporal modeling with LSTM
        lstm_output = self.lstm(features_seq)  # [B, hidden_dim]
        
        # Classification
        logits = self.classifier(lstm_output)  # [B, num_classes]
        return logits


class Video3DClassificationModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Video3DClassificationModel, self).__init__()
        # Load pretrained ResNet3D-18
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r3d_18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)  # Modify final layer

    def forward(self, x):
        """
        Args:
            x: Tensor [B, T, C, H, W]
        Returns:
            logits: Tensor [B, num_classes]
        """
        # Input x: [B, T, C, H, W] from DataLoader
        x = x.permute(0, 2, 1, 3, 4)  # Convert to [B, C, T, H, W]
        return self.backbone(x)        # Output: [B, num_classes]