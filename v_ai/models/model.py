# v_ai/models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from v_ai.models.cnn import ResNetBackbone
from v_ai.models.temporal import LSTMTemporalModel


class GroupActivityRecognitionModel(nn.Module):
    def __init__(self, num_classes, resnet_size="18", person_hidden_dim=256, group_hidden_dim=256,
                 person_lstm_layers=1, group_lstm_layers=1, bidirectional=False, pretrained=True):
        """
        Args:
            num_classes (int): Number of group activity classes.
            person_hidden_dim (int): Hidden dimension for the person-level LSTM.
            group_hidden_dim (int): Hidden dimension for the group-level LSTM.
            person_lstm_layers (int): Number of layers for the person-level LSTM.
            group_lstm_layers (int): Number of layers for the group-level LSTM.
            bidirectional (bool): Whether to use bidirectional LSTMs.
            pretrained (bool): Whether to use pretrained CNN weights.
        """
        super(GroupActivityRecognitionModel, self).__init__()
        # CNN backbone to extract person features from cropped person images.
        self.person_cnn = ResNetBackbone(resnet_size=resnet_size, pretrained=pretrained)
        cnn_feature_dim = self.person_cnn.output_dim

        # Person-level LSTM: processes the sequence of person–crop features.
        self.person_lstm = LSTMTemporalModel(input_dim=cnn_feature_dim,
                                             hidden_dim=person_hidden_dim,
                                             num_layers=person_lstm_layers,
                                             num_classes=person_hidden_dim,
                                             bidirectional=bidirectional)
        # Group-level LSTM: aggregates person-level features.
        direction = 2 if bidirectional else 1
        self.group_lstm = LSTMTemporalModel(input_dim=person_hidden_dim,
                                            hidden_dim=group_hidden_dim,
                                            num_layers=group_lstm_layers,
                                            num_classes=group_hidden_dim,
                                            bidirectional=bidirectional)
        # Final classifier for group activity.
        self.classifier = nn.Linear(group_hidden_dim, num_classes)

    def forward(self, frames, player_annots):
        """
        Args:
            frames: Tensor of shape [B, T, C, H, W]
            player_annots: List of lists, where each sublist contains dictionaries with "action" and "bbox" for each player.
                           Each sublist has been padded to max_players.
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        B, T, C, H, W = frames.shape
        device = frames.device
        batch_size = B
        max_players = len(player_annots[0])  # Assuming padded to max_players

        # Initialize a list to hold person features for each sample in the batch
        batch_person_features = []

        for b in range(batch_size):
            sample_frames = frames[b]  # [T, C, H, W]
            sample_annots = player_annots[b]
            person_features = []
            x_coords = []
            for annot in sample_annots:
                bbox = annot["bbox"]  # (x, y, w, h)
                if bbox == (0,0,0,0):  # Skip padded annotations
                    continue
                x, y, w, h = bbox
                crops = []
                for t in range(T):
                    frame = sample_frames[t]  # [C, H, W]
                    x_i = int(round(x))
                    y_i = int(round(y))
                    w_i = int(round(w))
                    h_i = int(round(h))
                    x_i = max(0, x_i)
                    y_i = max(0, y_i)
                    if x_i + w_i > W:
                        w_i = W - x_i
                    if y_i + h_i > H:
                        h_i = H - y_i
                    crop = frame[:, y_i:y_i+h_i, x_i:x_i+w_i]  # [C, h_i, w_i]
                    crop = crop.unsqueeze(0)  # [1, C, h_i, w_i]
                    crop = F.interpolate(crop, size=(224, 224), mode='bilinear', align_corners=False)
                    crop = crop.squeeze(0)  # [C, 224, 224]
                    crops.append(crop)
                person_seq = torch.stack(crops, dim=0)  # [T, C, 224, 224]
                # Process each frame crop via the CNN
                features = []
                for t in range(person_seq.size(0)):
                    frame_crop = person_seq[t].unsqueeze(0)  # [1, C, 224, 224]
                    feat = self.person_cnn(frame_crop)  # [1, cnn_feature_dim]
                    features.append(feat.squeeze(0))
                features = torch.stack(features, dim=0)  # [T, cnn_feature_dim]
                features = features.unsqueeze(0)  # [1, T, cnn_feature_dim]
                person_feat = self.person_lstm(features)  # [1, person_hidden_dim]
                person_feat = person_feat.squeeze(0)  # [person_hidden_dim]
                person_features.append(person_feat)
                x_coords.append(x)
            if len(person_features) == 0:
                group_feat = torch.zeros(self.group_lstm.fc.out_features, device=device)
            else:
                # Sort person features by x coordinate
                sorted_indices = sorted(range(len(person_features)), key=lambda i: x_coords[i])
                sorted_feats = [person_features[i] for i in sorted_indices]
                persons_seq = torch.stack(sorted_feats, dim=0)  # [num_persons, person_hidden_dim]
                persons_seq = persons_seq.unsqueeze(0)  # [1, num_persons, person_hidden_dim]
                group_feat = self.group_lstm(persons_seq)  # [1, group_hidden_dim]
                group_feat = group_feat.squeeze(0)  # [group_hidden_dim]
            batch_person_features.append(group_feat)

        # Stack group features for the batch
        group_feats = torch.stack(batch_person_features, dim=0)  # [B, group_hidden_dim]
        logits = self.classifier(group_feats)  # [B, num_classes]
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
