# v_ai/models/temporal.py

import torch
import torch.nn as nn


class LSTMTemporalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=False):
        super(LSTMTemporalModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=bidirectional)
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, input_dim]
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        output, (hn, _) = self.lstm(x)
        # Use the last hidden state (if bidirectional, concatenate both directions)
        if self.lstm.bidirectional:
            final_feature = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            final_feature = hn[-1]
        logits = self.fc(final_feature)
        return logits
