# v_ai/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from v_ai.data import GROUP_ACTIVITY_MAPPING, GroupActivityDataset
from v_ai.models.model import GroupActivityRecognitionModel
from v_ai.transforms import get_val_transforms  # You may define a transform that does NOT resize if needed.
from v_ai.utils.utils import custom_collate, get_device

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        frames = batch["frames"].to(device)      # [T, C, H, W]
        player_annots = batch["player_annots"]     # list of lists (variable length)
        labels = batch["group_label"].to(device)   # [B]
        
        # For simplicity, process one sample at a time if batch size > 1 is problematic
        optimizer.zero_grad()
        # Here, assume batch size = 1 for clarity (or loop over batch)
        logits = model(frames.squeeze(0), player_annots[0])
        loss = criterion(logits.unsqueeze(0), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:
            print(f"Train Batch {i}, Loss: {loss.item():.4f}")
    return running_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device)
            player_annots = batch["player_annots"]
            labels = batch["group_label"].to(device)
            logits = model(frames.squeeze(0), player_annots[0])
            loss = criterion(logits.unsqueeze(0), labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)


def test_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device)
            player_annots = batch["player_annots"]
            labels = batch["group_label"].to(device)
            logits = model(frames.squeeze(0), player_annots[0])
            loss = criterion(logits.unsqueeze(0), labels)
            running_loss += loss.item()
            _, preds = torch.max(logits.unsqueeze(0), 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def main():
    # Hyperparameters.
    num_epochs = 10
    batch_size = 1  # For simplicity, we use batch size 1 due to variable-length player annotations.
    learning_rate = 1e-4

    # Video splits (as provided)
    # train_ids = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
    train_ids = [7]
    # val_ids = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
    val_ids = []
    # test_ids = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

    # Base directory for samples.
    samples_base = os.path.join(os.getcwd(), "data", "videos_sample")
    # You can optionally define a transform (e.g., for normalization) for the full frames.
    transform = None  # Or use get_val_transforms() if you wish to normalize without resizing.

    # Create datasets.
    train_dataset = GroupActivityDataset(samples_base, video_ids=train_ids, transform=transform)
    val_dataset = GroupActivityDataset(samples_base, video_ids=val_ids, transform=transform)
    # test_dataset = GroupActivityDataset(samples_base, video_ids=test_ids, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = get_device()
    num_group_classes = len(GROUP_ACTIVITY_MAPPING)

    model = GroupActivityRecognitionModel(num_classes=num_group_classes,
                                            person_hidden_dim=256,
                                            group_hidden_dim=256,
                                            person_lstm_layers=1,
                                            group_lstm_layers=1,
                                            bidirectional=False,
                                            pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop.
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # test_loss, test_accuracy = test_epoch(model, test_loader, criterion, device)
    # print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy*100:.2f}%")

if __name__ == '__main__':
    main()
