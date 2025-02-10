# v_ai/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from v_ai.data import VideoDataset
from v_ai.models.model import VideoClassificationModel
from v_ai.transforms import get_train_transforms, get_val_transforms


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (frames, labels) in enumerate(dataloader):
        frames = frames.to(device)  # [B, T, C, H, W]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
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
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)


def test_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # Get predictions
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def main():
    # Configuration parameters
    num_classes = 2             # Adjust for your use-case
    cnn_backbone = 'resnet'       # or 'efficientnet'
    lstm_hidden_dim = 256
    lstm_layers = 1
    bidirectional = False
    pretrained = True
    num_epochs = 10
    batch_size = 4
    learning_rate = 1e-4
    sequence_length = 16
    image_size = 224

    # Choose the type of input: either 'images' or 'video'
    input_type = 'images'  # Change to 'video' if you want to load from video files

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    train_transforms = get_train_transforms(image_size=image_size)
    val_transforms = get_val_transforms(image_size=image_size)

    # Create dummy data.
    # For 'images': each sample is a dict with key 'frames' (list of image paths) and 'label'.
    # For 'video': each sample is a dict with key 'video_path' and 'label'.
    dummy_data_list = []
    total_samples = 100
    if input_type == 'images':
        for i in range(total_samples):
            sample = {
                'frames': [f"/path/to/image_{i}_{j}.jpg" for j in range(sequence_length)],
                'label': i % 2  # example binary label
            }
            dummy_data_list.append(sample)
    elif input_type == 'video':
        for i in range(total_samples):
            sample = {
                'video_path': f"/path/to/video_{i}.mp4",
                'label': i % 2
            }
            dummy_data_list.append(sample)

    # Split data into train, validation, and test sets (e.g., 70/15/15 split)
    total = len(dummy_data_list)
    train_split = int(0.7 * total)
    val_split = int(0.85 * total)
    train_data_list = dummy_data_list[:train_split]
    val_data_list = dummy_data_list[train_split:val_split]
    test_data_list = dummy_data_list[val_split:]

    # Create datasets and dataloaders
    train_dataset = VideoDataset(train_data_list, transform=train_transforms,
                                 sequence_length=sequence_length, input_type=input_type)
    val_dataset = VideoDataset(val_data_list, transform=val_transforms,
                               sequence_length=sequence_length, input_type=input_type)
    test_dataset = VideoDataset(test_data_list, transform=val_transforms,
                                sequence_length=sequence_length, input_type=input_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model
    model = VideoClassificationModel(cnn_backbone=cnn_backbone,
                                     lstm_hidden_dim=lstm_hidden_dim,
                                     lstm_layers=lstm_layers,
                                     num_classes=num_classes,
                                     bidirectional=bidirectional,
                                     pretrained=pretrained)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Evaluate on the test set
    test_loss, test_accuracy = test_epoch(model, test_loader, criterion, device)
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
