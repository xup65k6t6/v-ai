# v_ai/train.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
import yaml
from v_ai.data import GROUP_ACTIVITY_MAPPING, GroupActivityDataset, SimplifiedGroupActivityDataset
from v_ai.models.model import GroupActivityRecognitionModel, VideoClassificationModel, Video3DClassificationModel
from v_ai.transforms import (
    get_3dcnn_transform,
    get_val_transforms,
    resize_only,
)  # You may define a transform that does NOT resize if needed.
from v_ai.utils.earlystopping import EarlyStopping
from v_ai.utils.utils import custom_collate, get_checkpoint_dir, get_device
from torch.profiler import profile, record_function, ProfilerActivity

os.environ["WANDB_SILENT"] = "true"


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        frames = batch["frames"].to(device).permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        labels = batch["group_label"].to(device)
        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device).permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
            labels = batch["group_label"].to(device)
            logits = model(frames)
            loss = criterion(logits, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def test_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device).permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
            labels = batch["group_label"].to(device)
            logits = model(frames)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def main():
    # Parse command-line argument for config file.
    parser = argparse.ArgumentParser(description="Train Group Activity Recognition Model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the YAML config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Extract parameters from config.
    # Hyperparameters.
    num_epochs = config.get("num_epochs", 10)
    batch_size = config.get("batch_size", 1)  # For simplicity, we use batch size 1 due to variable-length player annotations.
    learning_rate = config.get("learning_rate", 1e-4)
    patience = config.get("patience", 5)
    # dir setup
    samples_base = config.get("samples_base", os.path.join(os.getcwd(), "data", "videos"))
    checkpoint_dir = config.get("checkpoint_dir", get_checkpoint_dir())
    # Video splits 
    train_ids = config.get("train_ids", [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54])
    val_ids = config.get("val_ids", [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51])
    test_ids = config.get("test_ids", [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47])
    num_workers = config.get("num_workers", 4)
    wandb_project = config.get("wandb_project", "volleyball_group_activity")    
    resnet_size = config.get("resnet_size", "18")
    window_before = config.get("window_before", 5)
    window_after = config.get("window_after", 4)

    # Model-related parameters.
    hidden_dim = config.get("hidden_dim", 256)
    lstm_layers = config.get("lstm_layers", 1)
    bidirectional = config.get("bidirectional", False)
    pretrained = config.get("pretrained", True)

    device = get_device()

    transform = resize_only(image_size=320)  # Required for pretrained ResNet3D
    # transform = get_3dcnn_transform(image_size=112)  # Required for pretrained ResNet3D

    # Create datasets using SimplifiedGroupActivityDataset
    train_dataset = SimplifiedGroupActivityDataset(
        samples_base, video_ids=train_ids, transform=transform, window_before=window_before, window_after=window_after
    )
    val_dataset = SimplifiedGroupActivityDataset(
        samples_base, video_ids=val_ids, transform=transform, window_before=window_before, window_after=window_after
    )
    test_dataset = SimplifiedGroupActivityDataset(
        samples_base, video_ids=test_ids, transform=transform, window_before=window_before, window_after=window_after
    )

    # Standard DataLoader without custom_collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Model, criterion, optimizer, and scheduler
    model = Video3DClassificationModel(
        num_classes=len(GROUP_ACTIVITY_MAPPING),
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=os.path.join(checkpoint_dir, "best.pt")
    )

    # Initialize wandb.
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project=wandb_project,
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "bidirectional": bidirectional,
            "window_before": window_before,
            "window_after": window_after,
        },
    )
    wandb.watch(model, log="all")  # Optional: track gradients and model parameters.

    # Training loop.
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        # Log metrics to wandb.
        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        # Save last checkpoint every epoch.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
            },
            os.path.join(checkpoint_dir, "checkpoint_last.pt"),
        )

        # Call early stopping.
        early_stopping(val_loss, model, device)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    model = early_stopping.load_best_model(model)
    # test_loss, test_accuracy = test_epoch(model, test_loader, criterion, device)
    # wandb.log({
    #     "test_loss": test_loss,
    #     "test_accuracy": test_accuracy
    # })
    # print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy*100:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    main()
