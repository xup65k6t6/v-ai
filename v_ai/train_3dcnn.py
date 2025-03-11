# v_ai/train_3dcnn.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
from v_ai.data import GROUP_ACTIVITY_MAPPING, SimplifiedGroupActivityDataset
from v_ai.models.model import Video3DClassificationModel
from v_ai.transforms import resize_only
from v_ai.utils.earlystopping import EarlyStopping
from v_ai.utils.utils import get_checkpoint_dir, get_device

os.environ["WANDB_SILENT"] = "true"


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        if i % 10 == 0:  # Log every 10 batches for progress tracking
            print(f"Batch {i+1}/{total_batches}")
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
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device).permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
            labels = batch["group_label"].to(device)
            logits = model(frames)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_loss = running_loss / len(dataloader)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return avg_loss, metrics


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
    parser = argparse.ArgumentParser(description="Train 3D CNN for Group Activity Recognition")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract parameters from config.
    num_epochs = config.get("num_epochs", 10)
    batch_size = config.get("batch_size", 1)
    learning_rate = config.get("learning_rate", 1e-4)
    patience = config.get("patience", 5)
    samples_base = config.get("samples_base", os.path.join(os.getcwd(), "data", "videos"))
    checkpoint_dir = config.get("checkpoint_dir", get_checkpoint_dir())
    train_ids = config.get("train_ids", [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54])
    val_ids = config.get("val_ids", [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51])
    test_ids = config.get("test_ids", [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47])
    image_size = config.get("image_size", 640)
    num_workers = config.get("num_workers", 4)
    wandb_project = config.get("wandb_project", "volleyball_group_activity")
    wandb_run_name = config.get("wandb_run_name", "3D_CNN")
    window_before = config.get("window_before", 3)
    window_after = config.get("window_after", 4)
    pretrained = config.get("pretrained", True)

    # Distributed setup
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = get_device()
        local_rank = 0

    # Determine rank (only rank 0 logs to WandB)
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Define transform
    transform = resize_only(image_size=image_size)

    # Create datasets
    train_dataset = SimplifiedGroupActivityDataset(
        samples_base, video_ids=train_ids, transform=transform, window_before=window_before, window_after=window_after
    )
    val_dataset = SimplifiedGroupActivityDataset(
        samples_base, video_ids=val_ids, transform=transform, window_before=window_before, window_after=window_after
    )
    test_dataset = SimplifiedGroupActivityDataset(
        samples_base, video_ids=test_ids, transform=transform, window_before=window_before, window_after=window_after
    )

    # Distributed samplers
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset) if dist.is_initialized() else None
    test_sampler = DistributedSampler(test_dataset) if dist.is_initialized() else None

    # DataLoaders with samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=test_sampler,
    )

    # Model, criterion, optimizer, and scheduler
    model = Video3DClassificationModel(
        num_classes=len(GROUP_ACTIVITY_MAPPING),
        pretrained=pretrained,
    ).to(device)

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=os.path.join(checkpoint_dir, "best.pt")
    )

    # Initialize wandb only on rank 0
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    if rank == 0:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config,
        )
        wandb.watch(model, log="all")

    # Training loop
    for epoch in range(num_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)  # Ensure shuffling is different each epoch
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        # Log metrics only on rank 0
        if rank == 0:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "lr": optimizer.param_groups[0]["lr"],
            })
            print(
                f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                f"Acc = {val_metrics['accuracy']:.4f}, Prec = {val_metrics['precision']:.4f}, "
                f"Recall = {val_metrics['recall']:.4f}, F1 = {val_metrics['f1']:.4f}"
            )

            # Save last checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(checkpoint_dir, "checkpoint_last.pt"))

            # Early stopping
            early_stopping(val_loss, model, device)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    # Load best model
    if rank == 0:
        model = early_stopping.load_best_model(model)

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()