# train_videomae_v2.py

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from v_ai.dataset_videomae import VideoMAE_V2_Dataset, GROUP_ACTIVITY_MAPPING
from v_ai.models.videomae_v2 import VideoMAEV2ClassificationModel
from v_ai.transforms import get_videomae_transforms, VideoMAETransform
from v_ai.utils.earlystopping import EarlyStopping
from v_ai.utils.utils import get_checkpoint_dir, get_device
from torch.amp import autocast, GradScaler

os.environ["WANDB_SILENT"] = "true"

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    scaler = GradScaler() if device.type == 'cuda' else None
    running_loss = 0.0
    total_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        if i % 10 == 0:
            print(f"Batch {i+1}/{total_batches}")
        # Dataset returns {"frames": [C, T, H, W]}; add batch dimension if needed.
        frames = batch["frames"].to(device)
        labels = batch["group_label"].to(device)
        optimizer.zero_grad()
        if device.type == 'cuda':
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(frames)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()  # Scale loss for FP16
            scaler.step(optimizer)
            scaler.update()
        else:
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
            frames = batch["frames"].to(device)
            labels = batch["group_label"].to(device)
            if device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(frames)
                    loss = criterion(logits, labels)
            else:
                logits = model(frames)
                loss = criterion(logits, labels)
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
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
            frames = batch["frames"].to(device)
            labels = batch["group_label"].to(device)
            if device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(frames)
                    loss = criterion(logits, labels)
            else:
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
    parser = argparse.ArgumentParser(description="Train VideoMAE V2 for Volleyball Group Activity Recognition")
    parser.add_argument("--config", type=str, default="config/config_videomae.yaml", help="Path to config file")
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
    
    # Video splits: use your original train_ids, val_ids, and test_ids.
    train_ids = config.get("train_ids", [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54])
    val_ids = config.get("val_ids", [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51])
    test_ids = config.get("test_ids", [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47])
    
    image_size = config.get("image_size", 640)
    num_workers = config.get("num_workers", 4)
    wandb_project = config.get("wandb_project", "volleyball_group_activity")
    wandb_run_name = config.get("wandb_run_name", "videomae_v2_run")
    window_before = config.get("window_before", 3)
    window_after = config.get("window_after", 4)
    
    num_frames = config.get("num_frames", 8)
    num_classes = len(GROUP_ACTIVITY_MAPPING)
    pretrained = config.get("pretrained", True)
    videomae_v2_model_name = config.get("videomae_v2_model_name", "MCG-NJU/videomae-base-finetuned-kinetics")
    
    # Use the new VideoMAE transform that resizes to the expected size and normalizes properly.
    # transform = get_videomae_transforms(model_name= videomae_v2_model_name,image_size=image_size)
    transform = VideoMAETransform(model_name= videomae_v2_model_name, image_size=image_size)
    
    # device = get_device()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine rank (only rank 0 should log to WandB)
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    
    train_dataset = VideoMAE_V2_Dataset(
        samples_base = samples_base,
        video_ids = train_ids,
        window_before = window_before,
        window_after = window_after,
        transform = transform,
        num_frames = num_frames
    )
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    val_dataset = VideoMAE_V2_Dataset(
        samples_base = samples_base,
        video_ids = val_ids,
        window_before = window_before,
        window_after = window_after,
        transform = transform,
        num_frames = num_frames        
    )
    val_sampler = DistributedSampler(val_dataset) if dist.is_initialized() else None
    # For testing, if your test_ids refer to video files, the same dataset class handles it.
    test_dataset = VideoMAE_V2_Dataset(
        samples_base = samples_base,
        video_ids = test_ids,
        window_before = window_before,
        window_after = window_after,
        transform = transform,
        num_frames = num_frames
    )
    test_sampler = DistributedSampler(test_dataset) if dist.is_initialized() else None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=num_workers, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle= False , num_workers=num_workers, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=test_sampler)
    
    # Initialize VideoMAE V2 model.
    model = VideoMAEV2ClassificationModel(num_frames=num_frames, image_size = image_size,
                                           pretrained=pretrained, model_name=videomae_v2_model_name)
    model = model.to(device)
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(checkpoint_dir, "videomae_v2_best.pt"))
    
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    # Initialize WandB only on rank 0, disable logging on other ranks.
    if rank == 0:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
        wandb.watch(model, log="all")
    # else:
    #     wandb.init(project=wandb_project, name=wandb_run_name, config=config, mode="disabled")
    
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
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
        
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(checkpoint_dir, "checkpoint_last.pt"))
        
            early_stopping(val_loss, model, device)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        
    model = early_stopping.load_best_model(model)
    # Optionally, run test_epoch if desired.
    # test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    # print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc*100:.2f}%")
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()
