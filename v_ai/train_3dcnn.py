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
from v_ai.transforms import get_3dcnn_train_transforms, get_3dcnn_val_transforms
from v_ai.utils.earlystopping import EarlyStopping
from v_ai.utils.utils import get_device

os.environ["WANDB_SILENT"] = "true"

class Trainer:
    def __init__(self, config, model, train_loader, val_loader, test_loader, device, resume=False):
        self.config = config
        self.model = model.to(device)
        if dist.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[device.index])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=2)
        self.early_stopping = EarlyStopping(patience=config['patience'], verbose=True, path=os.path.join(config['checkpoint_dir'], "best_3dcnn.pt"))
        self.checkpoint_path = os.path.join(config['checkpoint_dir'], "checkpoint_last.pt")
        self.resume = resume
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.rank == 0:
            wandb.login(key=os.environ.get("WANDB_API_KEY"))
            wandb.init(project=config['wandb_project'], name=config.get('wandb_run_name', '3D_CNN'), config=config)
            wandb.watch(self.model, log="all")

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        total_batches = len(self.train_loader)
        all_labels = []
        all_preds = []
        for i, batch in enumerate(self.train_loader):
            if i % 10 == 0 and self.rank == 0:
                print(f"Batch {i+1}/{total_batches}")
            frames = batch["frames"].to(self.device).permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
            labels = batch["group_label"].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(frames)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        avg_loss = running_loss / len(self.train_loader)
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }        
        return avg_loss, metrics

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in self.val_loader:
                frames = batch["frames"].to(self.device).permute(0, 2, 1, 3, 4)
                labels = batch["group_label"].to(self.device)
                logits = self.model(frames)
                loss = self.criterion(logits, labels)
                running_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        avg_loss = running_loss / len(self.val_loader)
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        return avg_loss, metrics

    def train(self):
        # Check for existing checkpoint to resume training
        start_epoch = 0
        if self.resume and os.path.exists(self.checkpoint_path):
            # TODO: load checkpoint_path from separate variable in config or argument. It is possible the last checkpoint is not accessible from config[checkpoint_dir]
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.early_stopping.best_score = checkpoint.get('early_stopping_best_score', None)
            self.early_stopping.counter = checkpoint.get('early_stopping_counter', 0)
            self.early_stopping.val_loss_min = checkpoint.get('early_stopping_val_loss_min', float('inf'))
            if self.rank == 0:
                print(f"Resuming training from epoch {start_epoch + 1}")
        else:
            if self.rank == 0:
                print("Starting training from scratch")

        for epoch in range(start_epoch, self.config['num_epochs']):
            if hasattr(self.train_loader, 'sampler') and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            if self.rank == 0:
                print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            if self.rank == 0:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_metrics["accuracy"],
                    "train_precision": train_metrics["precision"],
                    "train_recall": train_metrics["recall"],
                    "train_f1": train_metrics["f1"],
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
                print(
                    f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                    f"Acc = {val_metrics['accuracy']:.4f}, Prec = {val_metrics['precision']:.4f}, "
                    f"Recall = {val_metrics['recall']:.4f}, F1 = {val_metrics['f1']:.4f}"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "val_loss": val_loss,
                    'early_stopping_best_score': self.early_stopping.best_score,
                    'early_stopping_counter': self.early_stopping.counter,
                    'early_stopping_val_loss_min': self.early_stopping.val_loss_min,
                }, self.checkpoint_path)
                self.early_stopping(val_loss, self.model, self.device)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break
        if self.rank == 0:
            self.model = self.early_stopping.load_best_model(self.model)
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train 3D CNN for Group Activity Recognition")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Distributed setup
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = get_device()
        local_rank = 0


    train_ids = config.get("train_ids", [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54])
    val_ids = config.get("val_ids", [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51])
    test_ids = config.get("test_ids", [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47])

    # Dataset and DataLoader setup
    train_transform = get_3dcnn_train_transforms(image_size=config['image_size'])
    val_transform = get_3dcnn_val_transforms(image_size=config['image_size'])

    train_dataset = SimplifiedGroupActivityDataset(
        config['samples_base'], video_ids=train_ids, transform=train_transform,
        window_before=config['window_before'], window_after=config['window_after']
    )
    val_dataset = SimplifiedGroupActivityDataset(
        config['samples_base'], video_ids=val_ids, transform=val_transform,
        window_before=config['window_before'], window_after=config['window_after']
    )
    test_dataset = SimplifiedGroupActivityDataset(
        config['samples_base'], video_ids=test_ids, transform=val_transform, # Use val_transform for test set too
        window_before=config['window_before'], window_after=config['window_after']
    )

    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset) if dist.is_initialized() else None
    test_sampler = DistributedSampler(test_dataset) if dist.is_initialized() else None

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None),
        num_workers=config['num_workers'], sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], sampler=val_sampler
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], sampler=test_sampler
    )

    # Model initialization
    model = Video3DClassificationModel(num_classes=len(GROUP_ACTIVITY_MAPPING), pretrained=config['pretrained'])

    # Trainer initialization and training
    trainer = Trainer(config, model, train_loader, val_loader, test_loader, device, resume=args.resume)
    trainer.train()

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()