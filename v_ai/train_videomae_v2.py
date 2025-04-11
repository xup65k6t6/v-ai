# v_ai/train_videomae_v2.py

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
from v_ai.transforms import VideoMAETransform # Assuming VideoMAETransform is the correct one
from v_ai.utils.earlystopping import EarlyStopping
from v_ai.utils.utils import get_checkpoint_dir, get_device

os.environ["WANDB_SILENT"] = "true"

class Trainer:
    def __init__(self, config, model, train_loader, val_loader, test_loader, device, resume=False):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.resume = resume
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        if dist.is_initialized():
            # Ensure the device is correctly set for DDP
            if device.type == 'cuda':
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[device.index], output_device=device.index)
            else: # CPU
                 self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader # Kept for potential future use

        self.criterion = nn.CrossEntropyLoss()
        # Use AdamW as per original script
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0.05))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=config.get('scheduler_patience', 2))
        
        self.checkpoint_dir = config['checkpoint_dir']
        self.early_stopping = EarlyStopping(patience=config['patience'], verbose=True, path=os.path.join(self.checkpoint_dir, "videomae_v2_best.pt"))
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "videomae_v2_checkpoint_last.pt") 

        if self.rank == 0:
            wandb_project = config.get("wandb_project", "volleyball_group_activity")
            wandb_run_name = config.get("wandb_run_name", "videomae_v2_run")
            wandb.login(key=os.environ.get("WANDB_API_KEY"))
            wandb.init(project=wandb_project, name=wandb_run_name, config=config)
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
            frames = batch["frames"].to(self.device)
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

        # Calculate metrics
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
                frames = batch["frames"].to(self.device)
                labels = batch["group_label"].to(self.device)

                logits = self.model(frames)
                loss = self.criterion(logits, labels)
                running_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Calculate metrics
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
        start_epoch = 0
        # Load checkpoint if resuming
        if self.resume and os.path.exists(self.checkpoint_path):
            # Map location based on current device
            map_location = self.device
            checkpoint = torch.load(self.checkpoint_path, map_location=map_location)
            
            # Adjust loading for DDP model state_dict if necessary
            model_state_dict = checkpoint['model_state_dict']
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                 # Load onto the underlying model directly
                 self.model.module.load_state_dict(model_state_dict)
            else:
                 self.model.load_state_dict(model_state_dict)

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            # Restore early stopping state
            self.early_stopping.best_score = checkpoint.get('early_stopping_best_score', None)
            self.early_stopping.counter = checkpoint.get('early_stopping_counter', 0)
            self.early_stopping.val_loss_min = checkpoint.get('early_stopping_val_loss_min', float('inf'))
            if self.rank == 0:
                print(f"Resuming training from epoch {start_epoch}")
        elif self.rank == 0:
            print("Starting training from scratch")

        for epoch in range(start_epoch, self.config['num_epochs']):
            # Set epoch for distributed sampler
            if hasattr(self.train_loader, 'sampler') and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            if self.rank == 0:
                print(f"Epoch {epoch+1}/{self.config['num_epochs']}")

            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()

            # Step the scheduler based on validation loss
            self.scheduler.step(val_loss)

            if self.rank == 0:
                # Log metrics to WandB
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

                # Print epoch summary
                print(
                    f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                    f"Acc = {val_metrics['accuracy']:.4f}, Prec = {val_metrics['precision']:.4f}, "
                    f"Recall = {val_metrics['recall']:.4f}, F1 = {val_metrics['f1']:.4f}"
                )

                # Save the last checkpoint
                # Ensure we save the underlying model state_dict when using DDP
                model_state_to_save = self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model_state_to_save,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "val_loss": val_loss,
                    'early_stopping_best_score': self.early_stopping.best_score,
                    'early_stopping_counter': self.early_stopping.counter,
                    'early_stopping_val_loss_min': self.early_stopping.val_loss_min,
                }, self.checkpoint_path)

                # Early stopping check (pass the underlying model if DDP)
                model_to_check = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
                self.early_stopping(val_loss, model_to_check, self.device)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

        # Load the best model state after training loop (only on rank 0)
        if self.rank == 0:
            # Load best model state into the underlying model if using DDP
            model_to_load = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            self.model = self.early_stopping.load_best_model(model_to_load)
            print("Loaded best model weights from early stopping.")
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train VideoMAE V2 for Volleyball Group Activity Recognition")
    parser.add_argument("--config", type=str, default="config/config_videomae.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Ensure checkpoint directory exists
    checkpoint_dir = config.get("checkpoint_dir", get_checkpoint_dir())
    os.makedirs(checkpoint_dir, exist_ok=True)
    config['checkpoint_dir'] = checkpoint_dir # Store resolved path back into config

    # Distributed setup
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device) # Important for DDP
        else:
            device = torch.device("cpu")
    else:
        # Use get_device utility for single process run
        device = get_device()
        local_rank = 0 # Default for non-distributed

    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"Initialized distributed training with backend: {dist.get_backend()}, world size: {dist.get_world_size()}")
        print(f"Using device: {device}")
    elif not dist.is_initialized():
         print(f"Not using distributed training. Using device: {device}")


    # Extract parameters from config (use .get for safety)
    batch_size = config.get("batch_size", 1)
    num_workers = config.get("num_workers", 4)
    samples_base = config.get("samples_base", os.path.join(os.getcwd(), "data", "videos"))
    train_ids = config.get("train_ids", [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54])
    val_ids = config.get("val_ids", [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51])
    test_ids = config.get("test_ids", [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47])
    window_before = config.get("window_before", 3)
    window_after = config.get("window_after", 4)
    num_frames = config.get("num_frames", 8)
    image_size = config.get("image_size", 320) 
    videomae_v2_model_name = config.get("videomae_v2_model_name", "MCG-NJU/videomae-base-finetuned-kinetics")

    # Initialize Transforms
    # Assuming VideoMAETransform handles both train and val/test cases internally or is suitable for both
    transform = VideoMAETransform(model_name=videomae_v2_model_name, image_size=image_size)

    # Datasets
    train_dataset = VideoMAE_V2_Dataset(
        samples_base=samples_base, video_ids=train_ids, transform=transform,
        window_before=window_before, window_after=window_after, num_frames=num_frames, model_name=videomae_v2_model_name
    )
    val_dataset = VideoMAE_V2_Dataset(
        samples_base=samples_base, video_ids=val_ids, transform=transform,
        window_before=window_before, window_after=window_after, num_frames=num_frames, model_name=videomae_v2_model_name
    )
    test_dataset = VideoMAE_V2_Dataset(
        samples_base=samples_base, video_ids=test_ids, transform=transform,
        window_before=window_before, window_after=window_after, num_frames=num_frames, model_name=videomae_v2_model_name
    )

    # Samplers for Distributed Training
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if dist.is_initialized() else None # Keep for consistency

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), # Shuffle only if not distributed
        num_workers=num_workers, sampler=train_sampler, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, sampler=val_sampler, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, sampler=test_sampler, pin_memory=True
    )

    if dist.get_rank() == 0 if dist.is_initialized() else True:
        print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}")
        print(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}")


    # Model Initialization
    model = VideoMAEV2ClassificationModel(
        num_frames=num_frames,
        image_size=image_size,
        pretrained=config.get('pretrained', True),
        model_name=videomae_v2_model_name
    )

    # Trainer Initialization and Training
    trainer = Trainer(config, model, train_loader, val_loader, test_loader, device, resume=args.resume)
    trainer.train()

    # Cleanup distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
