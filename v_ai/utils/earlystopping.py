import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="best_model.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.best_state = None

    def __call__(self, val_loss, model, device):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, device)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.verbose:
                print(f"Best loss: {self.val_loss_min:.6f} | Current loss: {val_loss:.6f}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, device)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, device):
        """Save model checkpoint and keep best state in memory."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(self.best_state, self.path)
        self.val_loss_min = val_loss

    def load_best_model(self, model):
        """Load the best model state from memory."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model
