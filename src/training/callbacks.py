from lightning.pytorch.callbacks import Callback
import torch
import time
from pathlib import Path
import json


class MetricLogger(Callback):
    """Log metrics to a JSON file for later analysis."""

    def __init__(self, log_dir: Path, filename: str = "metrics.json"):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        """Save metrics at end of each validation epoch."""
        metrics = {
            "epoch": trainer.current_epoch,
            "train_loss": trainer.callback_metrics.get("train_loss_epoch", 0).item(),
            "val_loss": trainer.callback_metrics.get("val_loss", 0).item(),
            "val_auc": trainer.callback_metrics.get("val_auc", 0).item(),
        }
        self.metrics.append(metrics)

        # Save to file
        with open(self.log_file, "w") as f:
            json.dump(self.metrics, f, indent=2)


class LearningRateMonitor(Callback):
    """Monitor and log learning rate."""

    def on_train_epoch_start(self, trainer, pl_module):
        """Log learning rate at start of each epoch."""
        optimizer = trainer.optimizers[0]
        lr = optimizer.param_groups[0]["lr"]
        pl_module.log("learning_rate", lr)


class TimeLogger(Callback):
    """Log training time per epoch."""

    def __init__(self):
        super().__init__()
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        """Record start time of epoch."""
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch duration."""
        if self.epoch_start_time:
            duration = time.time() - self.epoch_start_time
            pl_module.log("epoch_duration", duration)
            print(f"\nEpoch {trainer.current_epoch} completed in {duration:.2f}s")


class GradientNormMonitor(Callback):
    """Monitor gradient norms to detect training issues."""

    def on_after_backward(self, trainer, pl_module):
        """Log gradient norm after backward pass."""
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        pl_module.log("grad_norm", total_norm)


class ModelSizeLogger(Callback):
    """Log model size and parameter count."""

    def on_fit_start(self, trainer, pl_module):
        """Log model statistics at start of training."""
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")


class BestMetricTracker(Callback):
    """Track and save best metric values."""

    def __init__(self, save_dir: Path):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_metrics = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        """Update best metrics."""
        metrics = trainer.callback_metrics

        for key, value in metrics.items():
            if key.startswith("val_"):
                if key not in self.best_metrics:
                    self.best_metrics[key] = {"value": float("-inf"), "epoch": 0}

                current_value = value.item() if torch.is_tensor(value) else value

                if current_value > self.best_metrics[key]["value"]:
                    self.best_metrics[key] = {
                        "value": current_value,
                        "epoch": trainer.current_epoch,
                    }

        # Save to file
        with open(self.save_dir / "best_metrics.json", "w") as f:
            json.dump(self.best_metrics, f, indent=2)
