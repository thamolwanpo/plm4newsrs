# src/unlearning/base.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from configs import BaseConfig
from configs.unlearning.base_unlearning import BaseUnlearningConfig


class BaseUnlearningMethod(ABC):
    """
    Abstract base class for all unlearning methods.

    All unlearning methods must implement:
    - unlearn(): Main unlearning algorithm
    - get_hyperparameters(): Return method-specific hyperparameters

    The method receives BOTH model_config and unlearn_config:
    - model_config: Information about the model (architecture, paths, etc.)
    - unlearn_config: Unlearning-specific parameters (learning rate, iterations, etc.)
    """

    def __init__(
        self,
        model: nn.Module,
        model_config: BaseConfig,
        unlearn_config: BaseUnlearningConfig,
        device: torch.device,
    ):
        """
        Initialize unlearning method.

        Args:
            model: The model to unlearn from (already loaded)
            model_config: Model configuration (architecture, dataset, paths, etc.)
            unlearn_config: Unlearning configuration (method params, learning rate, etc.)
            device: Device to run on
        """
        self.model = model
        self.model_config = model_config
        self.unlearn_config = unlearn_config
        self.device = device

        # Move model to device
        self.model.to(self.device)

        # Training history
        self.history = {"iteration": [], "loss": [], "metrics": []}

        print(f"\n{'='*70}")
        print(f"INITIALIZED: {self.__class__.__name__}")
        print(f"{'='*70}")
        print(f"Model: {model_config.model_name}")
        print(f"Device: {device}")
        print(f"{'='*70}\n")

    @abstractmethod
    def unlearn(
        self,
        forget_loader: torch.utils.data.DataLoader,
        retain_loader: torch.utils.data.DataLoader,
        validation_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> nn.Module:
        """
        Main unlearning algorithm.

        Args:
            forget_loader: DataLoader for forget set
            retain_loader: DataLoader for retain set
            validation_loader: Optional validation data

        Returns:
            Unlearned model

        Example:
            >>> method = InfluenceUnlearning(model, model_config, unlearn_config, device)
            >>> unlearned_model = method.unlearn(forget_loader, retain_loader)
        """
        pass

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get method-specific hyperparameters.

        Returns:
            Dictionary of hyperparameters

        Example:
            >>> method.get_hyperparameters()
            {'learning_rate': 1e-5, 'max_iterations': 100, 'damping': 0.01}
        """
        pass

    def save_checkpoint(self, output_path: Path, metadata: Optional[Dict[str, Any]] = None):
        """
        Save unlearned model checkpoint.

        Args:
            output_path: Path to save checkpoint
            metadata: Additional metadata to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model_config.__dict__,
            "unlearn_config": self.unlearn_config.__dict__,
            "method": self.__class__.__name__,
            "history": self.history,
            "hyperparameters": self.get_hyperparameters(),
        }

        if metadata:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, output_path)
        print(f"✅ Saved checkpoint: {output_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load checkpoint into model.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", self.history)

        print(f"✅ Loaded checkpoint: {checkpoint_path}")

    def evaluate(
        self, dataloader: torch.utils.data.DataLoader, criterion: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader to evaluate on
            criterion: Loss function (default: CrossEntropyLoss)

        Returns:
            Dictionary with evaluation metrics
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue

                # Move ALL tensors in batch to device
                batch_tensors = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_tensors[k] = v.to(self.device)
                    else:
                        batch_tensors[k] = v

                labels = batch_tensors["label"]

                # Forward pass
                scores = self.model(batch_tensors)
                loss = criterion(scores, labels)

                total_loss += loss.item()

                # Calculate accuracy
                predictions = torch.argmax(scores, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0

        return {"loss": avg_loss, "accuracy": accuracy}

    def log_progress(self, iteration: int, metrics: Dict[str, Any]):
        """
        Log training progress.

        Args:
            iteration: Current iteration
            metrics: Metrics to log
        """
        self.history["iteration"].append(iteration)
        self.history["metrics"].append(metrics)

        if "loss" in metrics:
            self.history["loss"].append(metrics["loss"])

    def get_history(self) -> Dict[str, Any]:
        """Get training history."""
        return self.history

    def reset_history(self):
        """Reset training history."""
        self.history = {"iteration": [], "loss": [], "metrics": []}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_config.model_name}, "
            f"device={self.device})"
        )


class IterativeUnlearningMethod(BaseUnlearningMethod):
    """
    Base class for iterative unlearning methods.

    Provides common functionality for methods that:
    - Run for multiple iterations
    - Use gradient-based updates
    - Track convergence
    """

    def __init__(
        self,
        model: nn.Module,
        model_config: BaseConfig,
        unlearn_config: BaseUnlearningConfig,
        device: torch.device,
    ):
        super().__init__(model, model_config, unlearn_config, device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Convergence tracking
        self.best_metric = float("inf")
        self.patience_counter = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer for unlearning.

        Can be overridden by subclasses for custom optimizers.

        Returns:
            Optimizer instance
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.unlearn_config.learning_rate)

    def _check_convergence(self, current_metric: float) -> bool:
        """
        Check if unlearning has converged.

        Args:
            current_metric: Current metric value (loss or similar)

        Returns:
            True if converged, False otherwise
        """
        if hasattr(self.unlearn_config, "early_stopping_patience"):
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1

                if self.patience_counter >= self.unlearn_config.early_stopping_patience:
                    print(f"Early stopping triggered (patience: {self.patience_counter})")
                    return True

        return False

    def _update_learning_rate(self, iteration: int):
        """
        Update learning rate based on schedule.

        Args:
            iteration: Current iteration
        """
        # Simple cosine annealing (can be customized)
        if hasattr(self.unlearn_config, "lr_schedule") and self.unlearn_config.lr_schedule:
            lr = (
                self.unlearn_config.learning_rate
                * 0.5
                * (
                    1
                    + torch.cos(
                        torch.tensor(iteration / self.unlearn_config.max_iterations * 3.14159)
                    )
                )
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr.item()
