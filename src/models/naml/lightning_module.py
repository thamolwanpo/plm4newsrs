# src/models/naml/lightning_module.py

import torch
import torch.nn as nn
import lightning as pl
import torchmetrics

from .recommender import NAMLRecommenderModel


class LitNAMLRecommender(pl.LightningModule):
    """
    PyTorch Lightning wrapper for NAML model.

    Handles:
    - Training loop
    - Validation loop
    - Metrics computation (loss, AUC)
    - Optimizer configuration
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.model = NAMLRecommenderModel(config)
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.val_auc = torchmetrics.AUROC(task="binary")

    def forward(self, batch):
        """Forward pass through model."""
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        """
        Single training step.

        Args:
            batch: Dictionary from dataloader
            batch_idx: Index of batch

        Returns:
            loss: Training loss
        """
        if batch is None:
            return None

        # Move device indicator to correct device (for GloVe mode)
        if "device_indicator" in batch:
            batch["device_indicator"] = batch["device_indicator"].to(self.device)

        # Get predictions and compute loss
        labels = batch["label"]
        scores = self(batch)
        loss = self.loss_fn(scores, labels)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step.

        Args:
            batch: Dictionary from dataloader
            batch_idx: Index of batch
        """
        if batch is None:
            return

        # Move device indicator to correct device (for GloVe mode)
        if "device_indicator" in batch:
            batch["device_indicator"] = batch["device_indicator"].to(self.device)

        # Get predictions and compute loss
        labels = batch["label"]
        scores = self(batch)
        val_loss = self.loss_fn(scores, labels)

        # Log loss
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)

        # Calculate AUC
        # Convert scores to probabilities
        probs = torch.softmax(scores, dim=1)

        # Create binary labels: one-hot encoding where label index = 1
        binary_labels = (torch.arange(scores.size(1)).to(self.device) == labels.view(-1, 1)).float()

        # Update AUC metric
        self.val_auc.update(probs.flatten(), binary_labels.flatten().int())

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        try:
            auc_score = self.val_auc.compute()
            self.log("val_auc", auc_score, on_epoch=True, prog_bar=True, logger=True)
        except ValueError:
            # Handle case where AUC cannot be computed (e.g., only one class)
            self.log("val_auc", 0.0, on_epoch=True, prog_bar=True, logger=True)

        # Reset metric for next epoch
        self.val_auc.reset()

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def predict_step(self, batch, batch_idx):
        """
        Single prediction step.

        Args:
            batch: Dictionary from dataloader
            batch_idx: Index of batch

        Returns:
            scores: Predicted click scores
        """
        if "device_indicator" in batch:
            batch["device_indicator"] = batch["device_indicator"].to(self.device)

        scores = self(batch)
        return scores
