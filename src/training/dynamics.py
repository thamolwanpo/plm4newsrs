# src/training/dynamics.py

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import Callback


class TrainingDynamicsTracker(Callback):
    """Record per-interaction training dynamics, keyed by sample id.

    Ledgers (sample id = row position in the training CSV):
    - per_instance_loss: {sample_id: [loss_ep1, loss_ep2, ...]}, the BCE of
      every interaction from the standard training forward pass.
    - c_proxy: {sample_id: avg_confidence}, the predicted probability of the
      true class averaged over the epochs, also from the training forward pass.
    - el2n: {sample_id: |sigmoid(score) - label|}, from one inference-only
      pass over the training data at the end of the first epoch.

    The batch must carry ``sample_ids`` [batch, K+1] (see BaseNewsDataset), and
    training_step must return ``per_sample_loss`` and ``confidence`` next to
    ``loss`` (see per_interaction_bce). Ids < 0 (reused negatives) are skipped.
    """

    def __init__(self, el2n_loader: DataLoader):
        """
        Args:
            el2n_loader: Unshuffled loader over the training set for the EL2N pass
        """
        super().__init__()
        self.el2n_loader = el2n_loader
        self._step_buffers: Dict[str, List[torch.Tensor]] = {}
        # One {"ids", "loss", "confidence"} dict of CPU tensors per finished epoch
        self.epoch_records: List[Dict[str, torch.Tensor]] = []
        self.el2n: Optional[Dict[str, torch.Tensor]] = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._step_buffers = {"ids": [], "loss": [], "confidence": []}

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        if not outputs or batch is None or "per_sample_loss" not in outputs:
            return
        if "sample_ids" not in batch:
            # Update your dataset/dataloader to yield a unique `sample_ids` per interaction
            raise KeyError("TrainingDynamicsTracker needs batch['sample_ids'] [batch, K+1]")

        # Keep the values on device; they are moved to CPU once per epoch
        mask = batch["sample_ids"] >= 0
        self._step_buffers["ids"].append(batch["sample_ids"][mask])
        self._step_buffers["loss"].append(outputs["per_sample_loss"][mask])
        self._step_buffers["confidence"].append(outputs["confidence"][mask])

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._step_buffers.get("ids"):
            return
        self.epoch_records.append(
            {key: torch.cat(values).float().cpu() for key, values in self._step_buffers.items()}
        )
        self._step_buffers = {}

        if len(self.epoch_records) == 1:
            self.el2n = self._compute_el2n(pl_module)

    @torch.no_grad()
    def _compute_el2n(self, pl_module: pl.LightningModule) -> Dict[str, torch.Tensor]:
        """|sigmoid(score) - label| of every interaction, without updating the model."""
        was_training = pl_module.training
        pl_module.eval()

        ids, scores_el2n = [], []
        for batch in self.el2n_loader:
            if batch is None:
                continue
            batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
            scores = pl_module(batch)
            targets = torch.nn.functional.one_hot(batch["label"], num_classes=scores.size(1))
            el2n = (torch.sigmoid(scores) - targets).abs()

            mask = batch["sample_ids"] >= 0
            ids.append(batch["sample_ids"][mask])
            scores_el2n.append(el2n[mask])

        if was_training:
            pl_module.train()
        return {"ids": torch.cat(ids).cpu(), "el2n": torch.cat(scores_el2n).float().cpu()}

    def ledgers(self) -> Dict[str, dict]:
        """The three ledgers as {sample_id: value} dicts."""
        per_instance_loss: Dict[int, List[float]] = {}
        confidence_sum: Dict[int, float] = {}
        for record in self.epoch_records:
            for sample_id, loss, conf in zip(
                record["ids"].tolist(), record["loss"].tolist(), record["confidence"].tolist()
            ):
                per_instance_loss.setdefault(sample_id, []).append(loss)
                confidence_sum[sample_id] = confidence_sum.get(sample_id, 0.0) + conf

        c_proxy = {
            sample_id: total / len(per_instance_loss[sample_id])
            for sample_id, total in confidence_sum.items()
        }
        el2n = {}
        if self.el2n is not None:
            el2n = dict(zip(self.el2n["ids"].tolist(), self.el2n["el2n"].tolist()))

        return {
            "per_instance_loss": per_instance_loss,
            "c_proxy": c_proxy,
            "el2n": el2n,
            "num_epochs": len(self.epoch_records),
        }

    def save(self, path: Path) -> Path:
        """Save the ledgers with torch.save()."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.ledgers(), path)
        return path
