# src/training/dynamics.py

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import Callback

from src.models.components.losses import per_group_softmax, per_interaction_bce


class TrainingDynamicsTracker(Callback):
    """Record training dynamics, keyed by sample id (row position in the training CSV).

    Per-candidate ledgers (sigmoid of each candidate score vs its own label;
    one entry per interaction, positives and negatives):
    - per_instance_loss: {sample_id: [bce_ep1, bce_ep2, ...]}
    - c_proxy: {sample_id: probability of the true class, averaged over epochs}
    - el2n: {sample_id: |sigmoid(score) - label|}

    Per-group ledgers (softmax over the K+1 candidates, i.e. the cross-entropy
    training objective; one entry per group, keyed by the positive's sample id):
    - per_group_ce_loss: {sample_id: [ce_ep1, ce_ep2, ...]}
    - softmax_c_proxy: {sample_id: softmax probability of the positive, averaged over epochs}
    - softmax_el2n: {sample_id: ||softmax - one-hot label||_2}

    Losses and confidences come from the standard training forward pass
    (training_step must return the detached ``scores``); both EL2N scores come
    from one inference-only pass over the training data after the first epoch.
    The batch must carry ``sample_ids`` [batch, K+1] (see BaseNewsDataset);
    ids < 0 (reused negatives) are skipped.
    """

    def __init__(self, el2n_loader: DataLoader):
        """
        Args:
            el2n_loader: Unshuffled loader over the training set for the EL2N pass
        """
        super().__init__()
        self.el2n_loader = el2n_loader
        self._step_buffers: Dict[str, List[torch.Tensor]] = {}
        # One dict of CPU tensors per finished epoch (keys: see _metrics)
        self.epoch_records: List[Dict[str, torch.Tensor]] = []
        self.el2n: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _metrics(scores: torch.Tensor, batch) -> Dict[str, torch.Tensor]:
        """Per-candidate and per-group metrics of one batch (1-D, reused negatives dropped)."""
        if "sample_ids" not in batch:
            # Update your dataset/dataloader to yield a unique `sample_ids` per interaction
            raise KeyError("TrainingDynamicsTracker needs batch['sample_ids'] [batch, K+1]")
        sample_ids, labels = batch["sample_ids"], batch["label"]
        mask = sample_ids >= 0

        bce, confidence = per_interaction_bce(scores, labels)
        ce, softmax_confidence, softmax_el2n = per_group_softmax(scores, labels)
        targets = torch.nn.functional.one_hot(labels, num_classes=scores.size(1))

        return {
            "ids": sample_ids[mask],
            "loss": bce[mask],
            "confidence": confidence[mask],
            "el2n": (torch.sigmoid(scores) - targets).abs()[mask],
            "group_ids": sample_ids.gather(1, labels.unsqueeze(1)).squeeze(1),
            "group_loss": ce,
            "group_confidence": softmax_confidence,
            "group_el2n": softmax_el2n,
        }

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._step_buffers = {}

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        if not outputs or batch is None or "scores" not in outputs:
            return
        metrics = self._metrics(outputs["scores"], batch)
        # Keep the values on device; they are moved to CPU once per epoch
        for key in ("ids", "loss", "confidence", "group_ids", "group_loss", "group_confidence"):
            self._step_buffers.setdefault(key, []).append(metrics[key])

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._step_buffers:
            return
        self.epoch_records.append(_concat(self._step_buffers))
        self._step_buffers = {}

        if len(self.epoch_records) == 1:
            self.el2n = self._compute_el2n(pl_module)

    @torch.no_grad()
    def _compute_el2n(self, pl_module: pl.LightningModule) -> Dict[str, torch.Tensor]:
        """Per-candidate and per-group EL2N of the training set, without updating the model."""
        was_training = pl_module.training
        pl_module.eval()

        buffers: Dict[str, List[torch.Tensor]] = {}
        for batch in self.el2n_loader:
            if batch is None:
                continue
            batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
            metrics = self._metrics(pl_module(batch), batch)
            for key in ("ids", "el2n", "group_ids", "group_el2n"):
                buffers.setdefault(key, []).append(metrics[key])

        if was_training:
            pl_module.train()
        return _concat(buffers)

    def ledgers(self) -> Dict[str, dict]:
        """All ledgers as {sample_id: value} dicts."""
        per_instance_loss, c_proxy = _loss_and_confidence(self.epoch_records, "ids", "")
        per_group_ce_loss, softmax_c_proxy = _loss_and_confidence(
            self.epoch_records, "group_ids", "group_"
        )
        el2n, softmax_el2n = {}, {}
        if self.el2n is not None:
            el2n = dict(zip(self.el2n["ids"].tolist(), self.el2n["el2n"].tolist()))
            softmax_el2n = dict(
                zip(self.el2n["group_ids"].tolist(), self.el2n["group_el2n"].tolist())
            )

        return {
            "per_instance_loss": per_instance_loss,
            "c_proxy": c_proxy,
            "el2n": el2n,
            "per_group_ce_loss": per_group_ce_loss,
            "softmax_c_proxy": softmax_c_proxy,
            "softmax_el2n": softmax_el2n,
            "num_epochs": len(self.epoch_records),
        }

    def save(self, path: Path) -> Path:
        """Save the ledgers with torch.save()."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.ledgers(), path)
        return path


def _concat(buffers: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Concatenate per-step tensors and move them to CPU (ids stay integer)."""
    return {
        key: torch.cat(values).cpu() if key.endswith("ids") else torch.cat(values).float().cpu()
        for key, values in buffers.items()
    }


def _loss_and_confidence(records: List[Dict[str, torch.Tensor]], ids_key: str, prefix: str):
    """{id: [loss per epoch]} and {id: confidence averaged over epochs}."""
    losses: Dict[int, List[float]] = {}
    confidence_sum: Dict[int, float] = {}
    for record in records:
        for sample_id, loss, conf in zip(
            record[ids_key].tolist(),
            record[f"{prefix}loss"].tolist(),
            record[f"{prefix}confidence"].tolist(),
        ):
            losses.setdefault(sample_id, []).append(loss)
            confidence_sum[sample_id] = confidence_sum.get(sample_id, 0.0) + conf
    confidence = {
        sample_id: total / len(losses[sample_id]) for sample_id, total in confidence_sum.items()
    }
    return losses, confidence
