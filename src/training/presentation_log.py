# src/training/presentation_log.py

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.callbacks import Callback

from src.models.components import AdditiveAttention
from src.models.components.losses import per_group_softmax


class PresentationLogger(Callback):
    """Log every presentation of a K+1 group during training, plus per-article gradient sums.

    These numbers cannot be recovered from checkpoints. One row per group per
    epoch, in training order (``presentations``):
    - group_id: sample id of the positive (as per_group_ce_loss in the training dynamics)
    - epoch, step: epoch and optimizer step of the presentation
    - scores, probs: logits s and softmax p over the K+1 candidates [P, K+1]
    - loss, p0: cross-entropy and probability of the clicked item
    - margin: s_0 - max s_neg; ranked_first: margin > 0 (forgetting events)
    - attention: additive-attention weight alpha_i of each history slot [P, H]
      (NaN for LSTUR, whose user encoder is a GRU)
    - history_grad_norm: ||dl/dr_i|| of each history slot's news vector [P, H],
      for the group's own loss l (= batch size x the batch-loss gradient)
    - candidate_item_idx [P, K+1], history_item_idx [P, H]: article indices (-1 = padding)

    Per article (``accumulated_grad``, [num_articles, d] each, rows = article index):
    running sums of lr_t * dL/d(news vector) of the batch loss L, i.e. the
    gradient that drives the actual update, split by the role the article
    played: ``history`` (r_i), ``positive`` (v_0, the clicked item) and
    ``negative`` (v_j, j > 0, reused negatives included). Padding slots are
    left out of the sums; the per-slot columns keep them (the user encoder
    does not mask padding), so filter with history_item_idx >= 0.

    Gradients come from tensor hooks on the news vectors in the normal backward
    pass (the recommender's ``news_vector_hook``); lr_t is read from the
    optimizer before each step.
    """

    ROLES = ("history", "positive", "negative")

    def __init__(self, item_ids: List[str], item_is_fake: List[int]):
        """
        Args:
            item_ids: Article id of each article index (news_items.csv order)
            item_is_fake: is_fake flag of each article index
        """
        super().__init__()
        self.item_ids = item_ids
        self.item_is_fake = torch.tensor(item_is_fake, dtype=torch.bool)
        # One dict of CPU tensors per finished epoch (keys: see presentations in the docstring)
        self.epoch_records: List[Dict[str, torch.Tensor]] = []
        self.accumulated_grad: Dict[str, torch.Tensor] = {}

        self._step_buffers: Dict[str, List[torch.Tensor]] = {}
        self._capturing = False
        self._grads: Dict[str, torch.Tensor] = {}
        self._attention: Optional[torch.Tensor] = None
        self._lr: Optional[float] = None
        self._step = 0
        self._hook_handles = []
        self._warned_no_grad = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        model = pl_module.model
        model.news_vector_hook = self._capture_news_vectors
        attention = _history_attention(model.user_encoder)
        if attention is not None:
            self._hook_handles.append(attention.register_forward_hook(self._capture_attention))

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.model.news_vector_hook = None
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _capture_news_vectors(self, candidate_vectors, history_vectors) -> None:
        """Register gradient hooks on the news vectors of a training forward pass."""
        if not self._capturing:
            return
        for role, vectors in (("candidate", candidate_vectors), ("history", history_vectors)):
            if vectors.requires_grad:
                vectors.register_hook(self._grad_saver(role))
            elif not self._warned_no_grad:
                warnings.warn("News vectors do not require grad: gradients will not be logged")
                self._warned_no_grad = True

    def _grad_saver(self, role: str):
        def save_grad(grad: torch.Tensor) -> None:
            self._grads[role] = grad.detach()

        return save_grad

    def _capture_attention(self, module, inputs, output) -> None:
        if self._capturing:
            self._attention = output[1].detach()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._step_buffers = {}

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx: int
    ) -> None:
        self._capturing = True
        self._grads, self._attention, self._lr = {}, None, None
        self._step = trainer.global_step

    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer
    ) -> None:
        self._lr = optimizer.param_groups[0]["lr"]

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        self._capturing = False
        if not outputs or batch is None or "scores" not in outputs:
            return
        for key in ("candidate_item_idx", "history_item_idx"):
            if key not in batch:
                # Update your dataset/dataloader to yield article indices (see BaseNewsDataset)
                raise KeyError(f"PresentationLogger needs batch['{key}']")

        record = self._presentations(outputs["scores"].float(), batch)
        for key, value in record.items():
            self._step_buffers.setdefault(key, []).append(value)

        if self._lr is not None:
            self._accumulate(batch)

    def _presentations(self, scores: torch.Tensor, batch) -> Dict[str, torch.Tensor]:
        """One row per group of the batch (see the class docstring)."""
        labels = batch["label"]
        batch_size = scores.size(0)
        history_idx = batch["history_item_idx"]
        nan = torch.full(history_idx.shape, float("nan"), device=scores.device)

        loss, p0, _ = per_group_softmax(scores, labels)
        is_positive = F.one_hot(labels, num_classes=scores.size(1)).bool()
        s_pos = scores.gather(1, labels.unsqueeze(1)).squeeze(1)
        margin = s_pos - scores.masked_fill(is_positive, float("-inf")).max(dim=1).values

        history_grad = self._grads.get("history")
        if history_grad is None:
            history_grad_norm = nan
        else:
            # The batch loss is the mean over the groups: x batch_size = each group's own gradient
            history_grad_norm = batch_size * torch.linalg.vector_norm(history_grad.float(), dim=-1)

        return {
            "group_id": batch["sample_ids"].gather(1, labels.unsqueeze(1)).squeeze(1),
            "step": torch.full_like(labels, self._step),
            "scores": scores,
            "probs": torch.softmax(scores, dim=1),
            "loss": loss,
            "p0": p0,
            "margin": margin,
            "ranked_first": margin > 0,
            "attention": nan if self._attention is None else self._attention.float(),
            "history_grad_norm": history_grad_norm,
            "candidate_item_idx": batch["candidate_item_idx"],
            "history_item_idx": history_idx,
        }

    def _accumulate(self, batch) -> None:
        """Add lr_t * gradient of each news vector to its article's row, per role."""
        history_grad = self._grads.get("history")
        if history_grad is not None:
            self._add("history", batch["history_item_idx"].flatten(), history_grad.flatten(0, 1))

        candidate_grad = self._grads.get("candidate")
        if candidate_grad is not None:
            candidate_idx = batch["candidate_item_idx"]
            is_positive = F.one_hot(batch["label"], num_classes=candidate_idx.size(1)).bool()
            self._add("positive", candidate_idx[is_positive], candidate_grad[is_positive])
            self._add("negative", candidate_idx[~is_positive], candidate_grad[~is_positive])

    def _add(self, role: str, item_idx: torch.Tensor, grads: torch.Tensor) -> None:
        valid = item_idx >= 0
        if role not in self.accumulated_grad:
            self.accumulated_grad[role] = torch.zeros(
                len(self.item_ids), grads.size(-1), device=grads.device
            )
        # index_put_ with accumulate=True is deterministic under Trainer(deterministic=True)
        self.accumulated_grad[role].index_put_(
            (item_idx[valid],), self._lr * grads[valid].float(), accumulate=True
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._step_buffers:
            return
        record = {key: torch.cat(values).cpu() for key, values in self._step_buffers.items()}
        record["epoch"] = torch.full_like(record["group_id"], trainer.current_epoch)
        self.epoch_records.append(record)
        self._step_buffers = {}

    def log(self) -> dict:
        """Presentations of all epochs (flat, in training order) and the per-article sums."""
        presentations = {}
        if self.epoch_records:
            presentations = {
                key: torch.cat([record[key] for record in self.epoch_records])
                for key in self.epoch_records[0]
            }
        return {
            "presentations": presentations,
            "accumulated_grad": {role: t.cpu() for role, t in self.accumulated_grad.items()},
            "item_ids": self.item_ids,
            "item_is_fake": self.item_is_fake,
            "num_epochs": len(self.epoch_records),
        }

    def save(self, path: Path) -> Path:
        """Save the log with torch.save()."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.log(), path)
        return path


def _history_attention(user_encoder) -> Optional[AdditiveAttention]:
    """The additive attention that pools the history slots (None for LSTUR)."""
    for name in ("news_attention", "attention"):
        module = getattr(user_encoder, name, None)
        if isinstance(module, AdditiveAttention):
            return module
    return None
