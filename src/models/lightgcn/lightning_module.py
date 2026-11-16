# src/models/lightgcn/lightning_module.py

from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchmetrics

from .graph import build_interaction_edges, tokenize_titles
from .recommender import LightGCNRecommenderModel


class LitLightGCNRecommender(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the hybrid LightGCN model.

    Same interface as the other architectures (scores over the K+1
    candidates, cross-entropy by default), plus:
    - build_graph(): the trainer calls it once after construction, with the
      training file, to set the interaction graph and the article titles
    - loss "bpr" (config.loss) and the paper's L2 on layer-0 embeddings
    - two learning rates: embedding_learning_rate for the graph parameters,
      learning_rate for the language model when it is fine-tuned
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.model = LightGCNRecommenderModel(config)
        self.ce_loss = nn.CrossEntropyLoss()

        # Metrics
        self.val_auc = torchmetrics.AUROC(task="binary")

    # ------------------------------------------------------------------ #
    # Graph
    # ------------------------------------------------------------------ #
    def build_graph(
        self,
        train_csv: Union[str, Path],
        user_map: Dict[str, int],
        item_ids: List[str],
        titles: Dict[str, str],
        tokenizer,
    ) -> None:
        """Interaction graph of the training file + tokenized titles of every article.

        Args:
            train_csv: the CSV this model is trained on
            user_map: user_id -> user index (same map as the datasets)
            item_ids: article id of each article index (news_items.csv order)
            titles: item_id -> title
            tokenizer: tokenizer of config.model_name
        """
        item_index = {item_id: i for i, item_id in enumerate(item_ids)}
        edges = build_interaction_edges(train_csv, user_map, item_index, self.config.graph_edges)
        input_ids, attention_mask = tokenize_titles(
            tokenizer, item_ids, titles, self.config.max_seq_length
        )
        self.model.set_training_data(edges, input_ids, attention_mask)

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #
    def _ranking_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.config.loss == "bpr":
            positive = scores.gather(1, labels.unsqueeze(1))
            is_negative = ~F.one_hot(labels, num_classes=scores.size(1)).bool()
            diff = (positive - scores)[is_negative]
            return -F.logsigmoid(diff).mean()
        return self.ce_loss(scores, labels)

    def forward(self, batch):
        """Forward pass through model."""
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        """
        Single training step.

        Returns:
            Dict with the training loss (ranking loss + L2, used for backward)
            and the detached scores [batch, K+1], from which the
            TrainingDynamicsTracker callback computes the training dynamics.
        """
        if batch is None:
            return None

        labels = batch["label"]
        scores = self(batch)
        ranking_loss = self._ranking_loss(scores, labels)
        loss = ranking_loss
        if self.config.l2_reg > 0 and self.model.last_l2 is not None:
            loss = loss + self.config.l2_reg * self.model.last_l2
            self.log("train_l2", self.model.last_l2, on_step=False, on_epoch=True, logger=True)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ranking_loss", ranking_loss, on_step=False, on_epoch=True, logger=True)

        return {"loss": loss, "scores": scores.detach()}

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return

        labels = batch["label"]
        scores = self(batch)
        val_loss = self._ranking_loss(scores, labels)

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)

        probs = torch.softmax(scores, dim=1)
        binary_labels = (torch.arange(scores.size(1)).to(self.device) == labels.view(-1, 1)).float()
        self.val_auc.update(probs.flatten(), binary_labels.flatten().int())

    def on_validation_epoch_end(self):
        try:
            auc_score = self.val_auc.compute()
            self.log("val_auc", auc_score, on_epoch=True, prog_bar=True, logger=True)
        except ValueError:
            self.log("val_auc", 0.0, on_epoch=True, prog_bar=True, logger=True)
        self.val_auc.reset()

    # A fine-tuned LM changes every step: refresh the cached title vectors of
    # all articles at the start of each training and validation epoch.
    def on_train_epoch_start(self):
        if self.model.lm_trainable:
            self.model.invalidate_content_bank()

    def on_validation_epoch_start(self):
        if self.model.lm_trainable:
            self.model.invalidate_content_bank()

    def configure_optimizers(self):
        """AdamW with two groups: graph parameters first (their lr is the one the
        PresentationLogger reads), then the fine-tuned language model layers."""
        graph_params, lm_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (lm_params if name.startswith("lm.") else graph_params).append(param)

        groups = [{"params": graph_params, "lr": self.config.embedding_learning_rate}]
        if lm_params:
            groups.append({"params": lm_params, "lr": self.config.learning_rate})
        return torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)

    def predict_step(self, batch, batch_idx):
        return self(batch)
