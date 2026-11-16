# src/models/lightgcn/recommender.py

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class LightGCNRecommenderModel(nn.Module):
    """
    Hybrid LightGCN: graph propagation over PLM title vectors.

    Based on: "LightGCN: Simplifying and Powering Graph Convolution Network
    for Recommendation" (He et al., SIGIR 2020).

    Nodes: users 0..U-1 (0 = unknown user, never in the graph) and articles
    0..M-1 (rows of news_items.csv), stacked as [users; articles].

    Layer 0:
        user     e_u^(0) = user ID embedding
        article  e_i^(0) = W * (PLM(title_i)[CLS] - mean over articles)
                           (+ article ID embedding, zero-init)
    Propagation (no feature transform, no nonlinearity, no self-loops):
        E^(k+1) = D^-1/2 A D^-1/2 E^(k),   final E = mean(E^(0), ..., E^(K))
    Nodes without any edge (articles never clicked in the training file, the
    unknown user) keep E^(k) = E^(0) at every layer, so an unseen article is
    scored with its own content vector at the same scale as the others.
    Score: <user vector, article vector> (see config.user_mode).

    The graph (edge_index) and the tokenized titles of every article are
    persistent buffers: they are saved in the checkpoint, so evaluation needs
    neither the training file nor a rebuilt graph.

    PLM title vectors of all articles are kept in ``content_bank`` (not
    saved). With a frozen LM it is computed once. With a fine-tuned LM it is
    recomputed at the start of every training / validation epoch, and during
    training the articles of the batch are re-encoded with gradients (the
    others use their cached vector), as a memory bank.
    """

    _GRAPH_BUFFERS = ("edge_index", "item_title_input_ids", "item_title_attention_mask")

    def __init__(self, config):
        super().__init__()
        self.config = config
        if "glove" in config.model_name.lower():
            raise ValueError("LightGCN supports transformer backbones only (BERT / RoBERTa)")
        if config.num_items <= 0:
            raise ValueError("config.num_items must be set to the number of rows of news_items.csv")

        self.num_users = config.num_users
        self.num_items = config.num_items
        self.num_layers = config.num_layers
        dim = config.embedding_dim

        self._init_language_model()

        self.content_proj = nn.Linear(self.hidden_size, dim)
        nn.init.xavier_uniform_(self.content_proj.weight)
        nn.init.zeros_(self.content_proj.bias)

        self.user_embedding = nn.Embedding(self.num_users, dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)

        self.item_id_embedding: Optional[nn.Embedding] = None
        if config.item_init == "content+id":
            # Zero-init: an article never seen in training is scored by its content alone
            self.item_id_embedding = nn.Embedding(self.num_items, dim)
            nn.init.zeros_(self.item_id_embedding.weight)

        # Filled by set_training_data() (training) or by the checkpoint (evaluation)
        self.register_buffer("edge_index", torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer(
            "item_title_input_ids", torch.zeros(0, config.max_seq_length, dtype=torch.long)
        )
        self.register_buffer(
            "item_title_attention_mask", torch.zeros(0, config.max_seq_length, dtype=torch.long)
        )
        self.register_buffer("content_bank", torch.zeros(0, self.hidden_size), persistent=False)
        self._bank_valid = False
        self._norm_adj = None  # (src, dst, weight) of the normalized adjacency, built lazily

        # For analysis
        self.store_intermediate_outputs = False
        self.intermediate_outputs = {}

        # Called with (candidate_embeddings, history_embeddings) when set (see PresentationLogger)
        self.news_vector_hook = None

        # L2 term of the last training forward pass (read by the Lightning module)
        self.last_l2: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _init_language_model(self):
        """Load the title encoder; freeze it or fine-tune (the last N layers), as NRMS."""
        if self.config.use_pretrained_lm:
            print(f"Loading pretrained model: {self.config.model_name}")
            self.lm = AutoModel.from_pretrained(self.config.model_name)
        else:
            print(f"Initializing model from scratch: {self.config.model_name}")
            self.lm = AutoModel.from_config(AutoConfig.from_pretrained(self.config.model_name))
        self.hidden_size = self.lm.config.hidden_size

        if not self.config.fine_tune_lm:
            print("Freezing language model parameters")
            for param in self.lm.parameters():
                param.requires_grad = False
        elif getattr(self.config, "finetune_layers", 0) > 0:
            layers = self.lm.encoder.layer
            keep = self.config.finetune_layers
            print(f"Fine-tuning last {keep} layers out of {len(layers)}")
            for param in self.lm.parameters():
                param.requires_grad = False
            for layer in layers[len(layers) - keep :]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            print("Fine-tuning all language model layers")
            for param in self.lm.parameters():
                param.requires_grad = True

    @property
    def lm_trainable(self) -> bool:
        return any(p.requires_grad for p in self.lm.parameters())

    def set_training_data(
        self, edge_index: torch.Tensor, title_input_ids: torch.Tensor, title_attention_mask: torch.Tensor
    ) -> None:
        """Set the interaction graph [2, E] (user index, article index) and the tokenized titles."""
        if title_input_ids.size(0) != self.num_items:
            raise ValueError(
                f"{title_input_ids.size(0)} tokenized titles for {self.num_items} articles"
            )
        if edge_index.numel() and (
            edge_index[0].max() >= self.num_users or edge_index[1].max() >= self.num_items
        ):
            raise ValueError("Edge index out of range of num_users / num_items")
        device = self.content_proj.weight.device
        self.edge_index = edge_index.long().to(device)
        self.item_title_input_ids = title_input_ids.long().to(device)
        self.item_title_attention_mask = title_attention_mask.long().to(device)
        self._norm_adj = None
        self.invalidate_content_bank()

    def invalidate_content_bank(self) -> None:
        """Recompute the PLM title vectors of all articles before the next forward pass."""
        self._bank_valid = False

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The graph buffers' sizes depend on the training file: take them from the checkpoint
        for name in self._GRAPH_BUFFERS:
            key = prefix + name
            if key in state_dict:
                current = self._buffers[name]
                self._buffers[name] = torch.empty_like(state_dict[key], device=current.device)
        self._norm_adj = None
        self._bank_valid = False
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ------------------------------------------------------------------ #
    # Analysis mode (same interface as the other recommenders)
    # ------------------------------------------------------------------ #
    def enable_analysis_mode(self):
        self.store_intermediate_outputs = True

    def disable_analysis_mode(self):
        self.store_intermediate_outputs = False

    # ------------------------------------------------------------------ #
    # Layer 0
    # ------------------------------------------------------------------ #
    def _encode_titles(self, idx: torch.Tensor) -> torch.Tensor:
        """[CLS] vectors of the titles of articles idx: [len(idx), hidden]."""
        out = self.lm(
            input_ids=self.item_title_input_ids[idx],
            attention_mask=self.item_title_attention_mask[idx],
        )
        return out[0][:, 0].float()

    def _refresh_content_bank(self, chunk_size: int = 256) -> None:
        """PLM title vectors of every article, without gradients and without dropout."""
        if self.item_title_input_ids.size(0) != self.num_items:
            raise RuntimeError(
                "LightGCN has no graph: call set_training_data() before training "
                "(train_model does it), or load a trained checkpoint"
            )
        device = self.content_proj.weight.device
        was_training = self.lm.training
        self.lm.eval()
        chunks = []
        # Not an inference tensor even when first built in a validation / eval
        # pass (inference_mode): training steps save it for backward.
        with torch.inference_mode(False), torch.no_grad():
            for idx in torch.arange(self.num_items, device=device).split(chunk_size):
                chunks.append(self._encode_titles(idx))
        self.lm.train(was_training)
        self.content_bank = torch.cat(chunks)
        self._bank_valid = True

    def _item_layer0(self, batch_item_idx: torch.Tensor) -> torch.Tensor:
        """Layer-0 embeddings of all articles [M, d]."""
        if not self._bank_valid or self.content_bank.device != self.content_proj.weight.device:
            self._refresh_content_bank()
        content = self.content_bank

        if self.training and self.lm_trainable and torch.is_grad_enabled():
            # Re-encode the batch's articles with gradients; the rest come from the bank
            idx = batch_item_idx[batch_item_idx >= 0].unique()
            fresh = self._encode_titles(idx)
            content = content.index_put((idx,), fresh)
            self.content_bank = self.content_bank.index_put((idx,), fresh.detach())

        if self.config.center_content:
            # PLM [CLS] vectors share a large common direction (anisotropy): remove
            # the mean over all articles so dot products reflect what differs.
            content = content - self.content_bank.mean(dim=0)

        item0 = self.content_proj(content)
        if self.item_id_embedding is not None:
            item0 = item0 + self.item_id_embedding.weight
        return item0

    # ------------------------------------------------------------------ #
    # Propagation
    # ------------------------------------------------------------------ #
    def _normalized_adjacency(self, device):
        """(src, dst, weight) of D^-1/2 A D^-1/2 over [users; articles], symmetric.

        Nodes without edges get a self-loop of weight 1 so they keep their
        layer-0 embedding at every layer.
        """
        if self._norm_adj is not None and self._norm_adj[0].device == device:
            return self._norm_adj
        with torch.inference_mode(False):  # reused by training steps (see _refresh_content_bank)
            self._norm_adj = self._build_normalized_adjacency(device)
        return self._norm_adj

    def _build_normalized_adjacency(self, device):
        num_nodes = self.num_users + self.num_items
        users = self.edge_index[0].to(device)
        items = self.edge_index[1].to(device) + self.num_users
        src = torch.cat([users, items])
        dst = torch.cat([items, users])

        degree = torch.zeros(num_nodes, device=device).index_put(
            (src,), torch.ones_like(src, dtype=torch.float), accumulate=True
        )
        inv_sqrt = degree.clamp(min=1).pow(-0.5)
        weight = inv_sqrt[src] * inv_sqrt[dst]

        isolated = (degree == 0).nonzero(as_tuple=True)[0]
        src = torch.cat([src, isolated])
        dst = torch.cat([dst, isolated])
        weight = torch.cat([weight, torch.ones(isolated.numel(), device=device)])
        return src, dst, weight

    def _propagate(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Mean of E^(0..K), E^(k+1) = A_hat E^(k).

        Uses gather + index_put(accumulate=True), which is deterministic on GPU
        (the trainer runs with deterministic=True).
        """
        src, dst, weight = self._normalized_adjacency(embeddings.device)
        weight = weight.to(embeddings.dtype).unsqueeze(1)
        layer = embeddings
        total = embeddings
        for _ in range(self.num_layers):
            layer = torch.zeros_like(layer).index_put((dst,), layer[src] * weight, accumulate=True)
            total = total + layer
        return total / (self.num_layers + 1)

    def compute_embeddings(self, batch_item_idx: Optional[torch.Tensor] = None):
        """Final (user [U, d], article [M, d]) embeddings and their layer-0 versions."""
        if batch_item_idx is None:
            batch_item_idx = torch.zeros(0, dtype=torch.long, device=self.content_proj.weight.device)
        item0 = self._item_layer0(batch_item_idx)
        user0 = self.user_embedding.weight
        final = self._propagate(torch.cat([user0, item0]))
        return final[: self.num_users], final[self.num_users :], user0, item0

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gather(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """table[idx] with zero vectors for idx < 0 (padding / unknown article)."""
        valid = idx >= 0
        out = table[idx.clamp(min=0)]
        return out * valid.unsqueeze(-1).to(out.dtype)

    def forward(self, batch):
        """
        Args:
            batch: Dictionary with keys
                - candidate_item_idx: (batch, num_candidates) article indices, -1 = padding
                - history_item_idx: (batch, history_len) article indices, -1 = padding
                - user_ids: (batch,) user indices (0 = unknown user)

        Returns:
            scores: (batch, num_candidates)
        """
        for key in ("candidate_item_idx", "history_item_idx", "user_ids"):
            if key not in batch:
                raise KeyError(f"LightGCN needs batch['{key}']")
        candidate_idx = batch["candidate_item_idx"].long()
        history_idx = batch["history_item_idx"].long()
        user_idx = batch["user_ids"].long().view(-1)

        if max(candidate_idx.max().item(), history_idx.max().item()) >= self.num_items:
            raise ValueError(
                f"Article index >= {self.num_items}: news_items.csv differs from the one "
                "the model was trained with"
            )
        user_idx = torch.where(user_idx < self.num_users, user_idx, torch.zeros_like(user_idx))

        batch_items = torch.cat([candidate_idx.flatten(), history_idx.flatten()])
        user_final, item_final, user0, item0 = self.compute_embeddings(batch_items)

        candidate_embeddings = self._gather(item_final, candidate_idx)  # (B, C, d)
        history_embeddings = self._gather(item_final, history_idx)  # (B, H, d)

        if self.news_vector_hook is not None:
            self.news_vector_hook(candidate_embeddings, history_embeddings)

        user_embedding = self._user_vector(user_final, user_idx, history_embeddings, history_idx)

        scores = torch.bmm(candidate_embeddings, user_embedding.unsqueeze(-1)).squeeze(-1)

        if self.training:
            # Paper's L2 on the layer-0 embeddings of the batch's nodes
            self.last_l2 = 0.5 * (
                user0[user_idx].pow(2).sum(-1)
                + self._gather(item0, candidate_idx).pow(2).sum(dim=(-1, -2))
            ).mean()

        if self.store_intermediate_outputs:
            self.intermediate_outputs["candidate_embeddings"] = candidate_embeddings.detach().cpu()
            self.intermediate_outputs["history_embeddings"] = history_embeddings.detach().cpu()
            self.intermediate_outputs["user_embedding"] = user_embedding.detach().cpu()
            self.intermediate_outputs["scores"] = scores.detach().cpu()

        return scores

    def _user_vector(self, user_final, user_idx, history_embeddings, history_idx):
        """User vector for scoring, per config.user_mode (unknown users -> history mean)."""
        mask = (history_idx >= 0).unsqueeze(-1).to(history_embeddings.dtype)
        history_mean = (history_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1)
        if self.config.user_mode == "history":
            return history_mean

        known = (user_idx > 0).unsqueeze(-1)
        graph_user = user_final[user_idx]
        if self.config.user_mode == "graph+history":
            return torch.where(known, graph_user, torch.zeros_like(graph_user)) + history_mean
        return torch.where(known, graph_user, history_mean)  # "graph"
