"""Output cache for a frozen (not fine-tuned) language model."""

from typing import Dict

import torch
import torch.nn as nn


class FrozenLMCache:
    """Memoizes a frozen LM's outputs per distinct (input_ids, attention_mask) row.

    With fine_tune_lm=False the LM never changes, but the same few hundred
    articles are encoded again in every batch (each group has 5 candidates +
    5 history articles). The cache runs the LM once per distinct article and
    reuses the result. The LM is run in eval mode (no dropout) so outputs are
    deterministic. It is a plain attribute, not a module: nothing is saved in
    checkpoints.
    """

    def __init__(
        self,
        cls_only: bool = True,
        aggregation: str = "cls",
        chunk_size: int = 256,
    ):
        """
        Args:
            cls_only: cache a single pooled vector per article (batch, hidden)
                instead of the full last hidden state (batch, seq_len, hidden)
            aggregation: pooling strategy when caching pooled vectors:
                'cls' (CLS token), 'mean' (masked token mean),
                'last4_mean' (average of last 4 layers, masked token mean)
            chunk_size: max sequences per LM call when filling the cache
        """
        self.cls_only = cls_only
        self.aggregation = aggregation
        self.chunk_size = chunk_size
        self.store: Dict[bytes, torch.Tensor] = {}

    def _pool_chunk(
        self, lm: nn.Module, chunk_ids: torch.Tensor, chunk_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.aggregation == "last4_mean":
            out = lm(input_ids=chunk_ids, attention_mask=chunk_mask, output_hidden_states=True)
            last_4 = torch.stack(out.hidden_states[-4:]).mean(dim=0)
            mask = chunk_mask.unsqueeze(-1).float()
            sum_emb = (last_4 * mask).sum(dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return (sum_emb / sum_mask).float()

        hidden = lm(input_ids=chunk_ids, attention_mask=chunk_mask)[0]
        if not self.cls_only and self.aggregation == "none":
            return hidden.half()
        elif self.aggregation == "mean":
            mask = chunk_mask.unsqueeze(-1).float()
            sum_emb = (hidden * mask).sum(dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return (sum_emb / sum_mask).float()
        else:  # 'cls' or fallback
            return hidden[:, 0].float()

    def __call__(
        self, lm: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        rows = torch.cat([input_ids, attention_mask], dim=1).cpu().numpy()
        keys = [row.tobytes() for row in rows]

        missing = {}
        for i, key in enumerate(keys):
            if key not in self.store and key not in missing:
                missing[key] = i

        if missing:
            idx = torch.tensor(list(missing.values()), device=input_ids.device)
            was_training = lm.training
            lm.eval()
            outputs = []
            with torch.no_grad():
                for chunk in idx.split(self.chunk_size):
                    outputs.append(
                        self._pool_chunk(lm, input_ids[chunk], attention_mask[chunk])
                    )
            lm.train(was_training)
            is_single_vector = self.cls_only or self.aggregation in ("cls", "mean", "last4_mean")
            for key, value in zip(missing, torch.cat(outputs)):
                self.store[key] = value.float() if is_single_vector else value

        out = torch.stack([self.store[key] for key in keys])
        return out.float().to(input_ids.device)
