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

    def __init__(self, cls_only: bool, chunk_size: int = 256):
        """
        Args:
            cls_only: cache only the [CLS] vector (batch, hidden) instead of the
                full last hidden state (batch, seq_len, hidden)
            chunk_size: max sequences per LM call when filling the cache
        """
        self.cls_only = cls_only
        self.chunk_size = chunk_size
        self.store: Dict[bytes, torch.Tensor] = {}

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
                    hidden = lm(input_ids=input_ids[chunk], attention_mask=attention_mask[chunk])[0]
                    outputs.append(hidden[:, 0] if self.cls_only else hidden.half())
            lm.train(was_training)
            for key, value in zip(missing, torch.cat(outputs)):
                self.store[key] = value.float() if self.cls_only else value

        out = torch.stack([self.store[key] for key in keys])
        return out.float().to(input_ids.device)
