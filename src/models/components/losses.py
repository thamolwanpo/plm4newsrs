# src/models/components/losses.py

import torch
import torch.nn.functional as F


def per_interaction_bce(scores: torch.Tensor, labels: torch.Tensor):
    """Binary cross-entropy for every user-candidate interaction of a batch.

    Each candidate score is treated as an independent click logit: the
    candidate at the label index is the positive (1), all others are 0.
    Used for the training dynamics only; the training loss is cross-entropy.

    Args:
        scores: Candidate logits [batch, K+1]
        labels: Index of the positive candidate [batch]

    Returns:
        per_sample_loss: BCE of each interaction [batch, K+1] (reduction='none')
        confidence: Predicted probability of the true class [batch, K+1]
    """
    targets = F.one_hot(labels, num_classes=scores.size(1)).to(scores.dtype)
    per_sample_loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
    probs = torch.sigmoid(scores)
    confidence = torch.where(targets == 1, probs, 1 - probs)
    return per_sample_loss, confidence
