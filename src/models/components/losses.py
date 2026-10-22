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


def per_group_softmax(scores: torch.Tensor, labels: torch.Tensor):
    """Softmax metrics of every K+1 group, matching the cross-entropy objective.

    Args:
        scores: Candidate logits [batch, K+1]
        labels: Index of the positive candidate [batch]

    Returns:
        loss: Cross-entropy of each group [batch] (reduction='none')
        confidence: Softmax probability of the positive [batch]
        el2n: L2 norm of (softmax - one-hot label) over the group [batch]
    """
    loss = F.cross_entropy(scores, labels, reduction="none")
    probs = torch.softmax(scores, dim=1)
    confidence = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    targets = F.one_hot(labels, num_classes=scores.size(1)).to(scores.dtype)
    el2n = torch.linalg.vector_norm(probs - targets, dim=1)
    return loss, confidence, el2n
