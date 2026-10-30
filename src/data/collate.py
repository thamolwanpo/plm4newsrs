# src/data/collate.py

import torch


def collate_fn(batch):
    """Collate function for news recommendation."""
    if not batch or all(item is None for item in batch):
        return None

    # Filter out None samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Check if this is GloVe or Transformer mode
    first_item = batch[0]
    use_glove = "candidate_texts" in first_item

    if use_glove:
        return _collate_glove(batch)
    else:
        return _collate_transformer(batch)


def _collate_transformer(batch):
    """Collate for transformer models."""
    # Stack all tensors
    candidate_input_ids = torch.stack([item["candidate_input_ids"] for item in batch])
    candidate_attention_mask = torch.stack([item["candidate_attention_mask"] for item in batch])
    candidate_title_input_ids = torch.stack([item["candidate_title_input_ids"] for item in batch])
    candidate_title_attention_mask = torch.stack(
        [item["candidate_title_attention_mask"] for item in batch]
    )
    history_input_ids = torch.stack([item["history_input_ids"] for item in batch])
    history_attention_mask = torch.stack([item["history_attention_mask"] for item in batch])
    history_title_input_ids = torch.stack([item["history_title_input_ids"] for item in batch])
    history_title_attention_mask = torch.stack(
        [item["history_title_attention_mask"] for item in batch]
    )
    labels = torch.stack([item["label"] for item in batch])

    # Stack user IDs if present
    user_ids = None
    if "user_ids" in batch[0]:
        user_ids = torch.stack([item["user_ids"] for item in batch])

    result = {
        "candidate_input_ids": candidate_input_ids,
        "candidate_attention_mask": candidate_attention_mask,
        "candidate_title_input_ids": candidate_title_input_ids,
        "candidate_title_attention_mask": candidate_title_attention_mask,
        "history_input_ids": history_input_ids,
        "history_attention_mask": history_attention_mask,
        "history_title_input_ids": history_title_input_ids,
        "history_title_attention_mask": history_title_attention_mask,
        "label": labels,
    }

    # Add user_ids if present
    if user_ids is not None:
        result["user_ids"] = user_ids

    # Per-candidate interaction ids [batch, K+1] (used by the training dynamics)
    if "sample_ids" in batch[0]:
        result["sample_ids"] = torch.stack([item["sample_ids"] for item in batch])

    # Article indices [batch, K+1] and [batch, H] (used by the presentation log)
    for key in ("candidate_item_idx", "history_item_idx"):
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])

    return result


def _collate_glove(batch):
    """Collate for GloVe models."""
    candidate_texts = [item["candidate_texts"] for item in batch]
    candidate_titles = [item["candidate_titles"] for item in batch]
    history_texts = [item["history_texts"] for item in batch]
    history_titles = [item["history_titles"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    # Stack user IDs if present
    user_ids = None
    if "user_ids" in batch[0]:
        user_ids = torch.stack([item["user_ids"] for item in batch])

    # Create device indicator for GloVe
    device_indicator = torch.zeros(1, dtype=torch.float32)

    result = {
        "candidate_texts": candidate_texts,
        "candidate_titles": candidate_titles,
        "history_texts": history_texts,
        "history_titles": history_titles,
        "label": labels,
        "device_indicator": device_indicator,
    }

    # Add user_ids if present
    if user_ids is not None:
        result["user_ids"] = user_ids

    # Per-candidate interaction ids [batch, K+1] (used by the training dynamics)
    if "sample_ids" in batch[0]:
        result["sample_ids"] = torch.stack([item["sample_ids"] for item in batch])

    # Article indices [batch, K+1] and [batch, H] (used by the presentation log)
    for key in ("candidate_item_idx", "history_item_idx"):
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])

    return result
