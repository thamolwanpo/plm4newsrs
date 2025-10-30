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
    use_glove = "candidate_titles" in first_item

    if use_glove:
        return _collate_glove(batch)
    else:
        return _collate_transformer(batch)


def _collate_transformer(batch):
    """Collate for transformer models."""
    # Stack all tensors
    candidate_input_ids = torch.stack([item["candidate_input_ids"] for item in batch])
    candidate_attention_mask = torch.stack([item["candidate_attention_mask"] for item in batch])
    history_input_ids = torch.stack([item["history_input_ids"] for item in batch])
    history_attention_mask = torch.stack([item["history_attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    # Check for "all negative" cases (label = -1)
    has_no_positive = labels == -1

    if has_no_positive.any():
        # For samples with no positive, assign random target
        # This makes the model learn uniform distribution over candidates
        for i in range(len(labels)):
            if labels[i] == -1:
                # Random candidate index
                num_candidates = candidate_input_ids.shape[1]
                labels[i] = torch.randint(0, num_candidates, (1,)).item()

    return {
        "candidate_input_ids": candidate_input_ids,
        "candidate_attention_mask": candidate_attention_mask,
        "history_input_ids": history_input_ids,
        "history_attention_mask": history_attention_mask,
        "label": labels,
    }


def _collate_glove(batch):
    """Collate for GloVe models."""
    candidate_titles = [item["candidate_titles"] for item in batch]
    history_titles = [item["history_titles"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    # Handle "all negative" cases
    has_no_positive = labels == -1

    if has_no_positive.any():
        for i in range(len(labels)):
            if labels[i] == -1:
                num_candidates = len(candidate_titles[i])
                labels[i] = torch.randint(0, num_candidates, (1,)).item()

    # Create device indicator for GloVe
    device_indicator = torch.zeros(1, dtype=torch.float32)

    return {
        "candidate_titles": candidate_titles,
        "history_titles": history_titles,
        "label": labels,
        "device_indicator": device_indicator,
    }
