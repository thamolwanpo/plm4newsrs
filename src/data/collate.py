import torch


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Check if using GloVe (text-based) or transformers (token-based)
    if "candidate_titles" in batch[0]:
        # GloVe mode: return raw text + device indicator
        return {
            "candidate_titles": [item["candidate_titles"] for item in batch],
            "history_titles": [item["history_titles"] for item in batch],
            "label": torch.stack([item["label"] for item in batch]),
            "device_indicator": torch.zeros(1),  # Will be moved to correct device by Lightning
        }
    else:
        # Transformer mode: return tokens
        return {
            "candidate_input_ids": torch.stack([item["candidate_input_ids"] for item in batch]),
            "candidate_attention_mask": torch.stack(
                [item["candidate_attention_mask"] for item in batch]
            ),
            "history_input_ids": torch.stack([item["history_input_ids"] for item in batch]),
            "history_attention_mask": torch.stack(
                [item["history_attention_mask"] for item in batch]
            ),
            "label": torch.stack([item["label"] for item in batch]),
        }
