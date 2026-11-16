# src/evaluation/benchmark_dataset.py

from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from configs.models.simple_config import ModelConfig
from src.data.preprocessing import (
    PAD_ID,
    load_news_index,
    load_news_items,
    needs_user_map,
    pad_history,
    read_model_csv,
    to_groups,
)


class BenchmarkDataset(Dataset):
    """Dataset for one stage6 benchmark file.

    Each sample is one query: the target (label=1) followed by all of its
    negatives. The number of negatives may differ between queries (99 by
    default, or the whole eligible pool with NUM_NEGATIVES_BENCHMARK="full"),
    so candidates are padded per batch in ``benchmark_collate_fn``.
    """

    def __init__(
        self,
        csv_path: Path,
        tokenizer,
        config: ModelConfig,
        user_map: Optional[Dict[str, int]] = None,
        target_filter: Optional[Set[str]] = None,
        news_items: Optional[tuple] = None,
    ):
        """
        Args:
            csv_path: Path to benchmark CSV file
            tokenizer: HuggingFace tokenizer (None for GloVe)
            config: Model configuration
            user_map: user_id -> index map (required for LSTUR and LightGCN)
            target_filter: keep only queries whose target id is in this set
            news_items: preloaded (texts, titles) dicts from load_news_items
        """
        print(f"Loading benchmark from: {csv_path}")
        pairwise_df = read_model_csv(csv_path)
        self.df = to_groups(pairwise_df, target_filter=target_filter)
        print(f"  Rows: {len(pairwise_df)} -> queries: {len(self.df)}")

        self.tokenizer = tokenizer
        self.config = config
        self.use_glove = "glove" in config.model_name.lower()

        self.use_user_map = needs_user_map(config.architecture)
        self.user_to_idx = user_map
        if self.use_user_map and self.user_to_idx is None:
            raise ValueError(
                f"{config.architecture} evaluation needs the user map saved at training time"
            )

        if news_items is None:
            news_items = load_news_items(config.news_items_path)
        self.news_text_dict, self.news_title_dict = news_items

        # Article index (row of news_items.csv, -1 = padding / unknown) for id-based models
        item_ids, _ = load_news_index(config.news_items_path)
        self.item_index = {item_id: i for i, item_id in enumerate(item_ids)}

        self.processed_data = [
            {
                "query_id": qid,
                "user_id": row.user_id,
                "candidate_ids": [info[0] for info in row.impressions],
                "history_ids": pad_history(row.history_ids, config.max_history_length),
                "impressions": row.impressions,
            }
            for qid, row in enumerate(self.df.itertuples(index=False))
        ]

        if not self.use_glove and self.tokenizer is not None:
            print("  🚀 Pre-tokenizing all items...")
            self._pretokenize_all_items()
        else:
            self.item_encodings = None

    def _pretokenize_all_items(self):
        """Pre-tokenize all unique news items (and the padding item)."""
        unique_ids = {PAD_ID}
        for data in self.processed_data:
            unique_ids.update(data["candidate_ids"])
            unique_ids.update(data["history_ids"])
        unique_ids = sorted(unique_ids)

        self.item_encodings = {}
        batch_size = 512
        for i in tqdm(range(0, len(unique_ids), batch_size), desc="    Tokenizing batches"):
            batch_ids = unique_ids[i : i + batch_size]
            text_encoded = self._tokenize([self._get_text_by_id(x) for x in batch_ids])
            title_encoded = self._tokenize([self._get_title_by_id(x) for x in batch_ids])

            for j, item_id in enumerate(batch_ids):
                self.item_encodings[item_id] = {
                    "text_input_ids": text_encoded["input_ids"][j],
                    "text_attention_mask": text_encoded["attention_mask"][j],
                    "title_input_ids": title_encoded["input_ids"][j],
                    "title_attention_mask": title_encoded["attention_mask"][j],
                }

    def _tokenize(self, texts):
        return self.tokenizer(
            texts,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def _get_user_idx(self, user_id: Any) -> int:
        """Mapped user index, or 0 (unknown) for users not seen in training."""
        if not self.use_user_map:
            return 0
        return self.user_to_idx.get(str(user_id), 0)

    def __len__(self):
        return len(self.processed_data)

    def _get_text_by_id(self, item_id: str) -> str:
        return self.news_text_dict.get(item_id, self.news_title_dict.get(item_id, ""))

    def _get_title_by_id(self, item_id: str) -> str:
        return self.news_title_dict.get(item_id, "")

    def _item_indices(self, ids):
        return torch.tensor([self.item_index.get(i, -1) for i in ids], dtype=torch.long)

    def _stack(self, ids, key):
        return torch.stack([self.item_encodings[i][key] for i in ids])

    def __getitem__(self, idx):
        data = self.processed_data[idx]
        candidate_ids = data["candidate_ids"]
        history_ids = data["history_ids"]

        item = {
            "query_id": data["query_id"],
            "user_id": data["user_id"],
            "user_ids": torch.tensor(self._get_user_idx(data["user_id"]), dtype=torch.long),
            "impression_data": data["impressions"],
            "candidate_item_idx": self._item_indices(candidate_ids),
            "history_item_idx": self._item_indices(history_ids),
        }

        if self.use_glove:
            item.update(
                {
                    "candidate_texts": [self._get_text_by_id(i) for i in candidate_ids],
                    "candidate_titles": [self._get_title_by_id(i) for i in candidate_ids],
                    "history_texts": [self._get_text_by_id(i) for i in history_ids],
                    "history_titles": [self._get_title_by_id(i) for i in history_ids],
                }
            )
        else:
            if self.item_encodings is None:
                raise ValueError("Tokenizer required but pre-tokenization not performed")
            pad = self.item_encodings[PAD_ID]
            item.update(
                {
                    "candidate_input_ids": self._stack(candidate_ids, "text_input_ids"),
                    "candidate_attention_mask": self._stack(candidate_ids, "text_attention_mask"),
                    "candidate_title_input_ids": self._stack(candidate_ids, "title_input_ids"),
                    "candidate_title_attention_mask": self._stack(
                        candidate_ids, "title_attention_mask"
                    ),
                    "history_input_ids": self._stack(history_ids, "text_input_ids"),
                    "history_attention_mask": self._stack(history_ids, "text_attention_mask"),
                    "history_title_input_ids": self._stack(history_ids, "title_input_ids"),
                    "history_title_attention_mask": self._stack(
                        history_ids, "title_attention_mask"
                    ),
                    "pad_encoding": pad,
                }
            )
        return item


_PAD_KEYS = {
    "candidate_input_ids": "text_input_ids",
    "candidate_attention_mask": "text_attention_mask",
    "candidate_title_input_ids": "title_input_ids",
    "candidate_title_attention_mask": "title_attention_mask",
}


def benchmark_collate_fn(batch):
    """Collate queries, padding candidates to the longest query in the batch.

    ``num_candidates`` holds each query's real candidate count; scores of
    padded slots must be ignored (the evaluator masks them).
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    num_candidates = [len(item["impression_data"]) for item in batch]
    max_c = max(num_candidates)

    result = {
        "query_id": [item["query_id"] for item in batch],
        "user_id": [item["user_id"] for item in batch],
        "impression_data": [item["impression_data"] for item in batch],
        "num_candidates": torch.tensor(num_candidates, dtype=torch.long),
        "user_ids": torch.stack([item["user_ids"] for item in batch]),
        # Article indices; padded candidate slots get -1
        "candidate_item_idx": torch.stack(
            [
                F.pad(item["candidate_item_idx"], (0, max_c - n), value=-1)
                for item, n in zip(batch, num_candidates)
            ]
        ),
        "history_item_idx": torch.stack([item["history_item_idx"] for item in batch]),
    }

    if "candidate_texts" in batch[0]:
        # GloVe mode: pad candidate lists with empty strings
        for key in ("candidate_texts", "candidate_titles"):
            result[key] = [item[key] + [""] * (max_c - len(item[key])) for item in batch]
        for key in ("history_texts", "history_titles"):
            result[key] = [item[key] for item in batch]
        result["device_indicator"] = torch.zeros(1)
    else:
        # Transformer mode: pad candidates with the padding item's encoding
        for key, pad_key in _PAD_KEYS.items():
            padded = []
            for item, n in zip(batch, num_candidates):
                t = item[key]
                if n < max_c:
                    pad = item["pad_encoding"][pad_key].unsqueeze(0).expand(max_c - n, -1)
                    t = torch.cat([t, pad], dim=0)
                padded.append(t)
            result[key] = torch.stack(padded)
        for key in (
            "history_input_ids",
            "history_attention_mask",
            "history_title_input_ids",
            "history_title_attention_mask",
        ):
            result[key] = torch.stack([item[key] for item in batch])

    return result
