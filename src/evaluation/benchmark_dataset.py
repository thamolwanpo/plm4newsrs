import ast
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict
from tqdm import tqdm
from configs.models.simple_config import ModelConfig


def convert_pairwise_to_listwise_eval(
    df_pairwise: pd.DataFrame, config: ModelConfig
) -> pd.DataFrame:
    """Convert pairwise format to listwise for evaluation."""
    listwise_rows = []
    interaction_groups = df_pairwise.groupby(["user_id", "history_ids"])

    for (user_id, history_ids), group in tqdm(interaction_groups, desc="Converting to listwise"):
        positive_samples = group[group["label"] == 1]
        negative_samples = group[group["label"] == 0]

        if positive_samples.empty:
            continue

        impressions = []

        # Add positive samples first
        for _, pos_row in positive_samples.iterrows():
            impressions.append(
                (
                    pos_row["candidate_id"],
                    pos_row["candidate_title"],
                    1,  # label
                    pos_row["candidate_is_fake"],
                )
            )

        # Add negative samples
        for _, neg_row in negative_samples.iterrows():
            impressions.append(
                (
                    neg_row["candidate_id"],
                    neg_row["candidate_title"],
                    0,  # label
                    neg_row["candidate_is_fake"],
                )
            )

        listwise_rows.append(
            {"user_id": user_id, "history_ids": history_ids, "impressions": impressions}
        )

    return pd.DataFrame(listwise_rows)


class BenchmarkDataset(Dataset):
    """Dataset for benchmark evaluation."""

    def __init__(self, csv_path: Path, tokenizer, config: ModelConfig):
        """
        Args:
            csv_path: Path to benchmark CSV file
            tokenizer: HuggingFace tokenizer (None for GloVe)
            config: Base configuration
        """
        print(f"Loading benchmark from: {csv_path}")
        pairwise_df = pd.read_csv(csv_path)
        print(f"  Pairwise samples: {len(pairwise_df)}")

        self.df = convert_pairwise_to_listwise_eval(pairwise_df, config)
        print(f"  Listwise samples: {len(self.df)}")

        self.tokenizer = tokenizer
        self.config = config
        self.use_glove = "glove" in config.model_name.lower()

        # Load news_items.csv for text lookup
        news_items_path = csv_path.parent.parent / "news_items.csv"
        print(f"Loading news items from: {news_items_path}")
        self.news_items = pd.read_csv(news_items_path, index_col='item_id')
        print(f"  Loaded {len(self.news_items)} news items")

    def __len__(self):
        return len(self.df)

    def _get_text_by_id(self, item_id: str) -> str:
        """Lookup text by item_id from news_items.csv."""
        try:
            return self.news_items.loc[item_id, 'text']
        except KeyError:
            # Fallback to title if text not found
            return self.news_items.loc[item_id, 'title']

    def _get_title_by_id(self, item_id: str) -> str:
        """Lookup title by item_id from news_items.csv."""
        return self.news_items.loc[item_id, 'title']

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        impressions = row["impressions"]

        # Sort so positives come first
        impressions.sort(key=lambda x: x[2], reverse=True)

        # Extract candidate IDs and lookup texts and titles
        candidate_ids = [info[0] for info in impressions]
        candidate_texts = [self._get_text_by_id(item_id) for item_id in candidate_ids]
        candidate_titles = [self._get_title_by_id(item_id) for item_id in candidate_ids]

        # Parse history IDs and lookup texts and titles
        try:
            history_ids = ast.literal_eval(row["history_ids"])
        except (ValueError, SyntaxError):
            history_ids = row["history_ids"]

        if not isinstance(history_ids, list):
            history_ids = [str(history_ids)]

        history_ids = history_ids[: self.config.max_history_length]
        history_texts = [self._get_text_by_id(item_id) for item_id in history_ids]
        history_titles = [self._get_title_by_id(item_id) for item_id in history_ids]

        if self.use_glove:
            # For GloVe: return raw text
            return {
                "candidate_texts": candidate_texts,
                "candidate_titles": candidate_titles,
                "history_texts": history_texts,
                "history_titles": history_titles,
                "user_id": row["user_id"],
                "impression_data": impressions,
            }
        else:
            # For transformers: tokenize
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for transformer models")

            candidate_text_inputs = self.tokenizer(
                candidate_texts,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            candidate_title_inputs = self.tokenizer(
                candidate_titles,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            history_text_inputs = self.tokenizer(
                history_texts,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            history_title_inputs = self.tokenizer(
                history_titles,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "candidate_input_ids": candidate_text_inputs["input_ids"],
                "candidate_attention_mask": candidate_text_inputs["attention_mask"],
                "candidate_title_input_ids": candidate_title_inputs["input_ids"],
                "candidate_title_attention_mask": candidate_title_inputs["attention_mask"],
                "history_input_ids": history_text_inputs["input_ids"],
                "history_attention_mask": history_text_inputs["attention_mask"],
                "history_title_input_ids": history_title_inputs["input_ids"],
                "history_title_attention_mask": history_title_inputs["attention_mask"],
                "user_id": row["user_id"],
                "impression_data": impressions,
            }


def benchmark_collate_fn(batch):
    """Collate function for benchmark evaluation."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Check if using GloVe
    if "candidate_texts" in batch[0]:
        # GloVe mode
        return {
            "candidate_texts": [item["candidate_texts"] for item in batch],
            "candidate_titles": [item["candidate_titles"] for item in batch],
            "history_texts": [item["history_texts"] for item in batch],
            "history_titles": [item["history_titles"] for item in batch],
            "user_id": [item["user_id"] for item in batch],
            "impression_data": [item["impression_data"] for item in batch],
            "device_indicator": torch.zeros(1),  # For device handling
        }
    else:
        # Transformer mode
        return {
            "candidate_input_ids": torch.stack([item["candidate_input_ids"] for item in batch]),
            "candidate_attention_mask": torch.stack(
                [item["candidate_attention_mask"] for item in batch]
            ),
            "candidate_title_input_ids": torch.stack([item["candidate_title_input_ids"] for item in batch]),
            "candidate_title_attention_mask": torch.stack(
                [item["candidate_title_attention_mask"] for item in batch]
            ),
            "history_input_ids": torch.stack([item["history_input_ids"] for item in batch]),
            "history_attention_mask": torch.stack(
                [item["history_attention_mask"] for item in batch]
            ),
            "history_title_input_ids": torch.stack([item["history_title_input_ids"] for item in batch]),
            "history_title_attention_mask": torch.stack(
                [item["history_title_attention_mask"] for item in batch]
            ),
            "user_id": [item["user_id"] for item in batch],
            "impression_data": [item["impression_data"] for item in batch],
        }
