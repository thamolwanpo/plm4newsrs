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
    interaction_groups = df_pairwise.groupby(["user_id", "history_titles"])

    for (user_id, history_titles), group in tqdm(interaction_groups, desc="Converting to listwise"):
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
            {"user_id": user_id, "history_titles": history_titles, "impressions": impressions}
        )

    return pd.DataFrame(listwise_rows)


class BenchmarkDataset(Dataset):
    """Dataset for benchmark evaluation."""

    def __init__(self, csv_path: Path, tokenizer, config: BaseConfig):
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        impressions = row["impressions"]

        # Sort so positives come first
        impressions.sort(key=lambda x: x[2], reverse=True)

        # Extract candidate titles
        candidate_titles = [info[1] for info in impressions]

        # Parse history
        try:
            history_list = ast.literal_eval(row["history_titles"])
        except (ValueError, SyntaxError):
            history_list = row["history_titles"]

        if not isinstance(history_list, list):
            history_list = [str(history_list)]

        history_list = history_list[: self.config.max_history_length]

        if self.use_glove:
            # For GloVe: return raw text
            return {
                "candidate_titles": candidate_titles,
                "history_titles": history_list,
                "user_id": row["user_id"],
                "impression_data": impressions,
            }
        else:
            # For transformers: tokenize
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for transformer models")

            candidate_inputs = self.tokenizer(
                candidate_titles,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            history_inputs = self.tokenizer(
                history_list,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "candidate_input_ids": candidate_inputs["input_ids"],
                "candidate_attention_mask": candidate_inputs["attention_mask"],
                "history_input_ids": history_inputs["input_ids"],
                "history_attention_mask": history_inputs["attention_mask"],
                "user_id": row["user_id"],
                "impression_data": impressions,
            }


def benchmark_collate_fn(batch):
    """Collate function for benchmark evaluation."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Check if using GloVe
    if "candidate_titles" in batch[0]:
        # GloVe mode
        return {
            "candidate_titles": [item["candidate_titles"] for item in batch],
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
            "history_input_ids": torch.stack([item["history_input_ids"] for item in batch]),
            "history_attention_mask": torch.stack(
                [item["history_attention_mask"] for item in batch]
            ),
            "user_id": [item["user_id"] for item in batch],
            "impression_data": [item["impression_data"] for item in batch],
        }
