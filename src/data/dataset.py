# src/data/dataset.py

import torch
from pathlib import Path
import pandas as pd
import ast
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List

from configs.models.simple_config import ModelConfig
from .preprocessing import convert_pairwise_to_listwise


class BaseNewsDataset(Dataset):
    """Base dataset class for news recommendation."""

    def __init__(self, csv_path: Path, config: ModelConfig):
        """
        Args:
            csv_path: Path to CSV file (pairwise format)
            config: Model configuration
        """
        print(f"Loading dataset from: {csv_path}")
        pairwise_df = pd.read_csv(csv_path)
        print(f"  Pairwise samples: {len(pairwise_df)}")

        self.df = convert_pairwise_to_listwise(pairwise_df, config)
        print(f"  Listwise samples: {len(self.df)}")

        self.config = config

    def __len__(self):
        return len(self.df)

    def _parse_impressions(self, idx: int) -> tuple:
        """Parse impressions and sort by label."""
        row = self.df.iloc[idx]
        impressions = row["impressions"]

        # Sort so positives come first
        impressions.sort(key=lambda x: x[2], reverse=True)

        # Extract candidate titles and other info
        candidate_titles = [info[1] for info in impressions]
        candidate_ids = [info[0] for info in impressions]
        labels = [info[2] for info in impressions]

        # Check if there are any positives
        has_positive = any(lbl == 1 for lbl in labels)

        if has_positive:
            # Normal case: label points to first positive (position 0)
            label = 0
        else:
            # All negatives case (corrected set): use -1 as sentinel
            # Or use a random candidate (to force uniform distribution)
            label = 1  # Special marker for "no positive"

        return candidate_titles, candidate_ids, labels, label, has_positive

    def _parse_history(self, idx: int) -> List[str]:
        """Parse history titles."""
        row = self.df.iloc[idx]

        try:
            history_list = ast.literal_eval(row["history_titles"])
        except (ValueError, SyntaxError):
            history_list = row["history_titles"]

        if not isinstance(history_list, list):
            history_list = [str(history_list)]

        # Truncate to max history length
        history_list = history_list[: self.config.max_history_length]

        return history_list


class NewsDataset(BaseNewsDataset):
    """
    Dataset for news recommendation that handles both GloVe and Transformer models.

    For GloVe: Returns raw text
    For Transformers: Returns tokenized inputs
    """

    def __init__(self, csv_path: Path, config: ModelConfig, tokenizer: Optional[Any] = None):
        """
        Args:
            csv_path: Path to CSV file (pairwise format)
            config: Model configuration
            tokenizer: HuggingFace tokenizer (None for GloVe)
        """
        super().__init__(csv_path, config)

        self.tokenizer = tokenizer
        self.use_glove = "glove" in config.model_name.lower()

        # Validate tokenizer requirement
        if not self.use_glove and self.tokenizer is None:
            raise ValueError(
                f"Tokenizer is required for model '{config.model_name}'. "
                "Only GloVe models can work without a tokenizer."
            )

        if self.use_glove:
            print("  Using GloVe mode: returning raw text")
        else:
            print(f"  Using Transformer mode: tokenizing with {type(tokenizer).__name__}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dict with different keys depending on model type:
            - GloVe: candidate_titles, history_titles, label
            - Transformer: candidate_input_ids, candidate_attention_mask,
                          history_input_ids, history_attention_mask, label
        """
        candidate_titles, candidate_ids, labels, label, has_positive = self._parse_impressions(idx)
        history_list = self._parse_history(idx)

        if self.use_glove:
            return self._get_glove_item(candidate_titles, history_list, label)
        else:
            return self._get_transformer_item(candidate_titles, history_list, label)

    def _get_glove_item(
        self, candidate_titles: List[str], history_list: List[str], label: int
    ) -> Dict[str, Any]:
        """Get item for GloVe models (raw text)."""
        return {
            "candidate_titles": candidate_titles,  # List of strings
            "history_titles": history_list,  # List of strings
            "label": torch.tensor(label, dtype=torch.long),
        }

    def _get_transformer_item(
        self, candidate_titles: List[str], history_list: List[str], label: int
    ) -> Dict[str, Any]:
        """Get item for Transformer models (tokenized)."""
        # Tokenize candidates
        candidate_inputs = self.tokenizer(
            candidate_titles,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize history
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
            "label": torch.tensor(label, dtype=torch.long),
        }
