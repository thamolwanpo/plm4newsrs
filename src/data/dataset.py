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

        # Load news_items.csv for text lookup
        news_items_path = config.data_dir / "news_items.csv"
        news_items_path = Path(str(news_items_path).replace("/models/", "/"))
        print(f"Loading news items from: {news_items_path}")
        self.news_items = pd.read_csv(news_items_path, index_col="item_id")
        print(f"  Loaded {len(self.news_items)} news items")

        # Build user ID mapping if using LSTUR
        self.use_lstur = config.architecture == "lstur"
        if self.use_lstur:
            self._build_user_mapping()

    def _build_user_mapping(self):
        """Build mapping from user_id to index for embedding lookup."""
        unique_users = self.df["user_id"].unique()
        self.user_to_idx = {user_id: idx + 1 for idx, user_id in enumerate(unique_users)}
        # Reserve index 0 for unknown users
        self.user_to_idx[0] = 0
        print(f"  Built user mapping: {len(unique_users)} unique users")

    def _get_user_idx(self, user_id: str) -> int:
        """Get user index for embedding lookup."""
        if not self.use_lstur:
            return 0

        # Try to convert to appropriate type
        try:
            # Handle different user_id formats
            if isinstance(user_id, str):
                # Try to parse as int if it looks like a number
                try:
                    user_id = int(user_id)
                except ValueError:
                    pass
        except:
            pass

        # Return mapped index, default to 0 (UNK) if not found
        return self.user_to_idx.get(user_id, 0)

    def __len__(self):
        return len(self.df)

    def _get_text_by_id(self, item_id: str) -> str:
        """Lookup text by item_id from news_items.csv."""
        try:
            return self.news_items.loc[item_id, "text"]
        except KeyError:
            # Fallback to title if text not found
            return self.news_items.loc[item_id, "title"]

    def _get_title_by_id(self, item_id: str) -> str:
        """Lookup title by item_id from news_items.csv."""
        return self.news_items.loc[item_id, "title"]

    def _parse_impressions(self, idx: int) -> tuple:
        """Parse impressions and sort by label."""
        row = self.df.iloc[idx]
        impressions = row["impressions"]

        # Sort so positives come first
        impressions.sort(key=lambda x: x[2], reverse=True)

        # Extract candidate texts, titles and other info
        candidate_ids = [info[0] for info in impressions]
        candidate_titles = [info[1] for info in impressions]
        candidate_texts = [self._get_text_by_id(item_id) for item_id in candidate_ids]
        labels = [info[2] for info in impressions]

        # Check if there are any positives
        has_positive = any(lbl == 1 for lbl in labels)

        if has_positive:
            # Normal case: label points to first positive (position 0)
            label = 0
        else:
            # All negatives case (corrected set): use -1 as sentinel
            # Or use a random candidate (to force uniform distribution)
            label = -1  # Special marker for "no positive"

        return candidate_texts, candidate_titles, candidate_ids, labels, label, has_positive

    def _parse_history(self, idx: int) -> tuple:
        """Parse history and return both text and titles."""
        row = self.df.iloc[idx]

        try:
            history_ids = ast.literal_eval(row["history_ids"])
        except (ValueError, SyntaxError):
            history_ids = row["history_ids"]

        if not isinstance(history_ids, list):
            history_ids = [str(history_ids)]

        # Truncate to max history length
        history_ids = history_ids[: self.config.max_history_length]

        # Lookup text and title for each history item
        history_texts = [self._get_text_by_id(item_id) for item_id in history_ids]
        history_titles = [self._get_title_by_id(item_id) for item_id in history_ids]

        return history_texts, history_titles


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
            - GloVe: candidate_texts, candidate_titles, history_texts, history_titles, label, user_ids
            - Transformer: candidate_input_ids, candidate_attention_mask,
                          candidate_title_input_ids, candidate_title_attention_mask,
                          history_input_ids, history_attention_mask,
                          history_title_input_ids, history_title_attention_mask, label, user_ids
        """
        candidate_texts, candidate_titles, candidate_ids, labels, label, has_positive = (
            self._parse_impressions(idx)
        )
        history_texts, history_titles = self._parse_history(idx)

        # Get user ID
        row = self.df.iloc[idx]
        user_id = row["user_id"]
        user_idx = self._get_user_idx(user_id)

        if self.use_glove:
            return self._get_glove_item(
                candidate_texts, candidate_titles, history_texts, history_titles, label, user_idx
            )
        else:
            return self._get_transformer_item(
                candidate_texts, candidate_titles, history_texts, history_titles, label, user_idx
            )

    def _get_glove_item(
        self,
        candidate_texts: List[str],
        candidate_titles: List[str],
        history_texts: List[str],
        history_titles: List[str],
        label: int,
        user_idx: int,
    ) -> Dict[str, Any]:
        """Get item for GloVe models (raw text)."""
        return {
            "candidate_texts": candidate_texts,  # List of strings
            "candidate_titles": candidate_titles,  # List of strings
            "history_texts": history_texts,  # List of strings
            "history_titles": history_titles,  # List of strings
            "label": torch.tensor(label, dtype=torch.long),
            "user_ids": torch.tensor(user_idx, dtype=torch.long),  # Add user ID
        }

    def _get_transformer_item(
        self,
        candidate_texts: List[str],
        candidate_titles: List[str],
        history_texts: List[str],
        history_titles: List[str],
        label: int,
        user_idx: int,
    ) -> Dict[str, Any]:
        """Get item for Transformer models (tokenized)."""
        # Tokenize candidate texts
        candidate_text_inputs = self.tokenizer(
            candidate_texts,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize candidate titles
        candidate_title_inputs = self.tokenizer(
            candidate_titles,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize history texts
        history_text_inputs = self.tokenizer(
            history_texts,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize history titles
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
            "label": torch.tensor(label, dtype=torch.long),
            "user_ids": torch.tensor(user_idx, dtype=torch.long),  # Add user ID
        }
