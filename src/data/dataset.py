# src/data/dataset.py

import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List

from configs.models.simple_config import ModelConfig
from .preprocessing import (
    load_news_index,
    load_news_items,
    pad_history,
    read_model_csv,
    to_groups,
)


class BaseNewsDataset(Dataset):
    """Base dataset for training/validation files (``*_model.csv``).

    Each sample is one K+1 group: the positive candidate first, followed by
    exactly ``config.num_negatives`` negatives, and a history left-padded to
    ``config.max_history_length``.

    Every candidate (= one user-candidate interaction) carries a sample id:
    its 0-based row position in the CSV. Negatives reused to fill a group
    get -1, so each interaction is counted once by the training dynamics.

    Every candidate and history slot also carries an article index: the
    article's row position in news_items.csv (-1 = padding / unknown id).
    """

    def __init__(
        self, csv_path: Path, config: ModelConfig, user_map: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            csv_path: Path to a 7-column model CSV
            config: Model configuration
            user_map: user_id -> index map (required for LSTUR)
        """
        self.config = config

        print(f"Loading dataset from: {csv_path}")
        pairwise_df = read_model_csv(csv_path)
        print(f"  Rows: {len(pairwise_df)}")

        groups = to_groups(pairwise_df, with_row_ids=True)
        self.df = self._fix_group_sizes(groups)
        print(f"  Groups: {len(self.df)} (dropped {len(groups) - len(self.df)} without negatives)")

        print(f"Loading news items from: {config.news_items_path}")
        self.news_texts, self.news_titles = load_news_items(config.news_items_path)
        print(f"  Loaded {len(self.news_titles) - 1} news items")
        self.item_ids, self.item_is_fake = load_news_index(config.news_items_path)
        self.item_index = {item_id: i for i, item_id in enumerate(self.item_ids)}

        self.use_lstur = config.architecture == "lstur"
        self.user_to_idx = user_map
        if self.use_lstur and self.user_to_idx is None:
            raise ValueError("LSTUR datasets need a user_map (see build_user_map)")

    def _fix_group_sizes(self, groups):
        """Give every group exactly 1 positive + num_negatives negatives.

        Groups short of negatives (rare: tiny candidate universe) reuse their
        own negatives cyclically; groups without any negative are dropped.
        """
        k = self.config.num_negatives
        fixed = []
        for imps in groups["impressions"]:
            pos = [x for x in imps if x[2] == 1][:1]
            negs = [x for x in imps if x[2] == 0]
            if not pos or not negs:
                fixed.append(None)
                continue
            fixed.append(
                pos
                + [negs[i] if i < len(negs) else (*negs[i % len(negs)][:4], -1) for i in range(k)]
            )
        groups = groups.assign(impressions=fixed)
        return groups[groups["impressions"].notna()].reset_index(drop=True)

    def _get_user_idx(self, user_id: str) -> int:
        """User index for embedding lookup (0 = unknown user / not LSTUR)."""
        if not self.use_lstur:
            return 0
        return self.user_to_idx.get(str(user_id), 0)

    def __len__(self):
        return len(self.df)

    def _get_text_by_id(self, item_id: str) -> str:
        return self.news_texts.get(item_id, self.news_titles.get(item_id, ""))

    def _get_title_by_id(self, item_id: str) -> str:
        return self.news_titles.get(item_id, "")

    def _parse_impressions(self, idx: int) -> tuple:
        """Candidates of one group (positive is always at index 0)."""
        impressions = self.df.iloc[idx]["impressions"]
        candidate_ids = [info[0] for info in impressions]
        candidate_titles = [self._get_title_by_id(i) for i in candidate_ids]
        candidate_texts = [self._get_text_by_id(i) for i in candidate_ids]
        return candidate_texts, candidate_titles

    def _get_sample_ids(self, idx: int) -> torch.Tensor:
        """Sample ids (CSV row positions, -1 = reused negative) of one group."""
        impressions = self.df.iloc[idx]["impressions"]
        return torch.tensor([int(info[4]) for info in impressions], dtype=torch.long)

    def _item_indices(self, item_ids: List[str]) -> torch.Tensor:
        """Article indices (news_items.csv rows) of ids; padding / unknown ids -> -1."""
        return torch.tensor([self.item_index.get(i, -1) for i in item_ids], dtype=torch.long)

    def _get_item_indices(self, idx: int) -> Dict[str, torch.Tensor]:
        """Article indices of the candidates [K+1] and of the padded history [H] of one group."""
        row = self.df.iloc[idx]
        history_ids = pad_history(row["history_ids"], self.config.max_history_length)
        return {
            "candidate_item_idx": self._item_indices([info[0] for info in row["impressions"]]),
            "history_item_idx": self._item_indices(history_ids),
        }

    def _parse_history(self, idx: int) -> tuple:
        """History texts and titles, left-padded to max_history_length."""
        history_ids = pad_history(
            self.df.iloc[idx]["history_ids"], self.config.max_history_length
        )
        history_texts = [self._get_text_by_id(i) for i in history_ids]
        history_titles = [self._get_title_by_id(i) for i in history_ids]
        return history_texts, history_titles


class NewsDataset(BaseNewsDataset):
    """
    Dataset for news recommendation that handles both GloVe and Transformer models.

    For GloVe: Returns raw text
    For Transformers: Returns tokenized inputs
    """

    def __init__(
        self,
        csv_path: Path,
        config: ModelConfig,
        tokenizer: Optional[Any] = None,
        user_map: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            csv_path: Path to a 7-column model CSV
            config: Model configuration
            tokenizer: HuggingFace tokenizer (None for GloVe)
            user_map: user_id -> index map (required for LSTUR)
        """
        super().__init__(csv_path, config, user_map=user_map)

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
            Dict with sample_ids (one id per candidate), candidate_item_idx [K+1] and
            history_item_idx [H] (article indices, -1 = padding) plus keys depending on model type:
            - GloVe: candidate_texts, candidate_titles, history_texts, history_titles, label, user_ids
            - Transformer: candidate_input_ids, candidate_attention_mask,
                          candidate_title_input_ids, candidate_title_attention_mask,
                          history_input_ids, history_attention_mask,
                          history_title_input_ids, history_title_attention_mask, label, user_ids
        """
        candidate_texts, candidate_titles = self._parse_impressions(idx)
        history_texts, history_titles = self._parse_history(idx)

        user_idx = self._get_user_idx(self.df.iloc[idx]["user_id"])
        label = 0  # the positive is always the first candidate

        if self.use_glove:
            item = self._get_glove_item(
                candidate_texts, candidate_titles, history_texts, history_titles, label, user_idx
            )
        else:
            item = self._get_transformer_item(
                candidate_texts, candidate_titles, history_texts, history_titles, label, user_idx
            )
        item["sample_ids"] = self._get_sample_ids(idx)
        item.update(self._get_item_indices(idx))
        return item

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
            "user_ids": torch.tensor(user_idx, dtype=torch.long),
        }

    def _tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

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
        candidate_text_inputs = self._tokenize(candidate_texts)
        candidate_title_inputs = self._tokenize(candidate_titles)
        history_text_inputs = self._tokenize(history_texts)
        history_title_inputs = self._tokenize(history_titles)

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
            "user_ids": torch.tensor(user_idx, dtype=torch.long),
        }
