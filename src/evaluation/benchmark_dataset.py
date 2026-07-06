# src/evaluation/benchmark_dataset.py

import ast
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Any
from tqdm import tqdm
from configs.models.simple_config import ModelConfig
import json

def convert_pairwise_to_listwise_eval(
    df_pairwise: pd.DataFrame, config=None
) -> pd.DataFrame:
    df = df_pairwise.copy()

    # 1. Rank pos/neg separately, keep only 1 pos + 99 neg per group
    df["_rank"] = df.groupby(["user_id", "history_ids", "label"]).cumcount()
    df = df[
        ((df["label"] == 1) & (df["_rank"] == 0)) |
        ((df["label"] == 0) & (df["_rank"] < 99))
    ].copy()

    # 2. Sort so pos (label=1) comes first, assign slot 0=pos, 1-99=neg
    df = df.sort_values(["user_id", "history_ids", "label"], ascending=[True, True, False])
    df["_slot"] = df.groupby(["user_id", "history_ids"]).cumcount()

    # 3. Drop groups that don't have all 100 slots
    valid_keys = df.groupby(["user_id", "history_ids"])["_slot"].max()
    valid_keys = valid_keys[valid_keys == 99].index
    df = df.set_index(["user_id", "history_ids"]).loc[valid_keys].reset_index()

    # 4. Pivot to wide: one row per group, columns per slot
    wide = df.pivot_table(
        index=["user_id", "history_ids"],
        columns="_slot",
        values=["candidate_id", "candidate_title", "label", "candidate_is_fake"],
        aggfunc="first",
    )
    wide.columns = [f"{c}_{s}" for c, s in wide.columns]
    wide = wide.reset_index()

    # 5. Pack each row into a list of 100 tuples
    impressions = [
        [
            (
                wide[f"candidate_id_{s}"].iat[i],
                wide[f"candidate_title_{s}"].iat[i],
                wide[f"label_{s}"].iat[i],
                wide[f"candidate_is_fake_{s}"].iat[i],
            )
            for s in range(100)
        ]
        for i in range(len(wide))
    ]

    print(f"✅ Conversion complete. Created {len(wide)} evaluation samples.")
    return pd.DataFrame({
        "user_id":     wide["user_id"].tolist(),
        "history_ids": wide["history_ids"].tolist(),
        "impressions": impressions,
    })

# def convert_pairwise_to_listwise_eval(df_pairwise: pd.DataFrame, config=None) -> pd.DataFrame:
#     """
#     Convert pairwise to listwise for evaluation with a strict 1+99 ratio.
#     Ensures every evaluation row has exactly 100 items (1 pos, 99 negs).
#     """
#     print(" ⚡ Starting strict 1+99 listwise conversion for evaluation...")
#     listwise_rows = []

#     # Group by the unique interaction context
#     groups = df_pairwise.groupby(["user_id", "history_ids"])

#     for (user_id, history_ids), group in tqdm(groups, desc="Evaluating"):
#         positives = group[group["label"] == 1]
#         negatives = group[group["label"] == 0]

#         # In evaluation, we usually only care about the FIRST positive
#         # encountered in this specific context paired with a large set of negatives.
#         if len(positives) >= 1 and len(negatives) >= 99:

#             # Take exactly 1 positive and exactly 99 negatives
#             selected_pos = positives.iloc[:1]
#             selected_neg = negatives.iloc[:99]

#             combined = pd.concat([selected_pos, selected_neg])

#             # Create the tuples list (Candidate_id, Title, Label, Is_fake)
#             # Using itertuples for speed over iterrows
#             impressions = [
#                 (row.candidate_id, row.candidate_title, row.label, row.candidate_is_fake)
#                 for row in combined.itertuples(index=False)
#             ]

#             listwise_rows.append(
#                 {"user_id": user_id, "history_ids": history_ids, "impressions": impressions}
#             )

#     final_df = pd.DataFrame(listwise_rows)
#     print(f" ✅ Conversion complete. Created {len(final_df)} evaluation samples.")
#     return final_df


class BenchmarkDataset(Dataset):
    """Dataset for benchmark evaluation with pre-tokenization."""

    def __init__(
        self,
        csv_path: Path,
        tokenizer,
        config: ModelConfig,
        user_map: Optional[Dict[Any, int]] = None,
    ):
        """
        Args:
            csv_path: Path to benchmark CSV file
            tokenizer: HuggingFace tokenizer (None for GloVe)
            config: Base configuration
            user_map: Optional pre-built user mapping for LSTUR
        """
        print(f"Loading benchmark from: {csv_path}")

        # Optimization: Specify dtypes to speed up loading and reduce memory
        dtype_spec = {
            "user_id": "str",
            "candidate_id": "str",
            "candidate_title": "str",
            "label": "int8",
            "candidate_is_fake": "int8",
        }

        pairwise_df = pd.read_csv(csv_path, dtype=dtype_spec)
        print(f"  Pairwise samples: {len(pairwise_df)}")

        # Use the optimized conversion
        self.df = convert_pairwise_to_listwise_eval(pairwise_df, config)
        print(f"  Listwise samples: {len(self.df)}")

        self.tokenizer = tokenizer
        self.config = config
        self.use_glove = "glove" in config.model_name.lower()

        # Check if this is LSTUR model
        self.use_lstur = config.architecture == "lstur"
        self.user_to_idx = None

        # Load news_items.csv for text lookup
        news_items_path = config.data_dir / "news_items.csv"
        news_items_path = Path(str(news_items_path).replace("/models/", "/"))
        print(f"Loading news items from: {news_items_path}")

        # Only load columns we need
        news_items_df = pd.read_csv(
            news_items_path,
            usecols=["item_id", "text", "title"],
            dtype={"item_id": "str", "text": "str", "title": "str"},
        )

        # Convert to dicts for O(1) lookups
        self.news_text_dict = dict(zip(news_items_df["item_id"], news_items_df["text"]))
        self.news_title_dict = dict(zip(news_items_df["item_id"], news_items_df["title"]))
        print(f"  Loaded {len(self.news_text_dict)} news items")

        # Pre-parse history IDs
        print("  Pre-parsing history IDs...")
        self.df["parsed_history_ids"] = self.df["history_ids"].apply(self._parse_history)
        print("  ✅ History IDs parsed")

        # Pre-process and pre-tokenize if using transformers
        if not self.use_glove and self.tokenizer is not None:
            print("  🚀 Pre-tokenizing all items (this may take a minute)...")
            self._pretokenize_all_items()
            print("  ✅ Pre-tokenization complete")
        else:
            self.item_encodings = None

        # Pre-process data structures
        print("  Pre-processing data structures...")
        self._preprocess_data()
        print("  ✅ Preprocessing complete")

        # Load the user map if LSTUR
        if self.use_lstur:
            if user_map is not None:
                self.user_to_idx = user_map
            else:
                self.user_map_path = self.config.data_dir / "user_to_idx_map.json"
                if self.user_map_path.exists():
                    print(f"  Loading user mapping from: {self.user_map_path}")
                    with open(self.user_map_path, "r") as f:
                        raw_map = json.load(f)
                        self.user_to_idx = {self._convert_key(k): v for k, v in raw_map.items()}
                else:
                    raise FileNotFoundError(
                        f"LSTUR requires a pre-built user map. File not found: {self.user_map_path}"
                    )
            print(f"  Loaded {len(self.user_to_idx)} user entries for LSTUR.")

    def _pretokenize_all_items(self):
        """Pre-tokenize all unique news items for instant lookup."""
        # Collect all unique item IDs that will be used
        unique_ids = set()
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            impressions = row["impressions"]
            unique_ids.update([info[0] for info in impressions])
            unique_ids.update(row["parsed_history_ids"][: self.config.max_history_length])

        unique_ids = list(unique_ids)
        print(f"    Tokenizing {len(unique_ids)} unique items...")

        # Initialize storage
        self.item_encodings = {}

        # Batch tokenize for efficiency
        batch_size = 512
        for i in tqdm(range(0, len(unique_ids), batch_size), desc="    Tokenizing batches"):
            batch_ids = unique_ids[i : i + batch_size]

            # Get texts and titles for this batch
            batch_texts = [
                self.news_text_dict.get(id_, self.news_title_dict.get(id_, "")) for id_ in batch_ids
            ]
            batch_titles = [self.news_title_dict.get(id_, "") for id_ in batch_ids]

            # Tokenize texts
            text_encoded = self.tokenizer(
                batch_texts,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Tokenize titles
            title_encoded = self.tokenizer(
                batch_titles,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Store encodings for each item
            for j, item_id in enumerate(batch_ids):
                self.item_encodings[item_id] = {
                    "text_input_ids": text_encoded["input_ids"][j],
                    "text_attention_mask": text_encoded["attention_mask"][j],
                    "title_input_ids": title_encoded["input_ids"][j],
                    "title_attention_mask": title_encoded["attention_mask"][j],
                }

    def _preprocess_data(self):
        """Pre-process all data structures for faster access."""
        self.processed_data = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            impressions = row["impressions"]

            # Sort impressions by label (descending)
            impressions_sorted = sorted(impressions, key=lambda x: x[2], reverse=True)
            candidate_ids = [info[0] for info in impressions_sorted]

            # Get history IDs (already parsed, just limit length)
            history_ids = row["parsed_history_ids"][: self.config.max_history_length]

            self.processed_data.append(
                {
                    "user_id": row["user_id"],
                    "candidate_ids": candidate_ids,
                    "history_ids": history_ids,
                    "impressions": impressions_sorted,
                }
            )

    def _parse_history(self, history_ids):
        """Parse history once during init to avoid repeated parsing."""
        try:
            parsed = ast.literal_eval(history_ids)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except (ValueError, SyntaxError):
            return [str(history_ids)]

    def _convert_key(self, key):
        """Helper to convert string keys from JSON back to original type (int or str)."""
        try:
            return int(key)
        except ValueError:
            return key

    def _get_user_idx(self, user_id: Any) -> int:
        """
        The Cold-Start Handler.
        Returns mapped index or 0 (UNK) for new users.
        """
        if not self.use_lstur:
            return 0

        # Standardize user_id to string to match JSON keys
        u_key = str(user_id)

        # Return the index if found, else return 0 (Cold Start/UNK)
        return self.user_to_idx.get(u_key, 0)

    def __len__(self):
        return len(self.processed_data)

    def _get_text_by_id(self, item_id: str) -> str:
        """Lookup text by item_id - O(1) dict lookup."""
        return self.news_text_dict.get(item_id, self.news_title_dict.get(item_id, ""))

    def _get_title_by_id(self, item_id: str) -> str:
        """Lookup title by item_id - O(1) dict lookup."""
        return self.news_title_dict.get(item_id, "")

    def __getitem__(self, idx):
        # Get pre-processed data
        data = self.processed_data[idx]
        user_id = data["user_id"]
        candidate_ids = data["candidate_ids"]
        history_ids = data["history_ids"]
        impressions = data["impressions"]

        user_idx = self._get_user_idx(user_id)

        if self.use_glove:
            # For GloVe: lookup texts directly
            candidate_texts = [self._get_text_by_id(id_) for id_ in candidate_ids]
            candidate_titles = [self._get_title_by_id(id_) for id_ in candidate_ids]
            history_texts = [self._get_text_by_id(id_) for id_ in history_ids]
            history_titles = [self._get_title_by_id(id_) for id_ in history_ids]

            return {
                "candidate_texts": candidate_texts,
                "candidate_titles": candidate_titles,
                "history_texts": history_texts,
                "history_titles": history_titles,
                "user_id": user_id,
                "user_ids": torch.tensor(user_idx, dtype=torch.long),
                "impression_data": impressions,
            }
        else:
            # For transformers: use pre-tokenized encodings
            if self.item_encodings is None:
                raise ValueError("Tokenizer required but pre-tokenization not performed")

            # Gather pre-tokenized encodings for candidates
            cand_text_input_ids = []
            cand_text_attention_mask = []
            cand_title_input_ids = []
            cand_title_attention_mask = []

            for cid in candidate_ids:
                enc = self.item_encodings[cid]
                cand_text_input_ids.append(enc["text_input_ids"])
                cand_text_attention_mask.append(enc["text_attention_mask"])
                cand_title_input_ids.append(enc["title_input_ids"])
                cand_title_attention_mask.append(enc["title_attention_mask"])

            # Gather pre-tokenized encodings for history
            hist_text_input_ids = []
            hist_text_attention_mask = []
            hist_title_input_ids = []
            hist_title_attention_mask = []

            for hid in history_ids:
                enc = self.item_encodings[hid]
                hist_text_input_ids.append(enc["text_input_ids"])
                hist_text_attention_mask.append(enc["text_attention_mask"])
                hist_title_input_ids.append(enc["title_input_ids"])
                hist_title_attention_mask.append(enc["title_attention_mask"])

            return {
                "candidate_input_ids": torch.stack(cand_text_input_ids),
                "candidate_attention_mask": torch.stack(cand_text_attention_mask),
                "candidate_title_input_ids": torch.stack(cand_title_input_ids),
                "candidate_title_attention_mask": torch.stack(cand_title_attention_mask),
                "history_input_ids": torch.stack(hist_text_input_ids),
                "history_attention_mask": torch.stack(hist_text_attention_mask),
                "history_title_input_ids": torch.stack(hist_title_input_ids),
                "history_title_attention_mask": torch.stack(hist_title_attention_mask),
                "user_id": user_id,
                "user_ids": torch.tensor(user_idx, dtype=torch.long),
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
        result = {
            "candidate_texts": [item["candidate_texts"] for item in batch],
            "candidate_titles": [item["candidate_titles"] for item in batch],
            "history_texts": [item["history_texts"] for item in batch],
            "history_titles": [item["history_titles"] for item in batch],
            "user_id": [item["user_id"] for item in batch],
            "impression_data": [item["impression_data"] for item in batch],
            "device_indicator": torch.zeros(1),
        }

        if "user_ids" in batch[0]:
            result["user_ids"] = torch.stack([item["user_ids"] for item in batch])

        return result
    else:
        # Transformer mode
        result = {
            "candidate_input_ids": torch.stack([item["candidate_input_ids"] for item in batch]),
            "candidate_attention_mask": torch.stack(
                [item["candidate_attention_mask"] for item in batch]
            ),
            "candidate_title_input_ids": torch.stack(
                [item["candidate_title_input_ids"] for item in batch]
            ),
            "candidate_title_attention_mask": torch.stack(
                [item["candidate_title_attention_mask"] for item in batch]
            ),
            "history_input_ids": torch.stack([item["history_input_ids"] for item in batch]),
            "history_attention_mask": torch.stack(
                [item["history_attention_mask"] for item in batch]
            ),
            "history_title_input_ids": torch.stack(
                [item["history_title_input_ids"] for item in batch]
            ),
            "history_title_attention_mask": torch.stack(
                [item["history_title_attention_mask"] for item in batch]
            ),
            "user_id": [item["user_id"] for item in batch],
            "impression_data": [item["impression_data"] for item in batch],
        }

        if "user_ids" in batch[0]:
            result["user_ids"] = torch.stack([item["user_ids"] for item in batch])

        return result
