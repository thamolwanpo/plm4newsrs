"""Data preprocessing utilities."""

import pandas as pd
from tqdm import tqdm

from configs.base_config import BaseConfig


def convert_pairwise_to_listwise(df_pairwise: pd.DataFrame, config: BaseConfig) -> pd.DataFrame:
    """Convert pairwise format to listwise format."""
    listwise_rows = []
    groups = df_pairwise.groupby(["user_id", "history_titles"])

    for (user_id, history_titles), group in tqdm(groups, desc="Converting"):
        positive_samples = group[group["label"] == 1]
        negative_samples = group[group["label"] == 0]

        if positive_samples.empty:
            continue

        impressions = []

        # Add positive samples
        for _, row in positive_samples.iterrows():
            impressions.append(
                (row["candidate_id"], row["candidate_title"], 1, row["candidate_is_fake"])
            )

        # Add negative samples
        for _, row in negative_samples.iterrows():
            impressions.append(
                (row["candidate_id"], row["candidate_title"], 0, row["candidate_is_fake"])
            )

        listwise_rows.append(
            {"user_id": user_id, "history_titles": history_titles, "impressions": impressions}
        )

    return pd.DataFrame(listwise_rows)
