"""Data preprocessing utilities for the fakeun-dataprep model/benchmark format.

Every ``*_model.csv`` and ``stage6/*.csv`` file has the 7 columns::

    user_id, candidate_id, candidate_title, candidate_is_fake,
    history_ids, history_titles, label

Rows come in groups: one ``label=1`` row followed by its ``label=0`` rows,
all sharing the same history. A new group starts at every ``label=1`` row.
Groups are identified by position, never by (user_id, history_ids): the same
user/history pair can appear in several groups (e.g. probe benchmarks pair
one user history with many probe articles).
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

# Stands in for missing history slots (left padding) and padded candidates.
PAD_ID = "<pad>"

ID_DTYPES = {"user_id": str, "candidate_id": str, "candidate_title": str}


def to_flag(value) -> int:
    """Parse a boolean-ish CSV value ("True"/"False", 1/0, bool) to 0/1."""
    if isinstance(value, str):
        return int(value.strip().lower() in ("true", "1", "1.0", "yes"))
    return int(bool(value))


def parse_id_list(value) -> List[str]:
    """Parse a history_ids cell ("['a1', 'a2']" or "[]") into a list of str ids."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value]
    if isinstance(parsed, (list, tuple)):
        return [str(v) for v in parsed]
    return [str(parsed)]


def pad_history(history_ids: List[str], max_len: int) -> List[str]:
    """Keep the most recent ``max_len`` clicks and left-pad with PAD_ID.

    Histories are chronological (most recent last), so padding goes in front
    and the most recent click always sits in the last slot.
    """
    history_ids = history_ids[-max_len:] if max_len > 0 else []
    return [PAD_ID] * (max_len - len(history_ids)) + history_ids


def read_model_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Read a 7-column model/benchmark CSV with ids kept as strings."""
    df = pd.read_csv(csv_path, dtype=ID_DTYPES, keep_default_na=False)
    df["label"] = df["label"].astype(int)
    df["candidate_is_fake"] = df["candidate_is_fake"].map(to_flag).astype(int)
    return df


def load_news_items(news_items_path: Union[str, Path]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load news_items.csv into {item_id: text} and {item_id: title} dicts.

    Missing text falls back to the title. PAD_ID maps to empty strings.
    """
    df = pd.read_csv(
        news_items_path,
        usecols=["item_id", "title", "text"],
        dtype={"item_id": str, "title": str, "text": str},
        keep_default_na=False,
    )
    titles = dict(zip(df["item_id"], df["title"]))
    texts = {i: (t if t else titles[i]) for i, t in zip(df["item_id"], df["text"])}
    titles[PAD_ID] = ""
    texts[PAD_ID] = ""
    return texts, titles


def to_groups(
    df_pairwise: pd.DataFrame, target_filter: Optional[Set[str]] = None
) -> pd.DataFrame:
    """Convert the row-per-candidate format into one row per K+1 group.

    Args:
        df_pairwise: 7-column frame (see read_model_csv).
        target_filter: if given, keep only groups whose positive candidate id
            is in this set (e.g. a poisoned set's injected fake items).

    Returns:
        DataFrame with columns user_id, history_ids (list), impressions, where
        impressions is a list of (candidate_id, title, label, is_fake) tuples
        with the positive first.
    """
    df = df_pairwise.reset_index(drop=True)
    group = (df["label"] == 1).cumsum()
    df = df[group > 0]  # rows before the first positive belong to no group
    group = group[group > 0]

    impressions = list(
        zip(df["candidate_id"], df["candidate_title"], df["label"], df["candidate_is_fake"])
    )
    grouped = (
        pd.DataFrame({"g": group.to_numpy(), "imp": impressions})
        .groupby("g", sort=True)["imp"]
        .agg(list)
    )
    first = df.groupby(group.to_numpy(), sort=True).first()

    out = pd.DataFrame(
        {
            "user_id": first["user_id"].to_numpy(),
            "history_ids": [parse_id_list(h) for h in first["history_ids"]],
            "impressions": grouped.to_numpy(),
        }
    )
    if target_filter is not None:
        keep = [imps[0][0] in target_filter for imps in out["impressions"]]
        out = out[keep].reset_index(drop=True)
    return out


def build_user_map(user_ids) -> Dict[str, int]:
    """LSTUR user index: 1..N for known users, 0 is reserved for unknown users."""
    return {u: i + 1 for i, u in enumerate(sorted({str(u) for u in user_ids}))}
