# src/models/lightgcn/graph.py
"""User-item interaction graph of a training file, for LightGCN."""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch

from src.data.preprocessing import parse_id_list, read_model_csv


def build_interaction_edges(
    train_csv: Union[str, Path],
    user_map: Dict[str, int],
    item_index: Dict[str, int],
    graph_edges: str = "positives",
) -> torch.Tensor:
    """Unique (user index, article index) clicks of a training CSV, as a [2, E] tensor.

    Args:
        train_csv: 7-column model CSV the model is trained on
        user_map: user_id -> user index (build_user_map; 0 = unknown, never an edge)
        item_index: item_id -> article index (row of news_items.csv)
        graph_edges: "positives" (every label=1 row) or "positives+history"
            (also every history_ids entry of those rows)

    Only training interactions become edges: validation / benchmark rows never do.
    """
    df = read_model_csv(train_csv)
    positives = df[df["label"] == 1]

    pairs = set()
    unknown_items = 0
    for user_id, item_id, history in zip(
        positives["user_id"], positives["candidate_id"], positives["history_ids"]
    ):
        user = user_map.get(str(user_id), 0)
        if user == 0:
            continue
        clicked = [item_id]
        if graph_edges == "positives+history":
            clicked += parse_id_list(history)
        for item in clicked:
            idx = item_index.get(str(item), -1)
            if idx < 0:
                unknown_items += 1
                continue
            pairs.add((user, idx))

    if unknown_items:
        print(f"  ⚠️  {unknown_items} clicks on items missing from news_items.csv (not in the graph)")
    if not pairs:
        raise ValueError(f"No user-item edges found in {train_csv}")
    edges = torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()
    print(
        f"  LightGCN graph: {edges.size(1)} edges, "
        f"{edges[0].unique().numel()} users, {edges[1].unique().numel()} items"
    )
    return edges


def tokenize_titles(
    tokenizer, item_ids: List[str], titles: Dict[str, str], max_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenized title of every article, in article index order: ([M, L], [M, L])."""
    encoded = tokenizer(
        [titles.get(item_id, "") for item_id in item_ids],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].long(), encoded["attention_mask"].long()
