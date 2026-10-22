#!/usr/bin/env python3
"""
Extract per-group features of a forget set from the model to be unlearned
(stage 1 of UPCORE coreset selection). Standalone: not part of training or
evaluation.

For every K+1 group of a forget_model.csv, the checkpoint encodes the user
history and the positive (fake) candidate and saves:

    user  (N, d)  user embedding
    cand  (N, d)  candidate embedding of the positive
    uc    (N, d)  user * cand  (sums to the click logit; UPCORE's default)

plus the group ids (same numbering as fakeun-dataprep's
coreset_selection.base.add_group_ids) and the time each step took. The .npz
is the --features input of `python -m coreset_selection.run --method upcore`
in fakeun-dataprep.

Examples:
  python scripts/extract_forget_features.py \\
      --config configs/experiments/nrms/nrms_bert_frozen_politifact.yaml \\
      --checkpoint outputs/politifact/.../checkpoints/poisoned_A_r0.5_s1-last.ckpt \\
      --forget ../fakeun-dataprep/output/problem2/politifact/stage4/A_r0.5_s1/\
               flags/flag_050/forget_model.csv
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_config

A_SET = re.compile(r"A_r[0-9.]+_s\d+")


def run_name_of(checkpoint: Path) -> str:
    """'poisoned_A_r0.5_s1-last.ckpt' -> 'poisoned_A_r0.5_s1'."""
    stem = checkpoint.stem
    for suffix in ("-last", "-resumed"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.split("-epoch=")[0]


def check_same_poison_set(checkpoint: Path, forget: Path):
    """The model must be the one trained on the A set the forget file is from."""
    ckpt_set = A_SET.search(checkpoint.name)
    forget_set = A_SET.search(str(forget))
    if ckpt_set and forget_set and ckpt_set.group() != forget_set.group():
        raise ValueError(
            f"checkpoint is for {ckpt_set.group()} but the forget file is from "
            f"{forget_set.group()} (pass --allow-mismatch to run anyway)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Extract forget-set features for UPCORE coreset selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1],
    )
    parser.add_argument(
        "--config", required=True, help="Experiment YAML the model was trained with"
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint of the model to unlearn")
    parser.add_argument("--forget", required=True, help="forget_model.csv")
    parser.add_argument(
        "--data-root",
        help="fakeun-dataprep output folder of the dataset (overrides config; "
        "used for news_items.csv)",
    )
    parser.add_argument(
        "--user-map",
        help="LSTUR user map (default: <checkpoint dir>/<run name>_user_map.json)",
    )
    parser.add_argument(
        "--out",
        help="output .npz (default: <forget dir>/coresets/forget_features__<run name>.npz)",
    )
    parser.add_argument("--batch-size", type=int, help="default: config val_batch_size")
    parser.add_argument("--gpu", type=int, help="GPU device ID (default: auto-detect)")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="allow a checkpoint whose A set differs from the forget file's",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    timing = {}

    checkpoint = Path(args.checkpoint)
    forget_path = Path(args.forget)
    if not args.allow_mismatch:
        check_same_poison_set(checkpoint, forget_path)
    run_name = run_name_of(checkpoint)
    out_path = (
        Path(args.out)
        if args.out
        else forget_path.parent / "coresets" / f"forget_features__{run_name}.npz"
    )

    config = load_config(args.config)
    if args.data_root:
        config.data_root = Path(args.data_root)

    import torch
    from torch.utils.data import DataLoader

    from src.data.collate import collate_fn
    from src.data.dataset import NewsDataset
    from src.data.preprocessing import read_model_csv
    from src.models.registry import get_model_class
    from src.training.trainer import user_map_path

    class ForgetPositiveDataset(NewsDataset):
        """One sample per K+1 group with only the positive candidate: the
        features describe the forget click, and no group is dropped for
        lacking negatives (sample i = i-th group of the file)."""

        def _fix_group_sizes(self, groups):
            groups = groups.assign(
                impressions=[[x for x in imps if x[2] == 1][:1] for imps in groups["impressions"]]
            )
            return groups.reset_index(drop=True)

    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- model ----------------------------------------------------------
    t0 = time.perf_counter()
    lit = get_model_class(config.architecture).load_from_checkpoint(
        str(checkpoint), weights_only=False, map_location=device
    )
    lit.to(device)
    lit.eval()
    recommender = lit.model
    recommender.store_intermediate_outputs = True  # embeddings only, no attention weights
    tokenizer = None
    if "glove" not in config.model_name.lower():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    user_map = None
    if config.architecture == "lstur":
        map_path = (
            Path(args.user_map) if args.user_map else user_map_path(checkpoint.parent, run_name)
        )
        with open(map_path) as f:
            user_map = json.load(f)
    timing["model_load_seconds"] = time.perf_counter() - t0

    # ---- data -----------------------------------------------------------
    t0 = time.perf_counter()
    dataset = ForgetPositiveDataset(forget_path, config, tokenizer=tokenizer, user_map=user_map)
    labels = read_model_csv(forget_path)["label"]
    group = (labels == 1).cumsum()
    group_ids = np.sort(group[group > 0].unique())
    if len(group_ids) != len(dataset):
        raise RuntimeError(f"{len(dataset)} samples for {len(group_ids)} groups")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    timing["data_load_seconds"] = time.perf_counter() - t0

    # ---- forward passes -------------------------------------------------
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    users, cands = [], []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                recommender(batch)
            out = recommender.intermediate_outputs
            users.append(out["user_embedding"].float())
            cands.append(out["candidate_embeddings"][:, 0].float())
    if device.type == "cuda":
        torch.cuda.synchronize()
    timing["forward_seconds"] = time.perf_counter() - t0

    user = torch.cat(users).numpy()
    cand = torch.cat(cands).numpy()
    timing["total_seconds"] = time.perf_counter() - t_start
    timing = {k: round(v, 4) for k, v in timing.items()}

    meta = {
        "checkpoint": str(checkpoint),
        "run_name": run_name,
        "forget": str(forget_path),
        "config": str(args.config),
        "architecture": config.architecture,
        "model_name": config.model_name,
        "device": str(device),
        "batch_size": loader.batch_size,
        "n_groups": int(len(group_ids)),
        "dim": int(user.shape[1]),
        "timing": timing,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        group_ids=group_ids,
        user=user,
        cand=cand,
        uc=user * cand,
        meta=json.dumps(meta),
    )
    with open(out_path.with_suffix(".timing.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"{len(group_ids)} groups, dim {user.shape[1]} -> {out_path}")
    print("timing: " + ", ".join(f"{k}={v}s" for k, v in timing.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
