from pathlib import Path
from typing import Tuple, List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import json


def create_ratio_split(
    data_path: Path,
    ratio: float = 0.05,
    num_trials: int = 3,
    output_dir: Optional[Path] = None,
    seed: int = 42,
    stratify_by: Optional[str] = None,
    removal_strategy: str = "fake_positive_history",  # new default
) -> List[Path]:
    """
    Create ratio-based splits for unlearning experiments.

    Splits can now remove fake-positive samples and their related negatives
    within the same (user_id, history_titles) context.

    removal_strategy options:
        - "complete": remove all rows with forget candidates
        - "positive_only": remove only positive samples of forget candidates
        - "fake_positive_history": remove fake-positive interactions and all
                                   other rows with the same (user_id, history_titles)
    """

    data_path = Path(data_path)

    valid_strategies = ["complete", "positive_only", "fake_positive_history"]
    if removal_strategy not in valid_strategies:
        raise ValueError(
            f"Invalid removal_strategy: {removal_strategy}. Must be one of {valid_strategies}."
        )

    # --- Load Data ---
    print(f"\n{'='*70}")
    print(f"CREATING RATIO-BASED SPLITS")
    print(f"{'='*70}")
    print(f"Source: {data_path}")
    print(f"Ratio: {ratio} ({ratio*100:.1f}%)")
    print(f"Trials: {num_trials}")
    print(f"Removal Strategy: {removal_strategy}")

    df = pd.read_csv(data_path)
    print(f"Total samples: {len(df)}")

    required_cols = {"user_id", "candidate_id", "label", "history_titles"}
    if removal_strategy == "fake_positive_history":
        required_cols.add("candidate_is_fake")
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    unique_candidates = df["candidate_id"].unique()
    print(f"Unique candidates: {len(unique_candidates)}")

    # --- Output directory setup ---
    if output_dir is None:
        output_dir = data_path.parent / "unlearning_splits"

    ratio_dir_name = f"ratio_{ratio:.2f}".replace(".", "_")
    ratio_dir = output_dir / ratio_dir_name
    ratio_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {ratio_dir}")

    trial_dirs = []

    # --- Create Trials ---
    for trial_idx in range(num_trials):
        print(f"\n{'─'*70}")
        print(f"Creating trial {trial_idx}...")
        np.random.seed(seed + trial_idx)

        # --- Removal Logic ---
        if removal_strategy == "complete":
            # Random candidate-level split
            shuffled_candidates = unique_candidates.copy()
            np.random.shuffle(shuffled_candidates)
            n_forget = int(len(shuffled_candidates) * ratio)

            forget_candidate_set = set(shuffled_candidates[:n_forget])
            retain_candidate_set = set(shuffled_candidates[n_forget:])

            forget_df = df[df["candidate_id"].isin(forget_candidate_set)].reset_index(drop=True)
            retain_df = df[~df["candidate_id"].isin(forget_candidate_set)].reset_index(drop=True)
            verification_message = "Removed all rows of forget candidates (complete)."

        elif removal_strategy == "positive_only":
            # Random positive-only split at (user, candidate, history_titles)
            pos_triples = df.loc[df["label"] == 1, ["user_id", "candidate_id", "history_titles"]]
            selected_triples = pos_triples.sample(frac=ratio, random_state=seed + trial_idx)
            forget_triples = set(map(tuple, selected_triples.values))

            forget_mask = df.apply(
                lambda r: (r["user_id"], r["candidate_id"], r["history_titles"]) in forget_triples,
                axis=1,
            )
            forget_df = df[forget_mask].reset_index(drop=True)
            retain_df = df[~forget_mask].reset_index(drop=True)
            verification_message = "Removed only selected positive triples."

        elif removal_strategy == "fake_positive_history":
            # Step 1: Identify all fake-positive rows
            fake_pos_mask = (df["candidate_is_fake"] == True) & (df["label"] == 1)
            fake_pos_df = df.loc[fake_pos_mask, ["user_id", "history_titles"]].drop_duplicates()
            print(f"Fake-positive interactions found: {len(fake_pos_df)}")

            if len(fake_pos_df) == 0:
                raise ValueError("No fake-positive interactions found in dataset.")

            # Step 2: Randomly select a subset (ratio)
            n_select = max(1, int(len(fake_pos_df) * ratio))
            selected_contexts = fake_pos_df.sample(n=n_select, random_state=seed + trial_idx)
            forget_contexts = set(map(tuple, selected_contexts.values))

            # Step 3: Forget all rows sharing same (user_id, history_titles)
            forget_mask = df.apply(
                lambda r: (r["user_id"], r["history_titles"]) in forget_contexts,
                axis=1,
            )
            forget_df = df[forget_mask].reset_index(drop=True)
            retain_df = df[~forget_mask].reset_index(drop=True)

            # Step 4: Verification — ensure no context overlap
            forget_pairs = set(map(tuple, forget_df[["user_id", "history_titles"]].values))
            retain_pairs = set(map(tuple, retain_df[["user_id", "history_titles"]].values))
            overlap = forget_pairs & retain_pairs
            if overlap:
                raise ValueError(
                    f"Overlap detected between forget and retain contexts: {len(overlap)}"
                )

            n_pos_forgot = forget_df.query("label == 1").shape[0]
            n_neg_forgot = forget_df.query("label == 0").shape[0]
            verification_message = (
                f"Forgot {len(forget_pairs)} (user_id, history_titles) contexts "
                f"→ {len(forget_df)} rows ({n_pos_forgot} pos, {n_neg_forgot} neg)"
            )

        # Shuffle within sets
        forget_df = forget_df.sample(frac=1, random_state=seed + trial_idx).reset_index(drop=True)
        retain_df = retain_df.sample(frac=1, random_state=seed + trial_idx).reset_index(drop=True)

        # --- Save output ---
        trial_dir = ratio_dir / f"trial_{trial_idx}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        forget_df.to_csv(trial_dir / "forget.csv", index=False)
        retain_df.to_csv(trial_dir / "retain.csv", index=False)

        metadata = {
            "ratio": ratio,
            "trial_idx": trial_idx,
            "seed": seed + trial_idx,
            "total_samples": len(df),
            "forget_samples": len(forget_df),
            "retain_samples": len(retain_df),
            "removal_strategy": removal_strategy,
            "created_at": datetime.now().isoformat(),
            "verification": verification_message,
        }

        with open(trial_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ {verification_message}")
        print(f"  ✅ Saved: {trial_dir}")
        trial_dirs.append(trial_dir)

    print(f"\n{'='*70}")
    print(f"✅ Created {num_trials} ratio-based splits (strategy={removal_strategy})")
    print(f"{'='*70}")

    return trial_dirs


# ---------------------------------------------------------------------


def create_multiple_ratios(
    data_path: Path,
    ratios: List[float] = [0.01, 0.05, 0.10, 0.20],
    num_trials: int = 3,
    output_dir: Optional[Path] = None,
    seed: int = 42,
    stratify_by: Optional[str] = None,
    removal_strategy: str = "fake_positive_history",
) -> Dict[float, List[Path]]:
    """
    Create splits for multiple forget ratios using the chosen removal strategy.
    """
    all_trial_dirs = {}
    for ratio in ratios:
        trial_dirs = create_ratio_split(
            data_path=data_path,
            ratio=ratio,
            num_trials=num_trials,
            output_dir=output_dir,
            seed=seed,
            stratify_by=stratify_by,
            removal_strategy=removal_strategy,
        )
        all_trial_dirs[ratio] = trial_dirs
    return all_trial_dirs
