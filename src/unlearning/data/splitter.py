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
    use_label_correction: bool = True,
) -> List[Path]:
    """Create ratio-based splits for unlearning experiments."""

    data_path = Path(data_path)

    print(f"\n{'='*70}")
    print(f"CREATING RATIO-BASED SPLITS")
    print(f"{'='*70}")
    print(f"Source: {data_path}")
    print(f"Ratio: {ratio} ({ratio*100:.1f}%)")
    print(f"Trials: {num_trials}")
    print(f"Mode: {'Label Correction' if use_label_correction else 'Data Removal'}")
    print(f"{'='*70}")

    df = pd.read_csv(data_path)
    print(f"Total samples: {len(df)}")

    required_cols = {"user_id", "candidate_id", "label", "history_titles", "candidate_is_fake"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if output_dir is None:
        output_dir = data_path.parent / "unlearning_splits"

    ratio_dir_name = f"ratio_{ratio:.2f}".replace(".", "_")
    ratio_dir = output_dir / ratio_dir_name
    ratio_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {ratio_dir}")

    trial_dirs = []

    for trial_idx in range(num_trials):
        print(f"\n{'─'*70}")
        print(f"Creating trial {trial_idx}...")
        np.random.seed(seed + trial_idx)

        fake_pos_mask = (df["candidate_is_fake"] == True) & (df["label"] == 1)
        fake_pos_df = df.loc[fake_pos_mask, ["user_id", "history_titles"]].drop_duplicates()

        print(f"Found {len(fake_pos_df)} unique fake-positive contexts")

        if len(fake_pos_df) == 0:
            raise ValueError("No fake-positive interactions found in dataset.")

        n_select = max(1, int(len(fake_pos_df) * ratio))
        selected_contexts = fake_pos_df.sample(n=n_select, random_state=seed + trial_idx)
        selected_contexts_set = set(map(tuple, selected_contexts.values))

        print(f"Selected {n_select} contexts ({ratio*100:.1f}%)")

        def is_selected_context(row):
            return (row["user_id"], row["history_titles"]) in selected_contexts_set

        forget_mask = df.apply(is_selected_context, axis=1)

        forget_group_df = df[forget_mask].copy()
        retain_df = df[~forget_mask].copy()

        n_pos = forget_group_df.query("label == 1").shape[0]
        n_neg = forget_group_df.query("label == 0").shape[0]

        print(
            f"Selected groups contain: {len(forget_group_df)} samples "
            f"({n_pos} positive, {n_neg} negative)"
        )

        trial_dir = ratio_dir / f"trial_{trial_idx}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        if use_label_correction:
            forget_df = forget_group_df.copy()

            corrected_df = forget_group_df.copy()
            fake_pos_in_group = (corrected_df["candidate_is_fake"] == True) & (
                corrected_df["label"] == 1
            )
            corrected_df.loc[fake_pos_in_group, "label"] = 0

            n_flipped = fake_pos_in_group.sum()

            forget_df = forget_df.sample(frac=1, random_state=seed + trial_idx).reset_index(
                drop=True
            )
            corrected_df = corrected_df.sample(frac=1, random_state=seed + trial_idx).reset_index(
                drop=True
            )
            retain_df = retain_df.sample(frac=1, random_state=seed + trial_idx).reset_index(
                drop=True
            )

            forget_df.to_csv(trial_dir / "forget.csv", index=False)
            corrected_df.to_csv(trial_dir / "corrected.csv", index=False)
            retain_df.to_csv(trial_dir / "retain.csv", index=False)

            verification_message = (
                f"Label Correction: {n_flipped} fake-positive labels flipped (1→0), "
                f"preserving {len(forget_df)} total samples in groups"
            )

            metadata = {
                "ratio": ratio,
                "trial_idx": trial_idx,
                "seed": seed + trial_idx,
                "mode": "label_correction",
                "use_label_correction": True,
                "total_samples": len(df),
                "forget_samples": len(forget_df),
                "corrected_samples": len(corrected_df),
                "retain_samples": len(retain_df),
                "labels_flipped": int(n_flipped),
                "created_at": datetime.now().isoformat(),
                "verification": verification_message,
                "note": "forget.csv has wrong labels, corrected.csv has flipped labels",
            }

        else:
            forget_df = forget_group_df.copy()

            forget_df = forget_df.sample(frac=1, random_state=seed + trial_idx).reset_index(
                drop=True
            )
            retain_df = retain_df.sample(frac=1, random_state=seed + trial_idx).reset_index(
                drop=True
            )

            forget_df.to_csv(trial_dir / "forget.csv", index=False)
            retain_df.to_csv(trial_dir / "retain.csv", index=False)

            verification_message = (
                f"Data Removal: {len(forget_df)} samples removed "
                f"({n_pos} positive, {n_neg} negative)"
            )

            metadata = {
                "ratio": ratio,
                "trial_idx": trial_idx,
                "seed": seed + trial_idx,
                "mode": "data_removal",
                "use_label_correction": False,
                "total_samples": len(df),
                "forget_samples": len(forget_df),
                "retain_samples": len(retain_df),
                "created_at": datetime.now().isoformat(),
                "verification": verification_message,
            }

        with open(trial_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ {verification_message}")
        print(f"  ✅ Saved: {trial_dir}")
        trial_dirs.append(trial_dir)

    print(f"\n{'='*70}")
    print(f"✅ Created {num_trials} trials")
    print(f"Mode: {'Label Correction' if use_label_correction else 'Data Removal'}")
    print(f"{'='*70}")

    return trial_dirs


def create_multiple_ratios(
    data_path: Path,
    ratios: List[float] = [0.01, 0.05, 0.10, 0.20],
    num_trials: int = 3,
    output_dir: Optional[Path] = None,
    seed: int = 42,
    stratify_by: Optional[str] = None,
    use_label_correction: bool = True,
) -> Dict[float, List[Path]]:
    """Create splits for multiple forget ratios."""
    all_trial_dirs = {}
    for ratio in ratios:
        trial_dirs = create_ratio_split(
            data_path=data_path,
            ratio=ratio,
            num_trials=num_trials,
            output_dir=output_dir,
            seed=seed,
            stratify_by=stratify_by,
            use_label_correction=use_label_correction,
        )
        all_trial_dirs[ratio] = trial_dirs
    return all_trial_dirs
