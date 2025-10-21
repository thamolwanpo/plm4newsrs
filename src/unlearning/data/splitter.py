# src/unlearning/data/splitter.py

from pathlib import Path
from typing import Tuple, List, Optional
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
) -> List[Path]:
    """
    Create ratio-based splits for unlearning experiments.

    Args:
        data_path: Path to source CSV file
        ratio: Ratio of samples to put in forget set (e.g., 0.05 = 5%)
        num_trials: Number of random splits to generate
        output_dir: Output directory (default: same as data_path parent)
        seed: Random seed base
        stratify_by: Column to stratify by (e.g., 'candidate_is_fake')

    Returns:
        List of paths to trial directories

    Example:
        >>> trial_dirs = create_ratio_split(
        ...     data_path=Path("data/politifact/train_poisoned.csv"),
        ...     ratio=0.05,
        ...     num_trials=3,
        ...     seed=42
        ... )
        >>> # Creates:
        >>> # data/politifact/unlearning_splits/ratio_0.05/
        >>> #   trial_0/
        >>> #     forget.csv
        >>> #     retain.csv
        >>> #     metadata.json
        >>> #   trial_1/
        >>> #   trial_2/
    """
    data_path = Path(data_path)

    print(f"\n{'='*70}")
    print("CREATING RATIO-BASED SPLITS")
    print(f"{'='*70}")
    print(f"Source: {data_path}")
    print(f"Ratio: {ratio} ({ratio*100:.1f}%)")
    print(f"Trials: {num_trials}")
    print(f"Stratify by: {stratify_by}")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    print(f"Total samples: {len(df)}")

    # Determine output directory
    if output_dir is None:
        output_dir = data_path.parent / "unlearning_splits"

    ratio_dir = output_dir / f"ratio_{ratio:.2f}".replace(".", "_")
    ratio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {ratio_dir}")

    trial_dirs = []

    # Create trials
    for trial_idx in range(num_trials):
        print(f"\n{'─'*70}")
        print(f"Creating trial {trial_idx}...")

        # Set seed for this trial
        np.random.seed(seed + trial_idx)

        # Split data
        if stratify_by and stratify_by in df.columns:
            # Stratified split
            forget_dfs = []
            retain_dfs = []

            for value in df[stratify_by].unique():
                subset = df[df[stratify_by] == value]
                n_forget = int(len(subset) * ratio)

                indices = np.random.permutation(len(subset))
                forget_indices = indices[:n_forget]
                retain_indices = indices[n_forget:]

                forget_dfs.append(subset.iloc[forget_indices])
                retain_dfs.append(subset.iloc[retain_indices])

            forget_df = pd.concat(forget_dfs, ignore_index=True)
            retain_df = pd.concat(retain_dfs, ignore_index=True)

        else:
            # Random split
            n_forget = int(len(df) * ratio)
            indices = np.random.permutation(len(df))

            forget_indices = indices[:n_forget]
            retain_indices = indices[n_forget:]

            forget_df = df.iloc[forget_indices].reset_index(drop=True)
            retain_df = df.iloc[retain_indices].reset_index(drop=True)

        # Shuffle
        forget_df = forget_df.sample(frac=1, random_state=seed + trial_idx).reset_index(drop=True)
        retain_df = retain_df.sample(frac=1, random_state=seed + trial_idx).reset_index(drop=True)

        print(f"  Forget: {len(forget_df)} samples ({len(forget_df)/len(df)*100:.2f}%)")
        print(f"  Retain: {len(retain_df)} samples ({len(retain_df)/len(df)*100:.2f}%)")

        # Save trial
        trial_dir = ratio_dir / f"trial_{trial_idx}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        forget_df.to_csv(trial_dir / "forget.csv", index=False)
        retain_df.to_csv(trial_dir / "retain.csv", index=False)

        # Save metadata
        metadata = {
            "ratio": ratio,
            "trial_idx": trial_idx,
            "seed": seed + trial_idx,
            "num_forget": len(forget_df),
            "num_retain": len(retain_df),
            "total": len(df),
            "source_file": str(data_path),
            "stratify_by": stratify_by,
            "created_at": datetime.now().isoformat(),
        }
