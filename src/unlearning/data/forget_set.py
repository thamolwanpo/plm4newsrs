# src/unlearning/data/forget_set.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import json


@dataclass
class ForgetSetMetadata:
    """Metadata for forget set."""

    mode: str  # "manual" or "ratio"
    ratio: Optional[float] = None
    trial_idx: Optional[int] = None
    num_forget: int = 0
    num_retain: int = 0
    created_at: Optional[str] = None
    source_file: Optional[str] = None


class ForgetSet:
    """
    Manages forget and retain sets for unlearning.

    Supports two modes:
    1. Manual: User provides explicit forget/retain CSV paths
    2. Ratio: Load from pre-generated splits directory

    Example:
        # Manual mode
        >>> forget_set = ForgetSet.from_manual(
        ...     forget_path="data/forget.csv",
        ...     retain_path="data/retain.csv"
        ... )

        # Ratio mode
        >>> forget_set = ForgetSet.from_ratio(
        ...     splits_dir="data/politifact/unlearning_splits/ratio_0.05",
        ...     trial_idx=0
        ... )
    """

    def __init__(
        self, forget_df: pd.DataFrame, retain_df: pd.DataFrame, metadata: ForgetSetMetadata
    ):
        """
        Initialize ForgetSet.

        Args:
            forget_df: DataFrame with forget samples
            retain_df: DataFrame with retain samples
            metadata: Metadata about this split
        """
        self.forget_df = forget_df
        self.retain_df = retain_df
        self.metadata = metadata

        # Validate
        self._validate()

    def _validate(self):
        """Validate forget and retain sets."""
        # Check not empty
        if len(self.forget_df) == 0:
            raise ValueError("Forget set is empty")

        if len(self.retain_df) == 0:
            raise ValueError("Retain set is empty")

        # Check required columns (should match training data format)
        required_cols = {
            "user_id",
            "history_titles",
            "candidate_id",
            "candidate_title",
            "label",
            "candidate_is_fake",
        }

        forget_cols = set(self.forget_df.columns)
        retain_cols = set(self.retain_df.columns)

        if not required_cols.issubset(forget_cols):
            missing = required_cols - forget_cols
            raise ValueError(f"Forget set missing columns: {missing}")

        if not required_cols.issubset(retain_cols):
            missing = required_cols - retain_cols
            raise ValueError(f"Retain set missing columns: {missing}")

        # Check no overlap (by candidate_id)
        forget_ids = set(self.forget_df["candidate_id"])
        retain_ids = set(self.retain_df["candidate_id"])
        overlap = forget_ids & retain_ids

        if overlap:
            raise ValueError(
                f"Overlap detected between forget and retain sets: "
                f"{len(overlap)} common candidate IDs"
            )

        print(f"✅ ForgetSet validation passed")
        print(f"   Forget: {len(self.forget_df)} samples")
        print(f"   Retain: {len(self.retain_df)} samples")
        print(f"   No overlap detected")

    @classmethod
    def from_manual(cls, forget_path: Path, retain_path: Path) -> "ForgetSet":
        """
        Load forget set from explicit paths (Manual mode).

        Args:
            forget_path: Path to forget set CSV
            retain_path: Path to retain set CSV

        Returns:
            ForgetSet instance

        Example:
            >>> forget_set = ForgetSet.from_manual(
            ...     forget_path=Path("data/forget.csv"),
            ...     retain_path=Path("data/retain.csv")
            ... )
        """
        forget_path = Path(forget_path)
        retain_path = Path(retain_path)

        print(f"\n{'='*70}")
        print("LOADING FORGET SET (Manual Mode)")
        print(f"{'='*70}")
        print(f"Forget path: {forget_path}")
        print(f"Retain path: {retain_path}")

        # Check files exist
        if not forget_path.exists():
            raise FileNotFoundError(f"Forget set not found: {forget_path}")

        if not retain_path.exists():
            raise FileNotFoundError(f"Retain set not found: {retain_path}")

        # Load CSVs
        print("\nLoading CSVs...")
        forget_df = pd.read_csv(forget_path)
        retain_df = pd.read_csv(retain_path)

        # Create metadata
        metadata = ForgetSetMetadata(
            mode="manual",
            num_forget=len(forget_df),
            num_retain=len(retain_df),
            source_file=f"{forget_path.name}, {retain_path.name}",
        )

        print(f"✅ Loaded forget set (manual mode)")
        print(f"   Forget samples: {len(forget_df)}")
        print(f"   Retain samples: {len(retain_df)}")

        return cls(forget_df, retain_df, metadata)

    @classmethod
    def from_ratio(cls, splits_dir: Path, trial_idx: int = 0) -> "ForgetSet":
        """
        Load forget set from splits directory (Ratio mode).

        Args:
            splits_dir: Directory containing splits (e.g., ratio_0.05/trial_0/)
            trial_idx: Trial index to load

        Returns:
            ForgetSet instance

        Example:
            >>> forget_set = ForgetSet.from_ratio(
            ...     splits_dir=Path("data/politifact/unlearning_splits/ratio_0.05"),
            ...     trial_idx=0
            ... )
        """
        splits_dir = Path(splits_dir)
        trial_dir = splits_dir / f"trial_{trial_idx}"

        print(f"\n{'='*70}")
        print("LOADING FORGET SET (Ratio Mode)")
        print(f"{'='*70}")
        print(f"Splits directory: {splits_dir}")
        print(f"Trial: {trial_idx}")

        # Check directory exists
        if not trial_dir.exists():
            raise FileNotFoundError(
                f"Trial directory not found: {trial_dir}\n"
                f"Available trials: {list(splits_dir.glob('trial_*'))}"
            )

        # Load forget and retain
        forget_path = trial_dir / "forget.csv"
        retain_path = trial_dir / "retain.csv"
        metadata_path = trial_dir / "metadata.json"

        if not forget_path.exists():
            raise FileNotFoundError(f"Forget set not found: {forget_path}")

        if not retain_path.exists():
            raise FileNotFoundError(f"Retain set not found: {retain_path}")

        print("\nLoading CSVs...")
        forget_df = pd.read_csv(forget_path)
        retain_df = pd.read_csv(retain_path)

        # Load metadata if available
        ratio = None
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
                ratio = metadata_dict.get("ratio")

        # Create metadata
        metadata = ForgetSetMetadata(
            mode="ratio",
            ratio=ratio,
            trial_idx=trial_idx,
            num_forget=len(forget_df),
            num_retain=len(retain_df),
            source_file=str(trial_dir),
        )

        print(f"✅ Loaded forget set (ratio mode)")
        print(f"   Ratio: {ratio}")
        print(f"   Trial: {trial_idx}")
        print(f"   Forget samples: {len(forget_df)}")
        print(f"   Retain samples: {len(retain_df)}")

        return cls(forget_df, retain_df, metadata)

    def get_forget_dataframe(self) -> pd.DataFrame:
        """Get forget set as DataFrame."""
        return self.forget_df.copy()

    def get_retain_dataframe(self) -> pd.DataFrame:
        """Get retain set as DataFrame."""
        return self.retain_df.copy()

    def get_combined_dataframe(self) -> pd.DataFrame:
        """Get combined forget + retain as single DataFrame."""
        return pd.concat([self.forget_df, self.retain_df], ignore_index=True)

    def save(self, output_dir: Path):
        """
        Save forget set to directory.

        Args:
            output_dir: Directory to save to

        Creates:
            output_dir/
                forget.csv
                retain.csv
                metadata.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSVs
        self.forget_df.to_csv(output_dir / "forget.csv", index=False)
        self.retain_df.to_csv(output_dir / "retain.csv", index=False)

        # Save metadata
        metadata_dict = {
            "mode": self.metadata.mode,
            "ratio": self.metadata.ratio,
            "trial_idx": self.metadata.trial_idx,
            "num_forget": self.metadata.num_forget,
            "num_retain": self.metadata.num_retain,
            "source_file": self.metadata.source_file,
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata_dict, f, indent=2)

        print(f"✅ Saved ForgetSet to: {output_dir}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about forget set."""
        stats = {
            "mode": self.metadata.mode,
            "num_forget": len(self.forget_df),
            "num_retain": len(self.retain_df),
            "total": len(self.forget_df) + len(self.retain_df),
            "forget_ratio": len(self.forget_df) / (len(self.forget_df) + len(self.retain_df)),
        }

        # Add fake news stats if available
        if "candidate_is_fake" in self.forget_df.columns:
            stats["forget_fake_count"] = self.forget_df["candidate_is_fake"].sum()
            stats["forget_fake_ratio"] = self.forget_df["candidate_is_fake"].mean()
            stats["retain_fake_count"] = self.retain_df["candidate_is_fake"].sum()
            stats["retain_fake_ratio"] = self.retain_df["candidate_is_fake"].mean()

        return stats

    def print_summary(self):
        """Print summary of forget set."""
        stats = self.get_statistics()

        print(f"\n{'='*70}")
        print("FORGET SET SUMMARY")
        print(f"{'='*70}")
        print(f"Mode: {stats['mode']}")
        print(f"Total samples: {stats['total']}")
        print(f"  Forget: {stats['num_forget']} ({stats['forget_ratio']:.1%})")
        print(f"  Retain: {stats['num_retain']} ({1-stats['forget_ratio']:.1%})")

        if "forget_fake_count" in stats:
            print(f"\nFake news distribution:")
            print(
                f"  Forget set: {stats['forget_fake_count']} fake ({stats['forget_fake_ratio']:.1%})"
            )
            print(
                f"  Retain set: {stats['retain_fake_count']} fake ({stats['retain_fake_ratio']:.1%})"
            )

        print(f"{'='*70}")

    def __repr__(self) -> str:
        return (
            f"ForgetSet(mode={self.metadata.mode}, "
            f"forget={len(self.forget_df)}, "
            f"retain={len(self.retain_df)})"
        )
