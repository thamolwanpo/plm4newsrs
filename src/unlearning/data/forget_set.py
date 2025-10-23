# src/unlearning/data/forget_set.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import json


@dataclass
class ForgetSetMetadata:
    """Metadata for forget set."""

    mode: str
    use_label_correction: bool = False
    ratio: Optional[float] = None
    trial_idx: Optional[int] = None
    num_forget: int = 0
    num_corrected: int = 0
    num_retain: int = 0
    created_at: Optional[str] = None
    source_file: Optional[str] = None


class ForgetSet:
    def __init__(
        self,
        forget_df: pd.DataFrame,
        retain_df: pd.DataFrame,
        metadata: ForgetSetMetadata,
        corrected_df: Optional[pd.DataFrame] = None,
    ):
        self.forget_df = forget_df
        self.retain_df = retain_df
        self.corrected_df = corrected_df
        self.metadata = metadata

        if corrected_df is not None:
            self.metadata.use_label_correction = True

        self._validate()

    def _validate(self):
        if len(self.forget_df) == 0:
            raise ValueError("Forget set is empty")
        if len(self.retain_df) == 0:
            raise ValueError("Retain set is empty")

        required_cols = {
            "user_id",
            "history_titles",
            "candidate_id",
            "candidate_title",
            "label",
            "candidate_is_fake",
        }

        for name, df in [("Forget", self.forget_df), ("Retain", self.retain_df)]:
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"{name} set missing columns: {missing}")

        if self.corrected_df is not None:
            missing = required_cols - set(self.corrected_df.columns)
            if missing:
                raise ValueError(f"Corrected set missing columns: {missing}")

            if len(self.forget_df) != len(self.corrected_df):
                raise ValueError(
                    f"Forget ({len(self.forget_df)}) and corrected ({len(self.corrected_df)}) "
                    f"must have same number of samples"
                )

            print(f"✅ ForgetSet validation passed (Label Correction Mode)")
            print(f"   Z (wrong labels): {len(self.forget_df)} samples")
            print(f"   Z̃ (correct labels): {len(self.corrected_df)} samples")
            print(f"   Retain: {len(self.retain_df)} samples")
        else:
            print(f"✅ ForgetSet validation passed (Data Removal Mode)")
            print(f"   Forget: {len(self.forget_df)} samples")
            print(f"   Retain: {len(self.retain_df)} samples")

    @classmethod
    def from_manual(
        cls, forget_path: Path, retain_path: Path, corrected_path: Optional[Path] = None
    ) -> "ForgetSet":
        forget_path = Path(forget_path)
        retain_path = Path(retain_path)

        print(f"\n{'='*70}")
        print("LOADING FORGET SET (Manual Mode)")
        print(f"{'='*70}")
        print(f"Forget path: {forget_path}")
        if corrected_path:
            print(f"Corrected path: {corrected_path}")
        print(f"Retain path: {retain_path}")

        if not forget_path.exists():
            raise FileNotFoundError(f"Forget set not found: {forget_path}")
        if not retain_path.exists():
            raise FileNotFoundError(f"Retain set not found: {retain_path}")

        print("\nLoading CSVs...")
        forget_df = pd.read_csv(forget_path)
        retain_df = pd.read_csv(retain_path)

        corrected_df = None
        if corrected_path is not None:
            corrected_path = Path(corrected_path)
            if not corrected_path.exists():
                raise FileNotFoundError(f"Corrected set not found: {corrected_path}")
            corrected_df = pd.read_csv(corrected_path)
            print(f"Loaded corrected set: {len(corrected_df)} samples")

        metadata = ForgetSetMetadata(
            mode="manual",
            use_label_correction=(corrected_df is not None),
            num_forget=len(forget_df),
            num_corrected=len(corrected_df) if corrected_df is not None else 0,
            num_retain=len(retain_df),
            source_file=forget_path.name,
        )

        print(f"✅ Loaded forget set (manual mode)")
        print(f"   Mode: {'Label Correction' if corrected_df is not None else 'Data Removal'}")
        print(f"   Forget samples: {len(forget_df)}")
        if corrected_df is not None:
            print(f"   Corrected samples: {len(corrected_df)}")
        print(f"   Retain samples: {len(retain_df)}")

        return cls(forget_df, retain_df, metadata, corrected_df=corrected_df)

    @classmethod
    def from_ratio(cls, splits_dir: Path, trial_idx: int = 0) -> "ForgetSet":
        splits_dir = Path(splits_dir)
        trial_dir = splits_dir / f"trial_{trial_idx}"

        print(f"\n{'='*70}")
        print("LOADING FORGET SET (Ratio Mode)")
        print(f"{'='*70}")
        print(f"Splits directory: {splits_dir}")
        print(f"Trial: {trial_idx}")

        if not trial_dir.exists():
            raise FileNotFoundError(
                f"Trial directory not found: {trial_dir}\n"
                f"Available trials: {list(splits_dir.glob('trial_*'))}"
            )

        forget_path = trial_dir / "Z_forget.csv"
        corrected_path = trial_dir / "Z_tilde.csv"
        retain_path = trial_dir / "Z_retain.csv"
        metadata_path = trial_dir / "metadata.json"

        if not forget_path.exists():
            raise FileNotFoundError(f"Forget set not found: {forget_path}")
        if not retain_path.exists():
            raise FileNotFoundError(f"Retain set not found: {retain_path}")

        use_label_correction = corrected_path.exists()

        print(f"\nDetected mode: {'Label Correction' if use_label_correction else 'Data Removal'}")
        print("Loading CSVs...")

        forget_df = pd.read_csv(forget_path)
        retain_df = pd.read_csv(retain_path)

        corrected_df = None
        if use_label_correction:
            corrected_df = pd.read_csv(corrected_path)
            print(f"Loaded Z_tilde.csv: {len(corrected_df)} samples")

        ratio = None
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
                ratio = metadata_dict.get("ratio")

        metadata = ForgetSetMetadata(
            mode="ratio",
            use_label_correction=use_label_correction,
            ratio=ratio,
            trial_idx=trial_idx,
            num_forget=len(forget_df),
            num_corrected=len(corrected_df) if corrected_df is not None else 0,
            num_retain=len(retain_df),
            source_file=str(trial_dir),
        )

        print(f"✅ Loaded forget set (ratio mode)")
        print(f"   Ratio: {ratio}")
        print(f"   Trial: {trial_idx}")
        print(f"   Mode: {'Label Correction' if use_label_correction else 'Data Removal'}")
        print(f"   Forget samples: {len(forget_df)}")
        if corrected_df is not None:
            print(f"   Corrected samples: {len(corrected_df)}")
        print(f"   Retain samples: {len(retain_df)}")

        return cls(forget_df, retain_df, metadata, corrected_df=corrected_df)

    def get_corrected_dataframe(self) -> pd.DataFrame:
        if self.corrected_df is None:
            raise ValueError("No corrected set available. This ForgetSet is in data removal mode.")
        return self.corrected_df.copy()

    def is_label_correction(self) -> bool:
        return self.corrected_df is not None

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
                Z_forget.csv
                Z_retain.csv
                metadata.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSVs
        self.forget_df.to_csv(output_dir / "Z_forget.csv", index=False)
        self.retain_df.to_csv(output_dir / "Z_retain.csv", index=False)

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
