"""Analyzer for machine unlearning effectiveness."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base_analyzer import BaseAnalyzer
from ..metrics.misinformation import (
    calculate_fake_exposure,
    calculate_unlearning_effectiveness,
    calculate_contamination_rate,
)
from ..metrics.ranking import calculate_auc, calculate_mrr, calculate_ndcg_at_k
from ..metrics.utility import safe_divide


class UnlearningAnalyzer(BaseAnalyzer):
    """
    Analyze machine unlearning effectiveness.

    Key metric: Gap from clean baseline
    - How close is the unlearned model to the original clean model?
    - Recovery rate: How much contamination was removed?

    This is the critical analysis for machine unlearning experiments.
    """

    def __init__(self, results_dir: Path, name: str = "unlearning", target_gap: float = 0.01):
        """
        Initialize unlearning analyzer.

        Args:
            results_dir: Directory to save results
            name: Name for this analyzer
            target_gap: Target gap threshold for success (default: 0.01)
        """
        super().__init__(results_dir, name)
        self.target_gap = target_gap

    def get_required_columns(self) -> List[str]:
        """Required columns for unlearning analysis."""
        return ["user_id", "score", "is_fake", "label"]

    def analyze(
        self,
        clean_results: pd.DataFrame,
        poisoned_results: pd.DataFrame,
        unlearned_results: Optional[pd.DataFrame] = None,
        unlearned_name: str = "unlearned",  # Add this parameter
        top_k: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze unlearning effectiveness by comparing clean, poisoned, and unlearned models.

        Args:
            clean_results: Results from clean baseline model
            poisoned_results: Results from poisoned model
            unlearned_results: Results from unlearned model (optional)
            top_k: Top-k for exposure analysis

        Returns:
            Dictionary with unlearning analysis:
            {
                'has_unlearned': Whether unlearned model exists,
                'contamination': Contamination metrics (clean vs poisoned),
                'effectiveness': Unlearning effectiveness (if unlearned exists),
                'exposure_comparison': Exposure stats for all models,
                'quality_comparison': Quality metrics for all models,
                'gap_analysis': Detailed gap from clean analysis
            }
        """
        # Validate all datasets
        self.validate_data(clean_results)
        self.validate_data(poisoned_results)

        has_unlearned = unlearned_results is not None
        if has_unlearned:
            self.validate_data(unlearned_results)

        print(f"\nAnalyzing unlearning effectiveness (target gap: {self.target_gap})...")

        # 1. Calculate contamination (clean vs poisoned)
        contamination = calculate_contamination_rate(clean_results, poisoned_results, k=top_k)

        # 2. Exposure comparison
        exposure_comparison = self._compare_exposure(
            clean_results, poisoned_results, unlearned_results, top_k
        )

        # 3. Quality comparison
        quality_comparison = self._compare_quality(
            clean_results, poisoned_results, unlearned_results
        )

        results = {
            "has_unlearned": has_unlearned,
            "contamination": contamination,
            "exposure_comparison": exposure_comparison,
            "quality_comparison": quality_comparison,
            "top_k": top_k,
            "target_gap": self.target_gap,
        }

        # 4. Unlearning effectiveness (if unlearned model exists)
        if has_unlearned:
            effectiveness = calculate_unlearning_effectiveness(
                clean_results,
                poisoned_results,
                unlearned_results,
                k=top_k,
                target_gap=self.target_gap,
            )

            gap_analysis = self._analyze_gaps(
                clean_results, poisoned_results, unlearned_results, top_k
            )

            results["effectiveness"] = effectiveness
            results["gap_analysis"] = gap_analysis
        else:
            results["effectiveness"] = None
            results["gap_analysis"] = None

        self.analysis_results = results
        return results

    def _compare_exposure(
        self,
        clean_results: pd.DataFrame,
        poisoned_results: pd.DataFrame,
        unlearned_results: Optional[pd.DataFrame],
        k: int,
    ) -> pd.DataFrame:
        """
        Compare fake news exposure across models.

        Args:
            clean_results: Clean model results
            poisoned_results: Poisoned model results
            unlearned_results: Unlearned model results (optional)
            k: Top-k cutoff

        Returns:
            DataFrame with exposure comparison
        """
        comparison_data = []

        # Clean model
        clean_exposure = calculate_fake_exposure(clean_results, k=k)
        comparison_data.append(
            {
                "model": "clean",
                "avg_fake_in_top_k": clean_exposure["avg_fake_count"],
                "max_fake_in_top_k": clean_exposure["max_fake_count"],
                "std_fake_in_top_k": clean_exposure["std_fake_count"],
                "users_with_fake_pct": clean_exposure["users_with_fake_pct"],
                "avg_fake_ratio": clean_exposure["avg_fake_ratio"],
                "avg_min_fake_rank": clean_exposure["avg_min_fake_rank"],
            }
        )

        # Poisoned model
        poisoned_exposure = calculate_fake_exposure(poisoned_results, k=k)
        comparison_data.append(
            {
                "model": "poisoned",
                "avg_fake_in_top_k": poisoned_exposure["avg_fake_count"],
                "max_fake_in_top_k": poisoned_exposure["max_fake_count"],
                "std_fake_in_top_k": poisoned_exposure["std_fake_count"],
                "users_with_fake_pct": poisoned_exposure["users_with_fake_pct"],
                "avg_fake_ratio": poisoned_exposure["avg_fake_ratio"],
                "avg_min_fake_rank": poisoned_exposure["avg_min_fake_rank"],
            }
        )

        # Unlearned model (if exists)
        if unlearned_results is not None:
            unlearned_exposure = calculate_fake_exposure(unlearned_results, k=k)
            comparison_data.append(
                {
                    "model": "unlearned",
                    "avg_fake_in_top_k": unlearned_exposure["avg_fake_count"],
                    "max_fake_in_top_k": unlearned_exposure["max_fake_count"],
                    "std_fake_in_top_k": unlearned_exposure["std_fake_count"],
                    "users_with_fake_pct": unlearned_exposure["users_with_fake_pct"],
                    "avg_fake_ratio": unlearned_exposure["avg_fake_ratio"],
                    "avg_min_fake_rank": unlearned_exposure["avg_min_fake_rank"],
                }
            )

        return pd.DataFrame(comparison_data)

    def _compare_quality(
        self,
        clean_results: pd.DataFrame,
        poisoned_results: pd.DataFrame,
        unlearned_results: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Compare ranking quality metrics across models.

        Args:
            clean_results: Clean model results
            poisoned_results: Poisoned model results
            unlearned_results: Unlearned model results (optional)

        Returns:
            DataFrame with quality comparison
        """
        comparison_data = []

        for model_name, results_df in [
            ("clean", clean_results),
            ("poisoned", poisoned_results),
            ("unlearned", unlearned_results),
        ]:
            if results_df is None:
                continue

            # Calculate metrics
            auc = calculate_auc(results_df)

            # Calculate MRR and NDCG per user, then average
            mrr_scores = []
            ndcg_scores = []

            for user_id, group in results_df.groupby("user_id"):
                # MRR
                sorted_group = group.sort_values("score", ascending=False)
                positive_ranks = np.where(sorted_group["label"].values == 1)[0]
                if len(positive_ranks) > 0:
                    rank = positive_ranks[0] + 1
                    mrr_scores.append(1.0 / rank)

                # NDCG@10
                ndcg = calculate_ndcg_at_k(group, k=10)
                ndcg_scores.append(ndcg)

            mrr = np.mean(mrr_scores) if mrr_scores else 0.0
            ndcg_10 = np.mean(ndcg_scores) if ndcg_scores else 0.0

            comparison_data.append(
                {"model": model_name, "AUC": auc, "MRR": mrr, "NDCG@10": ndcg_10}
            )

        return pd.DataFrame(comparison_data)

    def _analyze_gaps(
        self,
        clean_results: pd.DataFrame,
        poisoned_results: pd.DataFrame,
        unlearned_results: pd.DataFrame,
        k: int,
    ) -> Dict[str, Any]:
        """
        Detailed gap from clean baseline analysis.

        Args:
            clean_results: Clean model results
            poisoned_results: Poisoned model results
            unlearned_results: Unlearned model results
            k: Top-k cutoff

        Returns:
            Dictionary with detailed gap metrics
        """
        # Exposure gaps
        clean_exposure = calculate_fake_exposure(clean_results, k=k)
        poisoned_exposure = calculate_fake_exposure(poisoned_results, k=k)
        unlearned_exposure = calculate_fake_exposure(unlearned_results, k=k)

        clean_fake = clean_exposure["avg_fake_count"]
        poisoned_fake = poisoned_exposure["avg_fake_count"]
        unlearned_fake = unlearned_exposure["avg_fake_count"]

        fake_gap = abs(unlearned_fake - clean_fake)
        poisoned_gap = abs(poisoned_fake - clean_fake)

        # Quality gaps
        quality_comp = self._compare_quality(clean_results, poisoned_results, unlearned_results)

        clean_quality = quality_comp[quality_comp["model"] == "clean"].iloc[0]
        poisoned_quality = quality_comp[quality_comp["model"] == "poisoned"].iloc[0]
        unlearned_quality = quality_comp[quality_comp["model"] == "unlearned"].iloc[0]

        quality_gaps = {}
        for metric in ["AUC", "MRR", "NDCG@10"]:
            clean_val = clean_quality[metric]
            unlearned_val = unlearned_quality[metric]
            poisoned_val = poisoned_quality[metric]

            quality_gaps[metric] = {
                "clean": clean_val,
                "poisoned": poisoned_val,
                "unlearned": unlearned_val,
                "gap_from_clean": abs(unlearned_val - clean_val),
                "gap_from_poisoned": abs(unlearned_val - poisoned_val),
                "poisoned_degradation": poisoned_val - clean_val,
            }

        # Overall assessment
        all_gaps = [fake_gap] + [
            quality_gaps[m]["gap_from_clean"] for m in ["AUC", "MRR", "NDCG@10"]
        ]
        avg_gap = np.mean(all_gaps)
        max_gap = np.max(all_gaps)

        return {
            "fake_exposure_gap": fake_gap,
            "poisoned_fake_gap": poisoned_gap,
            "quality_gaps": quality_gaps,
            "avg_gap": avg_gap,
            "max_gap": max_gap,
            "all_gaps_below_target": all([g < self.target_gap for g in all_gaps]),
        }

    def save_results(
        self, results: Optional[Dict[str, Any]] = None, unlearned_name: str = "unlearned"
    ) -> Path:
        """
        Save unlearning analysis results.

        Args:
            results: Analysis results (uses self.analysis_results if None)
            unlearned_name: Name of unlearned model for file naming

        Returns:
            Path to main results directory
        """
        if results is None:
            results = self.analysis_results

        if results is None:
            raise ValueError("No results to save. Run analyze() first.")

        # Save exposure comparison
        exposure_path = self.results_dir / f"{self.name}_exposure_comparison_{unlearned_name}.csv"
        results["exposure_comparison"].to_csv(exposure_path, index=False)

        # Save quality comparison
        quality_path = self.results_dir / f"{self.name}_quality_comparison_{unlearned_name}.csv"
        results["quality_comparison"].to_csv(quality_path, index=False)

        # Save contamination metrics (same for all models)
        contamination_data = pd.DataFrame([results["contamination"]])
        contamination_path = self.results_dir / f"{self.name}_contamination.csv"
        contamination_data.to_csv(contamination_path, index=False)

        print(f"✅ Saved exposure comparison: {exposure_path.name}")
        print(f"✅ Saved quality comparison: {quality_path.name}")
        print(f"✅ Saved contamination: {contamination_path.name}")

        # Save effectiveness and gaps if unlearned model exists
        if results["has_unlearned"]:
            effectiveness_data = pd.DataFrame([results["effectiveness"]])
            effectiveness_path = (
                self.results_dir / f"{self.name}_effectiveness_{unlearned_name}.csv"
            )
            effectiveness_data.to_csv(effectiveness_path, index=False)

            # Save gap analysis
            gap_summary = {
                "fake_exposure_gap": results["gap_analysis"]["fake_exposure_gap"],
                "avg_gap": results["gap_analysis"]["avg_gap"],
                "max_gap": results["gap_analysis"]["max_gap"],
                "all_gaps_below_target": results["gap_analysis"]["all_gaps_below_target"],
            }
            gap_data = pd.DataFrame([gap_summary])
            gap_path = self.results_dir / f"{self.name}_gaps_{unlearned_name}.csv"
            gap_data.to_csv(gap_path, index=False)

            print(f"✅ Saved effectiveness: {effectiveness_path.name}")
            print(f"✅ Saved gaps: {gap_path.name}")

        # Save metadata
        self.save_metadata(
            {
                "has_unlearned": results["has_unlearned"],
                "unlearned_name": unlearned_name,
                "top_k": results["top_k"],
                "target_gap": results["target_gap"],
            }
        )

        return self.results_dir

    def get_summary(self, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Get human-readable summary of unlearning analysis.

        Args:
            results: Analysis results (uses self.analysis_results if None)

        Returns:
            Formatted summary string
        """
        if results is None:
            results = self.analysis_results

        if results is None:
            return "No analysis results available."

        contamination = results["contamination"]
        exposure = results["exposure_comparison"]
        quality = results["quality_comparison"]
        has_unlearned = results["has_unlearned"]
        k = results["top_k"]

        summary = []
        summary.append("=" * 70)
        summary.append("MACHINE UNLEARNING ANALYSIS")
        summary.append("=" * 70)
        summary.append(f"Target gap threshold: {self.target_gap}")
        summary.append("")

        # Contamination section
        summary.append("🦠 CONTAMINATION (Clean vs Poisoned):")
        summary.append(f"  Clean avg fake news: {contamination['clean_avg_fake']:.2f}")
        summary.append(f"  Poisoned avg fake news: {contamination['poisoned_avg_fake']:.2f}")
        summary.append(f"  Increase: +{contamination['fake_exposure_increase']:.2f}")
        summary.append(f"  Contamination rate: {contamination['contamination_rate']:.1f}%")
        summary.append(
            f"  Users affected increase: +{contamination['users_affected_increase']:.1f}%"
        )
        summary.append("")

        # Exposure comparison
        summary.append(f"📊 FAKE NEWS EXPOSURE (Top-{k}):")
        for _, row in exposure.iterrows():
            summary.append(f"  {row['model'].capitalize()}:")
            summary.append(f"    Avg fake: {row['avg_fake_in_top_k']:.2f}")
            summary.append(f"    Max fake: {row['max_fake_in_top_k']:.0f}")
            summary.append(f"    Users affected: {row['users_with_fake_pct']:.1f}%")
        summary.append("")

        # Quality comparison
        summary.append("🎯 RANKING QUALITY:")
        for _, row in quality.iterrows():
            summary.append(f"  {row['model'].capitalize()}:")
            summary.append(f"    AUC: {row['AUC']:.4f}")
            summary.append(f"    MRR: {row['MRR']:.4f}")
            summary.append(f"    NDCG@10: {row['NDCG@10']:.4f}")
        summary.append("")

        # Unlearning effectiveness
        if has_unlearned:
            effectiveness = results["effectiveness"]
            gap_analysis = results["gap_analysis"]

            summary.append("🔄 UNLEARNING EFFECTIVENESS:")
            summary.append(
                f"  Gap from clean (fake exposure): {effectiveness['gap_from_clean']:.4f}"
            )
            summary.append(f"  Recovery rate: {effectiveness['recovery_rate']:.1f}%")
            summary.append(f"  Status: {effectiveness['status']}")
            summary.append("")

            summary.append("📏 GAP ANALYSIS:")
            summary.append(f"  Fake exposure gap: {gap_analysis['fake_exposure_gap']:.4f}")

            for metric, gaps in gap_analysis["quality_gaps"].items():
                summary.append(f"  {metric} gap: {gaps['gap_from_clean']:.4f}")

            summary.append(f"  Average gap: {gap_analysis['avg_gap']:.4f}")
            summary.append(f"  Maximum gap: {gap_analysis['max_gap']:.4f}")
            summary.append("")

            # Final verdict
            summary.append("⚖️  OVERALL VERDICT:")
            if gap_analysis["all_gaps_below_target"]:
                verdict = f"✅ SUCCESS - All gaps below {self.target_gap}"
            elif gap_analysis["avg_gap"] < self.target_gap:
                verdict = f"✅ GOOD - Average gap below {self.target_gap}"
            elif gap_analysis["avg_gap"] < 0.05:
                verdict = "⚠️  ACCEPTABLE - Gaps within reasonable range"
            elif gap_analysis["avg_gap"] < 1.0:
                verdict = "❌ NEEDS IMPROVEMENT - Significant gaps remain"
            else:
                verdict = "❌ FAILED - Unlearning ineffective"

            summary.append(f"  {verdict}")

            if effectiveness["is_effective"]:
                summary.append(f"  ✅ Model very close to clean baseline")
            else:
                summary.append(f"  ⚠️  Model still differs from clean baseline")
        else:
            summary.append("⏳ UNLEARNING STATUS:")
            summary.append("  Unlearned model not yet available")
            summary.append("")
            summary.append("  Next steps:")
            summary.append("  1. Train poisoned model")
            summary.append("  2. Apply unlearning algorithm")
            summary.append("  3. Re-run this analysis with unlearned model")
            summary.append("")
            summary.append("  Expected outcomes:")
            summary.append(f"    ✅ Gap from clean < {self.target_gap}")
            summary.append("    ✅ Recovery rate > 90%")
            summary.append("    ✅ Quality metrics maintained")

        summary.append("=" * 70)

        return "\n".join(summary)
