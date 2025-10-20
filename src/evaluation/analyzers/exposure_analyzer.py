"""Analyzer for fake news exposure in recommendations."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base_analyzer import BaseAnalyzer
from ..metrics.misinformation import calculate_fake_exposure, calculate_mc_at_k


class ExposureAnalyzer(BaseAnalyzer):
    """
    Analyze fake news exposure in top-K recommendations.

    Calculates metrics like:
    - Average fake news count in top-k
    - Percentage of users exposed to fake news
    - Misinformation Count (MC@k)
    - Distribution statistics
    """

    def __init__(self, results_dir: Path, name: str = "exposure"):
        """
        Initialize exposure analyzer.

        Args:
            results_dir: Directory to save results
            name: Name for this analyzer
        """
        super().__init__(results_dir, name)

    def get_required_columns(self) -> List[str]:
        """Required columns for exposure analysis."""
        return ["user_id", "score", "is_fake"]

    def analyze(
        self,
        results_df: pd.DataFrame,
        top_k: int = 10,
        calculate_mc: bool = True,
        mc_cutoffs: List[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze fake news exposure in top-K recommendations.

        Args:
            results_df: DataFrame with columns ['user_id', 'score', 'is_fake']
            top_k: Number of top recommendations to analyze
            calculate_mc: Whether to calculate MC@k metrics
            mc_cutoffs: List of k values for MC@k (default: [5, 10, 20])

        Returns:
            Dictionary with exposure statistics:
            {
                'top_k': k value used,
                'exposure': {exposure statistics},
                'mc_metrics': {MC@k for different k values},  # if calculate_mc
                'per_user_stats': DataFrame with per-user statistics
            }
        """
        self.validate_data(results_df)

        print(f"Analyzing fake news exposure (top-{top_k})...")

        # Calculate exposure statistics
        exposure_stats = calculate_fake_exposure(results_df, k=top_k)

        # Calculate MC@k for multiple cutoffs
        mc_metrics = {}
        if calculate_mc:
            if mc_cutoffs is None:
                mc_cutoffs = [5, 10, 20]

            for k in mc_cutoffs:
                mc_metrics[f"MC@{k}"] = calculate_mc_at_k(results_df, k=k)

        # Per-user statistics for detailed analysis
        per_user_stats = self._calculate_per_user_stats(results_df, top_k)

        results = {
            "top_k": top_k,
            "exposure": exposure_stats,
            "mc_metrics": mc_metrics,
            "per_user_stats": per_user_stats,
            "metadata": {
                "num_users": results_df["user_id"].nunique(),
                "total_interactions": len(results_df),
                "fake_news_ratio": results_df["is_fake"].mean(),
            },
        }

        self.analysis_results = results
        return results

    def _calculate_per_user_stats(self, results_df: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Calculate per-user exposure statistics.

        Args:
            results_df: Results dataframe
            k: Top-k cutoff

        Returns:
            DataFrame with per-user statistics
        """
        user_stats = []

        for user_id, group in results_df.groupby("user_id"):
            # Get top-k recommendations
            top_k = group.nlargest(k, "score")

            # Calculate statistics
            fake_count = top_k["is_fake"].sum()
            fake_ratio = fake_count / k

            # Find minimum rank of fake news
            sorted_group = group.sort_values("score", ascending=False).reset_index(drop=True)
            fake_items = sorted_group[sorted_group["is_fake"] == 1]
            min_fake_rank = fake_items.index[0] + 1 if len(fake_items) > 0 else None

            user_stats.append(
                {
                    "user_id": user_id,
                    "fake_count": fake_count,
                    "fake_ratio": fake_ratio,
                    "has_fake": fake_count > 0,
                    "min_fake_rank": min_fake_rank,
                    "total_items": len(group),
                }
            )

        return pd.DataFrame(user_stats)

    def save_results(self, results: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save exposure analysis results.

        Args:
            results: Analysis results (uses self.analysis_results if None)

        Returns:
            Path to main results CSV
        """
        if results is None:
            results = self.analysis_results

        if results is None:
            raise ValueError("No results to save. Run analyze() first.")

        # Save summary statistics
        summary_data = {"metric": [], "value": []}

        # Add exposure metrics
        for key, value in results["exposure"].items():
            summary_data["metric"].append(key)
            summary_data["value"].append(value)

        # Add MC metrics
        for key, value in results["mc_metrics"].items():
            summary_data["metric"].append(key)
            summary_data["value"].append(value)

        summary_df = pd.DataFrame(summary_data)
        summary_path = self.results_dir / f"{self.name}_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        # Save per-user statistics
        per_user_path = self.results_dir / f"{self.name}_per_user.csv"
        results["per_user_stats"].to_csv(per_user_path, index=False)

        # Save metadata
        self.save_metadata(
            {
                "top_k": results["top_k"],
                "num_users": results["metadata"]["num_users"],
                "total_interactions": results["metadata"]["total_interactions"],
                "fake_news_ratio": results["metadata"]["fake_news_ratio"],
            }
        )

        print(f"✅ Saved summary: {summary_path.name}")
        print(f"✅ Saved per-user stats: {per_user_path.name}")

        return summary_path

    def get_summary(self, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Get human-readable summary of exposure analysis.

        Args:
            results: Analysis results (uses self.analysis_results if None)

        Returns:
            Formatted summary string
        """
        if results is None:
            results = self.analysis_results

        if results is None:
            return "No analysis results available."

        exposure = results["exposure"]
        mc = results["mc_metrics"]
        meta = results["metadata"]
        k = results["top_k"]

        summary = []
        summary.append("=" * 70)
        summary.append(f"FAKE NEWS EXPOSURE ANALYSIS (Top-{k})")
        summary.append("=" * 70)
        summary.append(
            f"Dataset: {meta['num_users']} users, {meta['total_interactions']} interactions"
        )
        summary.append(f"Fake news ratio in data: {meta['fake_news_ratio']:.2%}")
        summary.append("")

        summary.append("📊 EXPOSURE METRICS:")
        summary.append(f"  Average fake news in top-{k}: {exposure['avg_fake_count']:.2f}")
        summary.append(f"  Maximum fake news in top-{k}: {exposure['max_fake_count']:.0f}")
        summary.append(f"  Standard deviation: {exposure['std_fake_count']:.2f}")
        summary.append(f"  Users exposed to fake news: {exposure['users_with_fake_pct']:.1f}%")
        summary.append(f"  Average fake news ratio: {exposure['avg_fake_ratio']:.2%}")
        summary.append(f"  Average min rank of fake: {exposure['avg_min_fake_rank']:.1f}")
        summary.append("")

        if mc:
            summary.append("📈 MISINFORMATION COUNT (MC@k):")
            for metric, value in mc.items():
                summary.append(f"  {metric}: {value:.4f}")
            summary.append("")

        # Risk assessment
        summary.append("⚠️  RISK ASSESSMENT:")
        avg_fake = exposure["avg_fake_count"]
        if avg_fake < 0.5:
            risk = "LOW - Minimal fake news exposure"
        elif avg_fake < 1.0:
            risk = "MODERATE - Some fake news in recommendations"
        elif avg_fake < 2.0:
            risk = "HIGH - Significant fake news exposure"
        else:
            risk = "CRITICAL - Severe fake news contamination"
        summary.append(f"  {risk}")

        summary.append("=" * 70)

        return "\n".join(summary)

    def compare_models(
        self, model_results: Dict[str, pd.DataFrame], top_k: int = 10
    ) -> pd.DataFrame:
        """
        Compare exposure across multiple models.

        Args:
            model_results: Dict mapping model_name -> results_df
            top_k: Top-k cutoff

        Returns:
            Comparison DataFrame

        Example:
            >>> analyzer = ExposureAnalyzer(results_dir)
            >>> comparison = analyzer.compare_models({
            ...     'clean': clean_df,
            ...     'poisoned': poisoned_df,
            ...     'unlearned': unlearned_df
            ... }, top_k=10)
        """
        comparison_data = []

        for model_name, results_df in model_results.items():
            exposure = calculate_fake_exposure(results_df, k=top_k)
            mc_10 = calculate_mc_at_k(results_df, k=10)

            comparison_data.append(
                {
                    "model": model_name,
                    "avg_fake_in_top_k": exposure["avg_fake_count"],
                    "max_fake_in_top_k": exposure["max_fake_count"],
                    "users_with_fake_pct": exposure["users_with_fake_pct"],
                    "avg_fake_ratio": exposure["avg_fake_ratio"],
                    f"MC@{top_k}": mc_10,
                }
            )

        comparison_df = pd.DataFrame(comparison_data)

        # Save comparison
        comparison_path = self.results_dir / f"{self.name}_model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"✅ Saved model comparison: {comparison_path.name}")

        return comparison_df
