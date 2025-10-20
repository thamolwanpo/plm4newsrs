"""Analyzer for truth decay: ranking patterns of fake vs real news."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base_analyzer import BaseAnalyzer
from ..metrics.misinformation import calculate_truth_decay_metrics
from ..metrics.utility import safe_divide


class TruthDecayAnalyzer(BaseAnalyzer):
    """
    Analyze truth decay: ranking patterns of fake vs real news.

    Truth decay refers to the phenomenon where fake news is ranked
    higher than real news, indicating model contamination.

    Metrics:
    - Rank distributions for fake vs real news
    - Likelihood of appearing in top-K
    - Median ranks comparison
    - Statistical significance tests
    """

    def __init__(self, results_dir: Path, name: str = "truth_decay"):
        """
        Initialize truth decay analyzer.

        Args:
            results_dir: Directory to save results
            name: Name for this analyzer
        """
        super().__init__(results_dir, name)

    def get_required_columns(self) -> List[str]:
        """Required columns for truth decay analysis."""
        return ["user_id", "score", "is_fake", "label"]

    def analyze(self, results_df: pd.DataFrame, top_k: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Analyze truth decay: ranking patterns of fake vs real news.

        Args:
            results_df: DataFrame with columns ['user_id', 'score', 'is_fake', 'label']
            top_k: Cutoff for top-K likelihood analysis

        Returns:
            Dictionary with truth decay metrics:
            {
                'rank_distribution': Full rank data with news types,
                'top_k_likelihood': Likelihood of appearing in top-K,
                'median_ranks': Median ranks by news type,
                'rank_statistics': Statistical summary,
                'top_k_analysis': Detailed top-K breakdown
            }
        """
        self.validate_data(results_df)

        print(f"Analyzing truth decay patterns (top-{top_k})...")

        # Add ranks within each user group
        results_df = results_df.copy()
        results_df["rank"] = results_df.groupby("user_id")["score"].rank(
            method="first", ascending=False
        )
        results_df["news_type"] = results_df["is_fake"].apply(lambda x: "Fake" if x else "Real")

        # Calculate core metrics using metric functions
        truth_decay_metrics = calculate_truth_decay_metrics(results_df)

        # Additional statistics
        rank_stats = self._calculate_rank_statistics(results_df)
        top_k_analysis = self._analyze_top_k(results_df, top_k)

        results = {
            "rank_distribution": truth_decay_metrics["rank_distribution"],
            "top_k_likelihood": truth_decay_metrics["top_k_likelihood"],
            "median_ranks": truth_decay_metrics["median_ranks"],
            "rank_statistics": rank_stats,
            "top_k_analysis": top_k_analysis,
            "top_k": top_k,
            "metadata": {
                "num_users": results_df["user_id"].nunique(),
                "total_items": len(results_df),
                "fake_ratio": results_df["is_fake"].mean(),
                "real_ratio": 1 - results_df["is_fake"].mean(),
            },
        }

        self.analysis_results = results
        return results

    def _calculate_rank_statistics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistical summary of ranks by news type.

        Args:
            results_df: Results with 'rank' and 'news_type' columns

        Returns:
            DataFrame with statistics (mean, median, std, min, max) by news type
        """
        stats_data = []

        for news_type in ["Real", "Fake"]:
            subset = results_df[results_df["news_type"] == news_type]
            ranks = subset["rank"].values

            stats_data.append(
                {
                    "news_type": news_type,
                    "mean_rank": np.mean(ranks),
                    "median_rank": np.median(ranks),
                    "std_rank": np.std(ranks),
                    "min_rank": np.min(ranks),
                    "max_rank": np.max(ranks),
                    "q25_rank": np.percentile(ranks, 25),
                    "q75_rank": np.percentile(ranks, 75),
                    "count": len(ranks),
                }
            )

        return pd.DataFrame(stats_data)

    def _analyze_top_k(self, results_df: pd.DataFrame, k: int) -> Dict[str, Any]:
        """
        Detailed analysis of top-K recommendations.

        Args:
            results_df: Results with ranks
            k: Top-K cutoff

        Returns:
            Dictionary with top-K analysis
        """
        top_k_df = results_df[results_df["rank"] <= k].copy()

        # Count by news type in top-K
        type_counts = top_k_df["news_type"].value_counts()

        # Average rank in top-K by type
        avg_rank_in_topk = top_k_df.groupby("news_type")["rank"].mean()

        # Percentage of each type that appears in top-K
        pct_in_topk = {}
        for news_type in ["Real", "Fake"]:
            total_of_type = (results_df["news_type"] == news_type).sum()
            in_topk = (top_k_df["news_type"] == news_type).sum()
            pct_in_topk[news_type] = safe_divide(in_topk, total_of_type, default=0.0) * 100

        return {
            "counts_in_top_k": type_counts.to_dict(),
            "avg_rank_in_top_k": avg_rank_in_topk.to_dict(),
            "pct_appearing_in_top_k": pct_in_topk,
            "total_top_k_items": len(top_k_df),
        }

    def save_results(self, results: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save truth decay analysis results.

        Args:
            results: Analysis results (uses self.analysis_results if None)

        Returns:
            Path to main results directory
        """
        if results is None:
            results = self.analysis_results

        if results is None:
            raise ValueError("No results to save. Run analyze() first.")

        # Save likelihood data
        likelihood_path = self.results_dir / f"{self.name}_likelihood.csv"
        results["top_k_likelihood"].to_csv(likelihood_path, index=False)

        # Save median ranks
        medians_path = self.results_dir / f"{self.name}_medians.csv"
        results["median_ranks"].to_csv(medians_path, index=False)

        # Save rank statistics
        stats_path = self.results_dir / f"{self.name}_statistics.csv"
        results["rank_statistics"].to_csv(stats_path, index=False)

        # Save full rank distribution (sampled if too large)
        rank_dist = results["rank_distribution"]
        if len(rank_dist) > 10000:
            rank_dist = rank_dist.sample(n=10000, random_state=42)
        rank_dist_path = self.results_dir / f"{self.name}_rank_distribution.csv"
        rank_dist.to_csv(rank_dist_path, index=False)

        # Save metadata
        self.save_metadata(
            {
                "top_k": results["top_k"],
                "num_users": results["metadata"]["num_users"],
                "total_items": results["metadata"]["total_items"],
                "fake_ratio": results["metadata"]["fake_ratio"],
            }
        )

        print(f"✅ Saved likelihood: {likelihood_path.name}")
        print(f"✅ Saved medians: {medians_path.name}")
        print(f"✅ Saved statistics: {stats_path.name}")
        print(f"✅ Saved rank distribution: {rank_dist_path.name}")

        return self.results_dir

    def get_summary(self, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Get human-readable summary of truth decay analysis.

        Args:
            results: Analysis results (uses self.analysis_results if None)

        Returns:
            Formatted summary string
        """
        if results is None:
            results = self.analysis_results

        if results is None:
            return "No analysis results available."

        likelihood = results["top_k_likelihood"]
        medians = results["median_ranks"]
        stats = results["rank_statistics"]
        top_k_analysis = results["top_k_analysis"]
        k = results["top_k"]
        meta = results["metadata"]

        summary = []
        summary.append("=" * 70)
        summary.append("TRUTH DECAY ANALYSIS")
        summary.append("=" * 70)
        summary.append(f"Dataset: {meta['num_users']} users, {meta['total_items']} items")
        summary.append(f"Composition: {meta['real_ratio']:.1%} real, {meta['fake_ratio']:.1%} fake")
        summary.append("")

        # Median ranks comparison
        summary.append("📊 MEDIAN RANKS (lower is better):")
        for _, row in medians.iterrows():
            summary.append(f"  {row['news_type']}: {row['median_rank']:.1f}")

        real_median = medians[medians["news_type"] == "Real"]["median_rank"].values[0]
        fake_median = medians[medians["news_type"] == "Fake"]["median_rank"].values[0]

        if fake_median < real_median:
            diff = real_median - fake_median
            summary.append(f"  ⚠️  WARNING: Fake news ranked {diff:.1f} positions higher than real!")
        else:
            diff = fake_median - real_median
            summary.append(f"  ✅ Real news ranked {diff:.1f} positions higher than fake")
        summary.append("")

        # Top-K likelihood
        summary.append(f"📈 LIKELIHOOD OF APPEARING IN TOP-{k}:")
        for _, row in likelihood.iterrows():
            summary.append(f"  {row['news_type']}: {row['likelihood_top_10_pct']:.1f}%")
        summary.append("")

        # Rank statistics
        summary.append("📉 RANK STATISTICS:")
        for _, row in stats.iterrows():
            summary.append(f"  {row['news_type']}:")
            summary.append(f"    Mean rank: {row['mean_rank']:.2f}")
            summary.append(f"    Std dev: {row['std_rank']:.2f}")
            summary.append(f"    Range: [{row['min_rank']:.0f}, {row['max_rank']:.0f}]")
            summary.append(f"    IQR: [{row['q25_rank']:.1f}, {row['q75_rank']:.1f}]")
        summary.append("")

        # Top-K breakdown
        summary.append(f"🔝 TOP-{k} BREAKDOWN:")
        counts = top_k_analysis["counts_in_top_k"]
        pcts = top_k_analysis["pct_appearing_in_top_k"]
        for news_type in ["Real", "Fake"]:
            count = counts.get(news_type, 0)
            pct = pcts.get(news_type, 0.0)
            summary.append(f"  {news_type}: {count} items ({pct:.1f}% of all {news_type} news)")
        summary.append("")

        # Truth decay assessment
        summary.append("⚖️  TRUTH DECAY ASSESSMENT:")
        fake_likelihood = likelihood[likelihood["news_type"] == "Fake"][
            "likelihood_top_10_pct"
        ].values[0]

        if fake_median < real_median and fake_likelihood > 50:
            assessment = "SEVERE - Fake news systematically ranked higher"
        elif fake_median < real_median:
            assessment = "MODERATE - Some bias toward fake news"
        elif fake_likelihood > 30:
            assessment = "MILD - Fake news appears frequently in top results"
        else:
            assessment = "MINIMAL - Real news generally ranked higher"

        summary.append(f"  {assessment}")
        summary.append("=" * 70)

        return "\n".join(summary)

    def compare_models(
        self, model_results: Dict[str, pd.DataFrame], top_k: int = 10
    ) -> pd.DataFrame:
        """
        Compare truth decay across multiple models.

        Args:
            model_results: Dict mapping model_name -> results_df
            top_k: Top-k cutoff

        Returns:
            Comparison DataFrame
        """
        comparison_data = []

        for model_name, results_df in model_results.items():
            # Run analysis
            results_df = results_df.copy()
            results_df["rank"] = results_df.groupby("user_id")["score"].rank(
                method="first", ascending=False
            )
            results_df["news_type"] = results_df["is_fake"].apply(lambda x: "Fake" if x else "Real")

            metrics = calculate_truth_decay_metrics(results_df)

            # Extract key metrics
            real_median = metrics["median_ranks"][metrics["median_ranks"]["news_type"] == "Real"][
                "median_rank"
            ].values[0]

            fake_median = metrics["median_ranks"][metrics["median_ranks"]["news_type"] == "Fake"][
                "median_rank"
            ].values[0]

            real_likelihood = metrics["top_k_likelihood"][
                metrics["top_k_likelihood"]["news_type"] == "Real"
            ]["likelihood_top_10_pct"].values[0]

            fake_likelihood = metrics["top_k_likelihood"][
                metrics["top_k_likelihood"]["news_type"] == "Fake"
            ]["likelihood_top_10_pct"].values[0]

            comparison_data.append(
                {
                    "model": model_name,
                    "real_median_rank": real_median,
                    "fake_median_rank": fake_median,
                    "rank_gap": real_median - fake_median,  # Positive = fake ranked higher
                    "real_likelihood_top10": real_likelihood,
                    "fake_likelihood_top10": fake_likelihood,
                    "likelihood_gap": fake_likelihood - real_likelihood,
                }
            )

        comparison_df = pd.DataFrame(comparison_data)

        # Save comparison
        comparison_path = self.results_dir / f"{self.name}_model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"✅ Saved model comparison: {comparison_path.name}")

        return comparison_df
