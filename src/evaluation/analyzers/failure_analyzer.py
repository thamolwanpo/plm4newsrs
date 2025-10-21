"""Analyzer for ranking failure cases."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .base_analyzer import BaseAnalyzer
from ..metrics.utility import safe_divide


class FailureAnalyzer(BaseAnalyzer):
    """
    Analyze cases where models ranked negative items higher than positive ones.

    A failure occurs when the highest-scored negative item has a higher
    score than the highest-scored positive item for a user.

    Metrics:
    - Overall failure rate
    - Fake news involvement in failures
    - Failure patterns by user segments
    - Margin of failure (how wrong the model was)
    """

    def __init__(self, results_dir: Path, name: str = "failure"):
        """
        Initialize failure analyzer.

        Args:
            results_dir: Directory to save results
            name: Name for this analyzer
        """
        super().__init__(results_dir, name)

    def get_required_columns(self) -> List[str]:
        """Required columns for failure analysis."""
        return ["user_id", "news_id", "score", "label", "is_fake"]

    def analyze(
        self,
        results_df: pd.DataFrame,
        analyze_margins: bool = True,
        segment_by_history_length: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze ranking failure cases.

        Args:
            results_df: DataFrame with columns ['user_id', 'score', 'label', 'is_fake']
            analyze_margins: Whether to calculate failure margins
            segment_by_history_length: Whether to analyze by user history length

        Returns:
            Dictionary with failure metrics:
            {
                'overall': Overall failure statistics,
                'fake_involvement': Fake news role in failures,
                'failure_details': Per-user failure information,
                'margins': Failure margin statistics (if analyze_margins),
                'segments': Failure rates by segment (if segment_by_history_length)
            }
        """
        self.validate_data(results_df)

        print("Analyzing ranking failures...")

        # Core failure analysis
        failure_stats, failure_details = self._analyze_failures(results_df)

        # Fake news involvement
        fake_involvement = self._analyze_fake_involvement(failure_details)

        results = {
            "overall": failure_stats,
            "fake_involvement": fake_involvement,
            "failure_details": failure_details,
            "metadata": {
                "num_users": results_df["user_id"].nunique(),
                "total_interactions": len(results_df),
            },
        }

        # Optional: Analyze margins
        if analyze_margins:
            margins = self._analyze_margins(failure_details)
            results["margins"] = margins

        # Optional: Segment analysis
        if segment_by_history_length:
            # Would need history_length column in results_df
            if "history_length" in results_df.columns:
                segments = self._analyze_by_segments(results_df, failure_details)
                results["segments"] = segments

        self.analysis_results = results
        return results

    def _analyze_failures(self, results_df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Identify and analyze failure cases.

        Args:
            results_df: Results dataframe

        Returns:
            Tuple of (overall_stats, failure_details_df)
        """
        failure_rows = []

        total_interactions = 0
        total_failures = 0
        failures_with_fake = 0

        for user_id, group in results_df.groupby("user_id"):
            total_interactions += 1

            # Separate positive and negative items
            positive_items = group[group["label"] == 1]
            negative_items = group[group["label"] == 0]

            # Skip if no positive items
            if positive_items.empty:
                continue

            # Get highest positive score
            max_positive_score = positive_items["score"].max()
            max_positive_idx = positive_items["score"].idxmax()
            top_correct_item = positive_items.loc[max_positive_idx]  # ← MOVE THIS UP

            if not negative_items.empty:
                max_negative_score = negative_items["score"].max()
                max_negative_idx = negative_items["score"].idxmax()

                # Check for failure (negative ranked higher than positive)
                if max_negative_score > max_positive_score:
                    total_failures += 1

                    # Get the top wrong item details
                    top_wrong_item = negative_items.loc[max_negative_idx]

                    # Check if fake news was involved
                    is_fake_involved = bool(top_wrong_item["is_fake"])
                    if is_fake_involved:
                        failures_with_fake += 1

                    # Calculate margin
                    margin = max_negative_score - max_positive_score

                    failure_rows.append(
                        {
                            "user_id": user_id,
                            "failure": True,
                            "max_positive_score": max_positive_score,
                            "max_negative_score": max_negative_score,
                            "margin": margin,
                            "top_wrong_is_fake": is_fake_involved,
                            "top_wrong_news_id": top_wrong_item["news_id"],
                            "top_correct_news_id": top_correct_item["news_id"],
                            "num_positives": len(positive_items),
                            "num_negatives": len(negative_items),
                        }
                    )
                else:
                    # No failure
                    failure_rows.append(
                        {
                            "user_id": user_id,
                            "failure": False,
                            "max_positive_score": max_positive_score,
                            "max_negative_score": max_negative_score,
                            "margin": max_positive_score - max_negative_score,
                            "top_wrong_is_fake": False,
                            "top_wrong_news_id": None,
                            "top_correct_news_id": top_correct_item["news_id"],
                            "num_positives": len(positive_items),
                            "num_negatives": len(negative_items),
                        }
                    )
            else:
                # No negative items - can't have failure, but record it
                failure_rows.append(
                    {
                        "user_id": user_id,
                        "failure": False,
                        "max_positive_score": max_positive_score,
                        "max_negative_score": None,  # ← No negatives
                        "margin": None,
                        "top_wrong_is_fake": False,
                        "top_wrong_news_id": None,
                        "top_correct_news_id": top_correct_item["news_id"],
                        "num_positives": len(positive_items),
                        "num_negatives": 0,
                    }
                )

        # Overall statistics
        failure_rate = safe_divide(total_failures, total_interactions, default=0.0) * 100
        fake_involvement_rate = safe_divide(failures_with_fake, total_failures, default=0.0) * 100

        overall_stats = {
            "total_interactions": total_interactions,
            "total_failures": total_failures,
            "failure_rate_pct": failure_rate,
            "failures_with_fake": failures_with_fake,
            "fake_involvement_pct": fake_involvement_rate,
        }

        failure_details = pd.DataFrame(failure_rows)

        return overall_stats, failure_details

    def _analyze_fake_involvement(self, failure_details: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze fake news involvement in failures.

        Args:
            failure_details: DataFrame with failure information

        Returns:
            Dictionary with fake news involvement stats
        """
        failures_only = failure_details[failure_details["failure"] == True]

        if len(failures_only) == 0:
            return {
                "total_failures": 0,
                "fake_involved": 0,
                "fake_involvement_pct": 0.0,
                "avg_margin_with_fake": 0.0,
                "avg_margin_without_fake": 0.0,
            }

        fake_involved = failures_only["top_wrong_is_fake"].sum()
        fake_involvement_pct = safe_divide(fake_involved, len(failures_only), default=0.0) * 100

        # Margin analysis
        failures_with_fake = failures_only[failures_only["top_wrong_is_fake"] == True]
        failures_without_fake = failures_only[failures_only["top_wrong_is_fake"] == False]

        avg_margin_with_fake = (
            failures_with_fake["margin"].mean() if len(failures_with_fake) > 0 else 0.0
        )
        avg_margin_without_fake = (
            failures_without_fake["margin"].mean() if len(failures_without_fake) > 0 else 0.0
        )

        return {
            "total_failures": len(failures_only),
            "fake_involved": int(fake_involved),
            "fake_involvement_pct": fake_involvement_pct,
            "avg_margin_with_fake": avg_margin_with_fake,
            "avg_margin_without_fake": avg_margin_without_fake,
            "margin_difference": avg_margin_with_fake - avg_margin_without_fake,
        }

    def _analyze_margins(self, failure_details: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze failure margins (how wrong was the model).

        Args:
            failure_details: DataFrame with failure information

        Returns:
            Dictionary with margin statistics
        """
        failures_only = failure_details[failure_details["failure"] == True]

        if len(failures_only) == 0:
            return {"count": 0}

        margins = failures_only["margin"].values

        return {
            "count": len(margins),
            "mean": float(np.mean(margins)),
            "median": float(np.median(margins)),
            "std": float(np.std(margins)),
            "min": float(np.min(margins)),
            "max": float(np.max(margins)),
            "q25": float(np.percentile(margins, 25)),
            "q75": float(np.percentile(margins, 75)),
        }

    def _analyze_by_segments(
        self, results_df: pd.DataFrame, failure_details: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze failure rates by user segments.

        Args:
            results_df: Full results with segment information
            failure_details: Failure details

        Returns:
            DataFrame with failure rates by segment
        """
        # Create segments (example: by number of items)
        segment_data = []

        for user_id, group in results_df.groupby("user_id"):
            num_items = len(group)

            # Determine segment
            if num_items <= 5:
                segment = "small (≤5)"
            elif num_items <= 10:
                segment = "medium (6-10)"
            else:
                segment = "large (>10)"

            # Check if failed
            user_failure = (
                failure_details[failure_details["user_id"] == user_id]["failure"].any()
                if user_id in failure_details["user_id"].values
                else False
            )

            segment_data.append(
                {
                    "user_id": user_id,
                    "segment": segment,
                    "num_items": num_items,
                    "failed": user_failure,
                }
            )

        segment_df = pd.DataFrame(segment_data)

        # Calculate failure rates by segment
        segment_summary = []
        for segment in segment_df["segment"].unique():
            segment_users = segment_df[segment_df["segment"] == segment]
            failures = segment_users["failed"].sum()
            total = len(segment_users)
            failure_rate = safe_divide(failures, total, default=0.0) * 100

            segment_summary.append(
                {
                    "segment": segment,
                    "total_users": total,
                    "failures": failures,
                    "failure_rate_pct": failure_rate,
                }
            )

        return pd.DataFrame(segment_summary)

    def save_results(self, results: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save failure analysis results.

        Args:
            results: Analysis results (uses self.analysis_results if None)

        Returns:
            Path to main results file
        """
        if results is None:
            results = self.analysis_results

        if results is None:
            raise ValueError("No results to save. Run analyze() first.")

        # Save overall summary
        summary_data = []
        for key, value in results["overall"].items():
            summary_data.append({"metric": key, "value": value})

        for key, value in results["fake_involvement"].items():
            summary_data.append({"metric": f"fake_{key}", "value": value})

        summary_df = pd.DataFrame(summary_data)
        summary_path = self.results_dir / f"{self.name}_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        # Save failure details
        details_path = self.results_dir / f"{self.name}_details.csv"
        results["failure_details"].to_csv(details_path, index=False)

        # Save margins if available
        if "margins" in results:
            margins_data = pd.DataFrame([results["margins"]])
            margins_path = self.results_dir / f"{self.name}_margins.csv"
            margins_data.to_csv(margins_path, index=False)
            print(f"✅ Saved margins: {margins_path.name}")

        # Save segments if available
        if "segments" in results:
            segments_path = self.results_dir / f"{self.name}_segments.csv"
            results["segments"].to_csv(segments_path, index=False)
            print(f"✅ Saved segments: {segments_path.name}")

        # Save metadata
        self.save_metadata(results["metadata"])

        print(f"✅ Saved summary: {summary_path.name}")
        print(f"✅ Saved details: {details_path.name}")

        return summary_path

    def get_summary(self, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Get human-readable summary of failure analysis.

        Args:
            results: Analysis results (uses self.analysis_results if None)

        Returns:
            Formatted summary string
        """
        if results is None:
            results = self.analysis_results

        if results is None:
            return "No analysis results available."

        overall = results["overall"]
        fake = results["fake_involvement"]
        margins = results.get("margins")

        summary = []
        summary.append("=" * 70)
        summary.append("FAILURE CASE ANALYSIS")
        summary.append("=" * 70)
        summary.append(f"Total interactions: {overall['total_interactions']}")
        summary.append("")

        summary.append("❌ FAILURE STATISTICS:")
        summary.append(f"  Total failures: {overall['total_failures']}")
        summary.append(f"  Failure rate: {overall['failure_rate_pct']:.2f}%")
        summary.append("")

        summary.append("🎭 FAKE NEWS INVOLVEMENT:")
        summary.append(
            f"  Failures with fake news: {fake['fake_involved']}/{fake['total_failures']}"
        )
        summary.append(f"  Fake involvement rate: {fake['fake_involvement_pct']:.2f}%")

        if fake["fake_involved"] > 0:
            summary.append(f"  Avg margin (with fake): {fake['avg_margin_with_fake']:.4f}")
            summary.append(f"  Avg margin (without fake): {fake['avg_margin_without_fake']:.4f}")

            if fake["margin_difference"] > 0:
                summary.append(
                    f"  ⚠️  Failures with fake news have {fake['margin_difference']:.4f} larger margin"
                )
        summary.append("")

        if margins:
            summary.append("📊 FAILURE MARGINS:")
            summary.append(f"  Mean: {margins['mean']:.4f}")
            summary.append(f"  Median: {margins['median']:.4f}")
            summary.append(f"  Range: [{margins['min']:.4f}, {margins['max']:.4f}]")
            summary.append(f"  IQR: [{margins['q25']:.4f}, {margins['q75']:.4f}]")
            summary.append("")

        # Assessment
        summary.append("⚖️  ASSESSMENT:")
        failure_rate = overall["failure_rate_pct"]
        fake_involvement = fake["fake_involvement_pct"]

        if failure_rate < 5:
            quality = "EXCELLENT - Very low failure rate"
        elif failure_rate < 10:
            quality = "GOOD - Acceptable failure rate"
        elif failure_rate < 20:
            quality = "MODERATE - Noticeable failures"
        else:
            quality = "POOR - High failure rate"

        summary.append(f"  Model quality: {quality}")

        if fake_involvement > 50:
            summary.append(f"  ⚠️  CRITICAL: {fake_involvement:.1f}% of failures involve fake news")
        elif fake_involvement > 30:
            summary.append(f"  ⚠️  WARNING: {fake_involvement:.1f}% of failures involve fake news")

        summary.append("=" * 70)

        return "\n".join(summary)

    def compare_models(self, model_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compare failure rates across multiple models.

        Args:
            model_results: Dict mapping model_name -> results_df

        Returns:
            Comparison DataFrame
        """
        comparison_data = []

        for model_name, results_df in model_results.items():
            failure_stats, failure_details = self._analyze_failures(results_df)
            fake_involvement = self._analyze_fake_involvement(failure_details)

            comparison_data.append(
                {
                    "model": model_name,
                    "failure_rate_pct": failure_stats["failure_rate_pct"],
                    "total_failures": failure_stats["total_failures"],
                    "fake_involvement_pct": fake_involvement["fake_involvement_pct"],
                    "failures_with_fake": fake_involvement["fake_involved"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)

        # Save comparison
        comparison_path = self.results_dir / f"{self.name}_model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"✅ Saved model comparison: {comparison_path.name}")

        return comparison_df
