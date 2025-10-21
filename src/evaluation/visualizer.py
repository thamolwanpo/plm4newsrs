"""Visualization utilities for evaluation results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


class EvaluationVisualizer:
    """
    Create visualizations for evaluation results.

    Supports:
    - Exposure analysis plots
    - Truth decay visualizations
    - Failure analysis charts
    - Unlearning effectiveness plots
    - Model comparison charts
    """

    def __init__(self, results_dir: Path):
        """
        Initialize visualizer.

        Args:
            results_dir: Directory to save plots
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ========== EXPOSURE ANALYSIS PLOTS ==========

    def plot_exposure_comparison(
        self, exposure_df: pd.DataFrame, top_k: int = 10, filename: str = "exposure_comparison.png"
    ) -> Path:
        """
        Create bar chart comparing fake news exposure across models.

        Args:
            exposure_df: DataFrame with columns ['model', 'avg_fake_in_top_k', ...]
            top_k: Top-k value for title
            filename: Output filename

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Fake News Exposure Analysis (Top-{top_k})", fontsize=16, weight="bold")

        # Plot 1: Average fake news count
        ax1 = axes[0]
        colors = self._get_model_colors(exposure_df["model"].values)

        bars = ax1.bar(
            exposure_df["model"],
            exposure_df["avg_fake_in_top_k"],
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )
        ax1.set_title(f"Average Fake News in Top-{top_k}")
        ax1.set_ylabel("Avg Fake News Count")
        ax1.set_xlabel("Model")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Plot 2: Users affected percentage
        ax2 = axes[1]
        bars = ax2.bar(
            exposure_df["model"],
            exposure_df["users_with_fake_pct"],
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )
        ax2.set_title("Percentage of Users Exposed to Fake News")
        ax2.set_ylabel("Users (%)")
        ax2.set_xlabel("Model")
        ax2.set_ylim([0, 100])
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plot_path = self.results_dir / filename
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    def plot_exposure_heatmap(
        self,
        model_results: Dict[str, pd.DataFrame],
        metrics: List[str],
        top_k: int = 10,
        filename: str = "exposure_heatmap.png",
    ) -> Path:
        """
        Create heatmap of exposure metrics across models and benchmarks.

        Args:
            model_results: Dict of {model_name: results_df}
            metrics: List of metric names to plot
            top_k: Top-k value
            filename: Output filename

        Returns:
            Path to saved plot
        """
        from ..metrics.misinformation import calculate_fake_exposure

        # Collect data
        heatmap_data = []
        for model_name, results_df in model_results.items():
            exposure = calculate_fake_exposure(results_df, k=top_k)
            row_data = {"model": model_name}
            for metric in metrics:
                row_data[metric] = exposure.get(metric, 0)
            heatmap_data.append(row_data)

        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df = heatmap_df.set_index("model")

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            heatmap_df, annot=True, fmt=".2f", cmap="Reds", ax=ax, cbar_kws={"label": "Value"}
        )
        ax.set_title(f"Fake News Exposure Heatmap (Top-{top_k})", fontsize=14, weight="bold")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Model")

        plt.tight_layout()
        plot_path = self.results_dir / filename
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    # ========== TRUTH DECAY PLOTS ==========

    def plot_truth_decay_analysis(
        self,
        rank_distribution_df: pd.DataFrame,
        likelihood_df: pd.DataFrame,
        median_ranks_df: pd.DataFrame,
        filename: str = "truth_decay_analysis.png",
    ) -> Path:
        """
        Create comprehensive truth decay visualization.

        Args:
            rank_distribution_df: Full rank data with 'news_type' and 'rank'
            likelihood_df: Top-k likelihood data
            median_ranks_df: Median ranks by news type
            filename: Output filename

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Truth Decay: Ranking Analysis of Fake vs. Real News", fontsize=16, weight="bold"
        )

        # Plot 1: Box plot of ranks in top-10
        ax1 = axes[0, 0]
        top_10_df = rank_distribution_df[rank_distribution_df["rank"] <= 10]

        sns.boxplot(
            data=top_10_df,
            x="news_type",
            y="rank",
            palette={"Real": "lightgreen", "Fake": "lightcoral"},
            ax=ax1,
        )
        ax1.set_title("Top-10 Rank Distribution")
        ax1.set_ylabel("Rank (1-10, lower is better)")
        ax1.set_xlabel("News Type")
        ax1.grid(axis="y", alpha=0.3)
        ax1.invert_yaxis()  # Lower ranks at top

        # Plot 2: Likelihood of appearing in top-10
        ax2 = axes[0, 1]
        colors = ["skyblue" if nt == "Real" else "salmon" for nt in likelihood_df["news_type"]]

        bars = ax2.bar(
            likelihood_df["news_type"],
            likelihood_df["likelihood_top_10_pct"],
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )
        ax2.set_title("Likelihood of Appearing in Top-10")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_xlabel("News Type")
        ax2.grid(axis="y", alpha=0.3)

        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        # Plot 3: Median rank comparison
        ax3 = axes[1, 0]
        pivot_medians = median_ranks_df.set_index("news_type")["median_rank"]

        bars = ax3.bar(
            pivot_medians.index,
            pivot_medians.values,
            color=["lightgreen", "lightcoral"],
            edgecolor="black",
            alpha=0.8,
        )
        ax3.set_title("Median Ranks Comparison")
        ax3.set_ylabel("Median Rank (Lower is Better)")
        ax3.set_xlabel("News Type")
        ax3.grid(axis="y", alpha=0.3)
        ax3.invert_yaxis()

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="top",
                fontsize=10,
            )

        # Plot 4: Violin plot - full distribution
        ax4 = axes[1, 1]
        sns.violinplot(
            data=rank_distribution_df,
            x="news_type",
            y="rank",
            split=False,
            inner="quart",
            palette={"Real": "lightgreen", "Fake": "lightcoral"},
            ax=ax4,
        )
        ax4.set_title("Full Rank Distribution")
        ax4.set_ylabel("Rank (Lower is Better)")
        ax4.set_xlabel("News Type")
        ax4.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = self.results_dir / filename
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    def plot_rank_comparison(
        self, model_results: Dict[str, pd.DataFrame], filename: str = "rank_comparison.png"
    ) -> Path:
        """
        Compare median ranks across multiple models.

        Args:
            model_results: Dict of {model_name: results_df with 'rank' and 'news_type'}
            filename: Output filename

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(model_results))
        width = 0.35

        real_medians = []
        fake_medians = []

        for model_name, results_df in model_results.items():
            real_median = results_df[results_df["news_type"] == "Real"]["rank"].median()
            fake_median = results_df[results_df["news_type"] == "Fake"]["rank"].median()
            real_medians.append(real_median)
            fake_medians.append(fake_median)

        bars1 = ax.bar(
            x - width / 2,
            real_medians,
            width,
            label="Real News",
            color="lightgreen",
            edgecolor="black",
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            fake_medians,
            width,
            label="Fake News",
            color="lightcoral",
            edgecolor="black",
            alpha=0.8,
        )

        ax.set_xlabel("Model")
        ax.set_ylabel("Median Rank (Lower is Better)")
        ax.set_title("Median Rank Comparison Across Models")
        ax.set_xticks(x)
        ax.set_xticklabels(model_results.keys())
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.invert_yaxis()

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="top",
                    fontsize=9,
                )

        plt.tight_layout()
        plot_path = self.results_dir / filename
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    # ========== FAILURE ANALYSIS PLOTS ==========

    def plot_failure_analysis(
        self, failure_summary_df: pd.DataFrame, filename: str = "failure_analysis.png"
    ) -> Path:
        """
        Create failure analysis visualization.

        Args:
            failure_summary_df: DataFrame with failure statistics by model
            filename: Output filename

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Failure Case Analysis", fontsize=16, weight="bold")

        # Plot 1: Failure rates
        ax1 = axes[0]
        colors = self._get_model_colors(failure_summary_df["model"].values)

        bars = ax1.bar(
            failure_summary_df["model"],
            failure_summary_df["failure_rate_pct"],
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )
        ax1.set_title("Failure Rates by Model")
        ax1.set_ylabel("Failure Rate (%)")
        ax1.set_xlabel("Model")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Plot 2: Fake news involvement in failures
        ax2 = axes[1]
        bars = ax2.bar(
            failure_summary_df["model"],
            failure_summary_df["fake_involvement_pct"],
            color="salmon",
            edgecolor="black",
            alpha=0.8,
        )
        ax2.set_title("Fake News Involvement in Failures")
        ax2.set_ylabel("Fake News Involvement (%)")
        ax2.set_xlabel("Model")
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plot_path = self.results_dir / filename
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    # ========== UNLEARNING ANALYSIS PLOTS ==========

    def plot_unlearning_analysis(
        self,
        exposure_comparison: pd.DataFrame,
        quality_comparison: pd.DataFrame,
        gap_analysis: Optional[Dict[str, Any]] = None,
        target_gap: float = 0.01,
        filename: str = "unlearning_analysis.png",
    ) -> Path:
        """
        Create comprehensive unlearning effectiveness visualization.

        Args:
            exposure_comparison: Exposure metrics for all models
            quality_comparison: Quality metrics for all models
            gap_analysis: Gap from clean analysis (optional)
            target_gap: Target gap threshold
            filename: Output filename

        Returns:
            Path to saved plot
        """
        has_unlearned = "unlearned" in exposure_comparison["model"].values

        if has_unlearned and gap_analysis:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        fig.suptitle("Machine Unlearning Analysis", fontsize=16, weight="bold")
        axes = axes.flatten()

        # Plot 1: Fake news exposure
        ax1 = axes[0]
        colors = self._get_model_colors(exposure_comparison["model"].values)

        bars = ax1.bar(
            exposure_comparison["model"],
            exposure_comparison["avg_fake_in_top_k"],
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )
        ax1.set_title("Average Fake News in Top-K")
        ax1.set_ylabel("Avg Fake News Count")
        ax1.set_xlabel("Model")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Plot 2: Quality metrics (NDCG@10)
        ax2 = axes[1]
        bars = ax2.bar(
            quality_comparison["model"],
            quality_comparison["NDCG@10"],
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )
        ax2.set_title("Ranking Quality (NDCG@10)")
        ax2.set_ylabel("NDCG@10")
        ax2.set_xlabel("Model")
        ax2.set_ylim([0, 1])
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Plot 3: Users affected
        ax3 = axes[2]
        bars = ax3.bar(
            exposure_comparison["model"],
            exposure_comparison["users_with_fake_pct"],
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )
        ax3.set_title("Users Exposed to Fake News")
        ax3.set_ylabel("Users (%)")
        ax3.set_xlabel("Model")
        ax3.set_ylim([0, 100])
        ax3.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Plot 4: Gap from clean (if unlearned exists)
        ax4 = axes[3]

        if has_unlearned and gap_analysis:
            # Calculate gaps
            clean_fake = exposure_comparison[exposure_comparison["model"] == "clean"][
                "avg_fake_in_top_k"
            ].values[0]

            poisoned_fake = exposure_comparison[exposure_comparison["model"] == "poisoned"][
                "avg_fake_in_top_k"
            ].values[0]

            unlearned_fake = exposure_comparison[exposure_comparison["model"] == "unlearned"][
                "avg_fake_in_top_k"
            ].values[0]

            poisoned_gap = poisoned_fake - clean_fake
            unlearned_gap = unlearned_fake - clean_fake

            categories = ["Poisoned\nvs Clean", "Unlearned\nvs Clean"]
            gaps = [poisoned_gap, unlearned_gap]
            colors_gap = ["lightcoral", "lightblue"]

            bars = ax4.bar(categories, gaps, color=colors_gap, edgecolor="black", alpha=0.8)
            ax4.axhline(y=0, color="green", linestyle="--", linewidth=2, label="Clean baseline")
            ax4.axhline(
                y=target_gap,
                color="orange",
                linestyle=":",
                alpha=0.5,
                label=f"Target (|gap| < {target_gap})",
            )
            ax4.axhline(y=-target_gap, color="orange", linestyle=":", alpha=0.5)

            ax4.set_title("Gap from Clean Model\n(Fake News Exposure)")
            ax4.set_ylabel("Gap (Model - Clean)")
            ax4.legend(loc="upper right")
            ax4.grid(axis="y", alpha=0.3)

            # Add value labels with status
            for i, bar in enumerate(bars):
                height = bar.get_height()
                status = "✅" if abs(height) < target_gap else "⚠️"
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:+.3f}\n{status}",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                    fontsize=10,
                )
        else:
            # Multiple metrics comparison (no unlearned)
            metrics = ["AUC", "MRR", "NDCG@10"]
            x = np.arange(len(metrics))
            width = 0.35

            clean_vals = quality_comparison[quality_comparison["model"] == "clean"][metrics].values[
                0
            ]

            poisoned_vals = quality_comparison[quality_comparison["model"] == "poisoned"][
                metrics
            ].values[0]

            bars1 = ax4.bar(
                x - width / 2,
                clean_vals,
                width,
                label="Clean",
                color="lightgreen",
                edgecolor="black",
                alpha=0.8,
            )
            bars2 = ax4.bar(
                x + width / 2,
                poisoned_vals,
                width,
                label="Poisoned",
                color="lightcoral",
                edgecolor="black",
                alpha=0.8,
            )

            ax4.set_title("Quality Metrics Comparison")
            ax4.set_ylabel("Score")
            ax4.set_xlabel("Metric")
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.grid(axis="y", alpha=0.3)
            ax4.set_ylim([0, 1])

        plt.tight_layout()
        plot_path = self.results_dir / filename
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    def plot_gap_analysis(
        self,
        gap_analysis: Dict[str, Any],
        target_gap: float = 0.01,
        filename: str = "gap_analysis.png",
    ) -> Path:
        """
        Create detailed gap analysis visualization.

        Args:
            gap_analysis: Gap analysis results
            target_gap: Target gap threshold
            filename: Output filename

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Gap from Clean Baseline Analysis", fontsize=16, weight="bold")

        # Plot 1: All gaps
        ax1 = axes[0]

        gap_names = ["Fake\nExposure", "AUC", "MRR", "NDCG@10"]
        gap_values = [
            gap_analysis["fake_exposure_gap"],
            gap_analysis["quality_gaps"]["AUC"]["gap_from_clean"],
            gap_analysis["quality_gaps"]["MRR"]["gap_from_clean"],
            gap_analysis["quality_gaps"]["NDCG@10"]["gap_from_clean"],
        ]

        colors = ["red" if g > target_gap else "green" for g in gap_values]
        bars = ax1.bar(gap_names, gap_values, color=colors, edgecolor="black", alpha=0.7)

        ax1.axhline(
            y=target_gap, color="orange", linestyle="--", linewidth=2, label=f"Target: {target_gap}"
        )
        ax1.set_title("Gaps from Clean Model")
        ax1.set_ylabel("Absolute Gap")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, gap_values):
            height = bar.get_height()
            status = "✅" if val < target_gap else "❌"
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.4f}\n{status}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Plot 2: Quality metrics trajectory
        ax2 = axes[1]

        metrics = ["AUC", "MRR", "NDCG@10"]
        clean_vals = [gap_analysis["quality_gaps"][m]["clean"] for m in metrics]
        poisoned_vals = [gap_analysis["quality_gaps"][m]["poisoned"] for m in metrics]
        unlearned_vals = [gap_analysis["quality_gaps"][m]["unlearned"] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.25

        ax2.bar(
            x - width,
            clean_vals,
            width,
            label="Clean",
            color="lightgreen",
            edgecolor="black",
            alpha=0.8,
        )
        ax2.bar(
            x,
            poisoned_vals,
            width,
            label="Poisoned",
            color="lightcoral",
            edgecolor="black",
            alpha=0.8,
        )
        ax2.bar(
            x + width,
            unlearned_vals,
            width,
            label="Unlearned",
            color="lightblue",
            edgecolor="black",
            alpha=0.8,
        )

        ax2.set_title("Quality Metrics: Clean → Poisoned → Unlearned")
        ax2.set_ylabel("Score")
        ax2.set_xlabel("Metric")
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_ylim([0, 1])

        plt.tight_layout()
        plot_path = self.results_dir / filename
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    # ========== GENERAL COMPARISON PLOTS ==========

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str],
        title: str = "Model Comparison",
        filename: str = "model_comparison.png",
    ) -> Path:
        """
        Create multi-metric comparison across models.

        Args:
            comparison_df: DataFrame with model comparisons
            metrics: List of metrics to plot
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved plot
        """
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        fig.suptitle(title, fontsize=16, weight="bold")

        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]

            colors = self._get_model_colors(comparison_df["model"].values)
            bars = ax.bar(
                comparison_df["model"],
                comparison_df[metric],
                color=colors,
                edgecolor="black",
                alpha=0.8,
            )

            ax.set_title(metric)
            ax.set_ylabel("Score")
            ax.set_xlabel("Model")
            ax.grid(axis="y", alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Hide extra subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plot_path = self.results_dir / filename
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    # ========== HELPER METHODS ==========

    def _get_model_colors(self, models: np.ndarray) -> List[str]:
        """
        Get consistent colors for models.

        Args:
            models: Array of model names

        Returns:
            List of colors
        """
        color_map = {
            "clean": "lightgreen",
            "poisoned": "lightcoral",
            "unlearned": "lightblue",
        }

        return [color_map.get(str(m).lower(), "lightgray") for m in models]

    def create_report_summary(
        self, results_dict: Dict[str, Any], filename: str = "evaluation_report.png"
    ) -> Path:
        """
        Create a single comprehensive summary figure.

        Args:
            results_dict: Dictionary with all evaluation results
            filename: Output filename

        Returns:
            Path to saved plot
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        fig.suptitle("Comprehensive Evaluation Report", fontsize=18, weight="bold")

        # Add subplots for each analysis type
        # This is a template - customize based on available results

        # TODO: Add comprehensive summary logic

        plt.tight_layout()
        plot_path = self.results_dir / filename
        plt.savefig(plot_path)
        plt.close()

        return plot_path
