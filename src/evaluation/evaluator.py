"""Main evaluator orchestrating the evaluation pipeline."""

import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from configs import ModelConfig
from src.models.registry import get_model_class
from src.utils.seed import set_seed

from .benchmark_dataset import BenchmarkDataset, benchmark_collate_fn
from .analyzers import ExposureAnalyzer, TruthDecayAnalyzer, FailureAnalyzer, UnlearningAnalyzer
from .visualizer import EvaluationVisualizer
from .metrics.ranking import (
    calculate_auc,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)


class ModelEvaluator:
    """
    Main evaluator for news recommendation models.

    Orchestrates:
    - Model evaluation on benchmarks
    - Multiple analysis types (exposure, truth decay, failures, unlearning)
    - Visualization generation
    - Results aggregation
    """

    def __init__(
        self,
        config: ModelConfig,
        device: Optional[torch.device] = None,
        enable_analyses: Optional[List[str]] = None,
    ):
        """
        Initialize model evaluator.

        Args:
            config: Model configuration
            device: Device to run on (auto-detect if None)
            enable_analyses: List of analyses to run
                Options: ['exposure', 'truth_decay', 'failure', 'unlearning']
                If None, runs all applicable analyses
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get paths
        self.paths = config.get_paths()
        self.results_dir = self.paths["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analyzers
        self.analyzers = self._initialize_analyzers(enable_analyses)

        # Initialize visualizer
        self.visualizer = EvaluationVisualizer(self.results_dir)

        # Cache for results
        self.evaluation_results = {}
        self.detailed_results = {}

        print(f"\n{'='*70}")
        print(f"MODEL EVALUATOR INITIALIZED")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Results directory: {self.results_dir}")
        print(f"Enabled analyses: {list(self.analyzers.keys())}")
        print(f"{'='*70}\n")

    def _initialize_analyzers(self, enable_analyses: Optional[List[str]]) -> Dict:
        """Initialize enabled analyzers."""
        available_analyzers = {
            "exposure": ExposureAnalyzer,
            "truth_decay": TruthDecayAnalyzer,
            "failure": FailureAnalyzer,
            "unlearning": UnlearningAnalyzer,
        }

        if enable_analyses is None:
            enable_analyses = list(available_analyzers.keys())

        analyzers = {}
        for name in enable_analyses:
            if name in available_analyzers:
                analyzers[name] = available_analyzers[name](self.results_dir)
            else:
                print(f"⚠️  Warning: Unknown analyzer '{name}' - skipping")

        return analyzers

    def evaluate_model(
        self, model_path: Path, benchmark_path: Path, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on a benchmark.

        Args:
            model_path: Path to model checkpoint
            benchmark_path: Path to benchmark CSV
            model_name: Name for results (defaults to model filename)

        Returns:
            Dictionary with evaluation metrics and detailed results
        """
        if model_name is None:
            model_name = model_path.stem

        benchmark_name = benchmark_path.stem

        print(f"\n{'='*70}")
        print(f"EVALUATING: {model_name} on {benchmark_name}")
        print(f"{'='*70}")

        # Load tokenizer
        use_glove = "glove" in self.config.model_name.lower()

        if use_glove:
            print("Using GloVe - no tokenizer needed")
            tokenizer = None
        else:
            from transformers import AutoTokenizer

            print(f"Loading tokenizer: {self.config.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Load model
        print(f"Loading model from: {model_path}")
        model = get_model_class(self.config.architecture).load_from_checkpoint(
            str(model_path), config=self.config
        )
        model.to(self.device)
        model.eval()

        # Create dataset and dataloader
        print(f"Loading benchmark: {benchmark_path}")
        benchmark_dataset = BenchmarkDataset(benchmark_path, tokenizer, self.config)

        from torch.utils.data import DataLoader

        benchmark_loader = DataLoader(
            benchmark_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            collate_fn=benchmark_collate_fn,
            num_workers=2,
        )

        # Run inference
        print("Running inference...")
        detailed_results = self._run_inference(model, benchmark_loader)

        # Calculate metrics
        print("Calculating metrics...")
        metrics = self._calculate_metrics(detailed_results)

        # Store results
        key = f"{model_name}_{benchmark_name}"
        self.evaluation_results[key] = {"model": model_name, "benchmark": benchmark_name, **metrics}
        self.detailed_results[key] = detailed_results

        # Save detailed results
        detailed_path = self.results_dir / f"{key}_detailed.csv"
        detailed_results.to_csv(detailed_path, index=False)
        print(f"✅ Saved detailed results: {detailed_path.name}")

        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS: {model_name} on {benchmark_name}")
        print(f"{'='*70}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"{'='*70}\n")

        return {
            "model": model_name,
            "benchmark": benchmark_name,
            "metrics": metrics,
            "detailed_results": detailed_results,
        }

    def _run_inference(self, model, dataloader) -> pd.DataFrame:
        """
        Run inference and collect results.

        Args:
            model: Trained model
            dataloader: Benchmark dataloader

        Returns:
            DataFrame with predictions and labels
        """
        unrolled_rows = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                if batch is None:
                    continue

                # Move device indicator to correct device if using GloVe
                if "device_indicator" in batch:
                    batch["device_indicator"] = batch["device_indicator"].to(self.device)

                # Move tensors to device
                batch_tensors = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_tensors[k] = v.to(self.device)
                    else:
                        batch_tensors[k] = v

                # Get scores
                scores = model(batch_tensors)
                scores = torch.softmax(scores, dim=1).cpu().numpy()

                # Unroll results
                for i in range(len(batch["user_id"])):
                    user_id = batch["user_id"][i]
                    impression = batch["impression_data"][i]

                    for j, (news_id, title, label, is_fake) in enumerate(impression):
                        unrolled_rows.append(
                            {
                                "user_id": user_id,
                                "news_id": news_id,
                                "score": scores[i][j],
                                "label": label,
                                "is_fake": is_fake,
                            }
                        )

        return pd.DataFrame(unrolled_rows)

    def _calculate_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate evaluation metrics.

        Args:
            results_df: Detailed results

        Returns:
            Dictionary of metrics
        """
        # AUC
        auc = calculate_auc(results_df)

        # Per-user metrics
        mrr_scores = []
        ndcg_5_scores = []
        ndcg_10_scores = []
        recall_5_scores = []
        recall_10_scores = []

        for user_id, group in results_df.groupby("user_id"):
            mrr_scores.append(calculate_mrr(group))
            ndcg_5_scores.append(calculate_ndcg_at_k(group, k=5))
            ndcg_10_scores.append(calculate_ndcg_at_k(group, k=10))
            recall_5_scores.append(calculate_recall_at_k(group, k=5))
            recall_10_scores.append(calculate_recall_at_k(group, k=10))

        return {
            "AUC": auc,
            "MRR": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
            "NDCG@5": sum(ndcg_5_scores) / len(ndcg_5_scores) if ndcg_5_scores else 0.0,
            "NDCG@10": sum(ndcg_10_scores) / len(ndcg_10_scores) if ndcg_10_scores else 0.0,
            "Recall@5": sum(recall_5_scores) / len(recall_5_scores) if recall_5_scores else 0.0,
            "Recall@10": sum(recall_10_scores) / len(recall_10_scores) if recall_10_scores else 0.0,
        }

    def evaluate_all(
        self, model_types: Optional[List[str]] = None, benchmark_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Evaluate all models on all benchmarks.
        """
        print(f"\n{'='*70}")
        print("EVALUATING ALL MODELS")
        print(f"{'='*70}\n")

        set_seed(self.config.seed)

        checkpoints_dir = self.paths["checkpoints_dir"]
        benchmarks_dir = self.paths["benchmarks_dir"]

        # Find model checkpoints
        if model_types is None:
            model_files = list(checkpoints_dir.glob("*.ckpt"))
            print(f"Found {len(model_files)} model checkpoints")
        else:
            model_files = []
            for model_type in model_types:
                files = list(checkpoints_dir.glob(f"{model_type}*.ckpt"))
                model_files.extend(files)
            print(f"Evaluating {len(model_files)} specified models")

        # Find benchmarks
        if benchmark_names is None:
            benchmark_files = list(benchmarks_dir.glob("*.csv"))
            print(f"Found {len(benchmark_files)} benchmarks")
        else:
            benchmark_files = []
            for benchmark_name in benchmark_names:
                files = list(benchmarks_dir.glob(f"{benchmark_name}*.csv"))
                benchmark_files.extend(files)
            print(f"Evaluating on {len(benchmark_files)} specified benchmarks")

        if not model_files or not benchmark_files:
            print("❌ No models or benchmarks found")
            return pd.DataFrame()

        # Evaluate all combinations
        for model_file in model_files:
            # CRITICAL FIX: Use full filename stem, not just first part
            model_type = model_file.stem  # DON'T split by "-"!

            for benchmark_file in benchmark_files:
                try:
                    self.evaluate_model(model_file, benchmark_file, model_name=model_type)
                except Exception as e:
                    print(f"❌ Error evaluating {model_type} on {benchmark_file.stem}: {e}")
                    import traceback

                    traceback.print_exc()

        # Create summary
        if self.evaluation_results:
            summary_df = pd.DataFrame(list(self.evaluation_results.values()))
            summary_path = self.results_dir / "evaluation_summary.csv"
            summary_df.to_csv(summary_path, index=False)

            print(f"\n{'='*70}")
            print("EVALUATION SUMMARY")
            print(f"{'='*70}")
            print(summary_df.to_string(index=False))
            print(f"\n✅ Saved summary: {summary_path}")

            return summary_df

        return pd.DataFrame()

    def run_analyses(
        self,
        model_types: Optional[List[str]] = None,
        benchmark_filter: str = "mixed",
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Run all enabled analyses on evaluation results.

        Args:
            model_types: List of model types to analyze (default: all)
            benchmark_filter: Filter benchmarks by name (default: "mixed")
            top_k: Top-k for exposure analysis

        Returns:
            Dictionary with all analysis results
        """
        print(f"\n{'='*70}")
        print("RUNNING ANALYSES")
        print(f"{'='*70}\n")

        # Filter results by benchmark
        filtered_results = {k: v for k, v in self.detailed_results.items() if benchmark_filter in k}

        if not filtered_results:
            print(f"❌ No results found for benchmark filter: {benchmark_filter}")
            return {}

        # Group by model type - IMPROVED: Extract full model name properly
        model_results = {}
        for key, results_df in filtered_results.items():
            # Key format: "model_name_benchmark_name"
            # Extract model name by removing the benchmark suffix
            if f"_{benchmark_filter}" in key:
                # Split by benchmark name and take the first part
                model_name = key.split(f"_{benchmark_filter}")[0]
            else:
                # Fallback: assume last underscore-separated part is benchmark
                parts = key.split("_")
                model_name = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]

            # Filter by model_types if specified
            if model_types is None or any(model_name.startswith(mt) for mt in model_types):
                model_results[model_name] = results_df

        # Debug output
        print(f"DEBUG: Extracted model names from results:")
        for model_name in sorted(model_results.keys()):
            print(f"  - {model_name}")
        print()

        if not model_results:
            print(f"❌ No models found after filtering")
            return {}

        print(f"Analyzing {len(model_results)} models")

        analysis_results = {}

        # Run each enabled analyzer
        for analyzer_name, analyzer in self.analyzers.items():
            print(f"\n{'─'*70}")
            print(f"Running {analyzer_name} analysis...")
            print(f"{'─'*70}")

            try:
                if analyzer_name == "unlearning":
                    # Unlearning needs clean, poisoned, and multiple unlearned models

                    # Find clean and poisoned models
                    clean_results = None
                    poisoned_results = None

                    for model_name, results_df in model_results.items():
                        if model_name == "clean":
                            clean_results = results_df
                        elif model_name == "poisoned":
                            poisoned_results = results_df

                    if clean_results is None or poisoned_results is None:
                        print("⚠️  Skipping unlearning analysis - need clean and poisoned models")
                        missing = []
                        if clean_results is None:
                            missing.append("clean")
                        if poisoned_results is None:
                            missing.append("poisoned")
                        print(f"   Missing: {', '.join(missing)}")
                        continue

                    # Find ALL unlearned models - any model starting with "unlearned"
                    unlearned_models = {
                        model_name: results_df
                        for model_name, results_df in model_results.items()
                        if model_name.startswith("unlearned")
                    }

                    if not unlearned_models:
                        print("⚠️  No unlearned models found - skipping unlearning analysis")
                        print(f"   Available models: {list(model_results.keys())}")
                        continue

                    print(f"\n✅ Found {len(unlearned_models)} unlearned model(s):")
                    for idx, model_name in enumerate(sorted(unlearned_models.keys()), 1):
                        print(f"   {idx}. {model_name}")
                    print()

                    # Run analysis for EACH unlearned model
                    unlearning_results = {}
                    for unlearned_name, unlearned_df in unlearned_models.items():
                        print(f"\n{'  '*2}{'─'*66}")
                        print(f"{'  '*2}Analyzing: {unlearned_name}")
                        print(f"{'  '*2}{'─'*66}")

                        results = analyzer.analyze(
                            clean_results=clean_results,
                            poisoned_results=poisoned_results,
                            unlearned_results=unlearned_df,
                            unlearned_name=unlearned_name,
                            top_k=top_k,
                        )
                        unlearning_results[unlearned_name] = results

                        # Save results for this specific model
                        analyzer.save_results(results, unlearned_name=unlearned_name)

                        # Print quick summary
                        if results.get("has_unlearned"):
                            eff = results["effectiveness"]
                            print(f"{'  '*2}  → Recovery rate: {eff['recovery_rate']:.1f}%")
                            print(f"{'  '*2}  → Status: {eff['status']}")
                            print(f"{'  '*2}  → Gap from clean: {eff['gap_from_clean']:.4f}")

                    analysis_results[analyzer_name] = unlearning_results

                    # Create summary CSV for all unlearned models
                    print(f"\n{'  '*2}Creating summary for all unlearned models...")
                    if unlearning_results:
                        summary_data = []
                        for unlearned_name, results in unlearning_results.items():
                            if results.get("has_unlearned"):
                                eff = results["effectiveness"]

                                # Extract ratio and trial from model name if available
                                ratio = None
                                trial = None
                                method = None

                                # Parse name like: unlearned-first_order-ratio-0.050-trial-0
                                if "-ratio-" in unlearned_name:
                                    try:
                                        ratio_part = unlearned_name.split("-ratio-")[1].split("-")[
                                            0
                                        ]
                                        ratio = float(ratio_part)
                                    except:
                                        pass

                                if "-trial-" in unlearned_name:
                                    try:
                                        trial_part = unlearned_name.split("-trial-")[1].split("-")[
                                            0
                                        ]
                                        trial = int(trial_part)
                                    except:
                                        pass

                                if "unlearned-" in unlearned_name:
                                    try:
                                        method = unlearned_name.split("unlearned-")[1].split("-")[0]
                                    except:
                                        pass

                                summary_data.append(
                                    {
                                        "model": unlearned_name,
                                        "method": method,
                                        "ratio": ratio,
                                        "trial": trial,
                                        "status": eff["status"],
                                        "recovery_rate": eff["recovery_rate"],
                                        "gap_from_clean": eff["gap_from_clean"],
                                        "is_effective": eff["is_effective"],
                                        "clean_exposure": eff["clean_exposure"],
                                        "poisoned_exposure": eff["poisoned_exposure"],
                                        "unlearned_exposure": eff["unlearned_exposure"],
                                    }
                                )

                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)

                            # Sort by method, ratio, trial for better readability
                            sort_cols = []
                            if (
                                "method" in summary_df.columns
                                and summary_df["method"].notna().any()
                            ):
                                sort_cols.append("method")
                            if "ratio" in summary_df.columns and summary_df["ratio"].notna().any():
                                sort_cols.append("ratio")
                            if "trial" in summary_df.columns and summary_df["trial"].notna().any():
                                sort_cols.append("trial")

                            if sort_cols:
                                summary_df = summary_df.sort_values(sort_cols)

                            summary_path = self.results_dir / "unlearning_summary_all_models.csv"
                            summary_df.to_csv(summary_path, index=False)
                            print(f"{'  '*2}✅ Saved summary: {summary_path.name}")

                            # Print summary table
                            print(f"\n{'  '*2}{'='*66}")
                            print(f"{'  '*2}SUMMARY: All Unlearned Models")
                            print(f"{'  '*2}{'='*66}")
                            print(f"{'  '*2}{'Model':<35} {'Recovery':<10} {'Status':<15}")
                            print(f"{'  '*2}{'-'*66}")
                            for _, row in summary_df.iterrows():
                                model_short = (
                                    row["model"][:33] + ".."
                                    if len(row["model"]) > 35
                                    else row["model"]
                                )
                                print(
                                    f"{'  '*2}{model_short:<35} {row['recovery_rate']:>6.1f}%    {row['status']:<15}"
                                )
                            print(f"{'  '*2}{'='*66}")

                    # Create comparison visualization for all unlearned models
                    print(f"\n{'  '*2}Creating comparison visualization...")
                    self.visualizer.plot_unlearning_comparison(
                        clean_results=clean_results,
                        poisoned_results=poisoned_results,
                        unlearned_results=unlearned_models,
                        top_k=top_k,
                    )

                else:
                    # Other analyses run on each model
                    for model_name, results_df in model_results.items():
                        print(f"\n  Analyzing {model_name}...")
                        results = analyzer.run(results_df, top_k=top_k)
                        analysis_results[f"{analyzer_name}_{model_name}"] = results

                    # Create comparison visualizations
                    if analyzer_name == "exposure":
                        comparison = analyzer.compare_models(model_results, top_k=top_k)
                        self.visualizer.plot_exposure_comparison(comparison, top_k=top_k)

                    elif analyzer_name == "truth_decay":
                        comparison = analyzer.compare_models(model_results, top_k=top_k)
                        # Create rank comparison plot
                        rank_data = {}
                        for model_name, results_df in model_results.items():
                            df = results_df.copy()
                            df["rank"] = df.groupby("user_id")["score"].rank(
                                method="first", ascending=False
                            )
                            df["news_type"] = df["is_fake"].apply(lambda x: "Fake" if x else "Real")
                            rank_data[model_name] = df

                        self.visualizer.plot_rank_comparison(rank_data)

                    elif analyzer_name == "failure":
                        comparison = analyzer.compare_models(model_results)
                        self.visualizer.plot_failure_analysis(comparison)

            except Exception as e:
                print(f"❌ Error in {analyzer_name} analysis: {e}")
                import traceback

                traceback.print_exc()

        print(f"\n{'='*70}")
        print("ALL ANALYSES COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved in: {self.results_dir}")

        return analysis_results

    def generate_report(self, output_path: Optional[Path] = None):
        """
        Generate comprehensive evaluation report.

        Args:
            output_path: Path to save report (default: results_dir/report.txt)
        """
        if output_path is None:
            output_path = self.results_dir / "evaluation_report.txt"

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("COMPREHENSIVE EVALUATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Dataset: {self.config.dataset}")
        report_lines.append(f"Experiment: {self.config.experiment_name}")
        report_lines.append(f"Model: {self.config.model_name}")
        report_lines.append("")

        # Add evaluation summary
        if self.evaluation_results:
            report_lines.append("EVALUATION METRICS:")
            report_lines.append("-" * 70)
            for key, result in self.evaluation_results.items():
                report_lines.append(f"\n{key}:")
                for metric, value in result.items():
                    if metric not in ["model", "benchmark"]:
                        report_lines.append(f"  {metric}: {value:.4f}")

        # Add analyzer summaries
        for analyzer_name, analyzer in self.analyzers.items():
            if analyzer.analysis_results:
                report_lines.append(f"\n{analyzer.get_summary()}")

        report_lines.append("\n" + "=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)

        # Save report
        with open(output_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"✅ Report saved: {output_path}")
