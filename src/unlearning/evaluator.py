# src/unlearning/evaluator.py

"""
Unlearning Evaluator - Orchestrates all metric computation.

Evaluates unlearning from three perspectives:
1. Forget Quality - How well did we forget?
2. Utility Preservation - Did we maintain performance on retain/test data?
3. Efficiency - How fast and resource-efficient was it?
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import json
from datetime import datetime
from torch.utils.data import DataLoader

from configs import BaseConfig
from src.unlearning.metrics.forget_quality import (
    calculate_forget_quality,
    calculate_forget_delta,
    evaluate_forgetting_completeness,
    calculate_forget_efficacy,
)
from src.unlearning.metrics.utility import (
    calculate_retain_quality,
    calculate_utility_preservation,
    calculate_test_performance,
    calculate_unlearning_efficiency,
)
from src.unlearning.metrics.efficiency import (
    calculate_parameter_changes,
    calculate_memory_usage,
    calculate_efficiency_metrics,
    UnlearningTimer,
)


class UnlearningEvaluator:
    """
    Comprehensive evaluator for machine unlearning.
    Uses standard metrics from the metrics modules.
    """

    def __init__(
        self, config: BaseConfig, device: torch.device, results_dir: Optional[Path] = None
    ):
        """Initialize evaluator."""
        self.config = config
        self.device = device

        if results_dir is None:
            paths = config.get_paths()
            self.results_dir = paths["results_dir"] / "unlearning"
        else:
            self.results_dir = Path(results_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss()

        print(f"UnlearningEvaluator initialized")
        print(f"  Results directory: {self.results_dir}")

    def evaluate_full(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        unlearning_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Comprehensive evaluation using standard metrics."""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE UNLEARNING EVALUATION")
        print(f"{'='*70}\n")

        results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
        }

        # 1. Forget Quality
        print("1/3 Evaluating Forget Quality...")
        forget_results = self._evaluate_forget_quality(model_before, model_after, forget_loader)
        results["forget_quality"] = forget_results
        self._print_forget_summary(forget_results)

        # 2. Utility Preservation
        print(f"\n2/3 Evaluating Utility Preservation...")
        utility_results = self._evaluate_utility(
            model_before, model_after, retain_loader, test_loader
        )
        results["utility"] = utility_results
        self._print_utility_summary(utility_results)

        # 3. Efficiency
        print(f"\n3/3 Evaluating Efficiency...")
        efficiency_results = self._evaluate_efficiency(model_before, model_after, unlearning_time)
        results["efficiency"] = efficiency_results
        self._print_efficiency_summary(efficiency_results)

        # Overall score
        results["overall"] = self._calculate_overall_score(
            forget_results, utility_results, efficiency_results
        )

        print(f"\n{'='*70}")
        print("EVALUATION COMPLETE")
        print(f"{'='*70}\n")

        return results

    def _evaluate_forget_quality(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        forget_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Evaluate forget quality using standard metrics."""
        # Before unlearning
        print("    Computing before metrics...")
        forget_before = calculate_forget_quality(
            model_before, forget_loader, self.device, self.criterion
        )

        # After unlearning
        print("    Computing after metrics...")
        forget_after = calculate_forget_quality(
            model_after, forget_loader, self.device, self.criterion
        )

        # Delta
        forget_delta = calculate_forget_delta(forget_before, forget_after)

        # Completeness
        completeness = evaluate_forgetting_completeness(model_after, forget_loader, self.device)

        # Efficacy
        efficacy = calculate_forget_efficacy(forget_before, forget_after)

        return {
            "before": forget_before,
            "after": forget_after,
            "delta": forget_delta,
            "completeness": completeness,
            "efficacy": efficacy,
        }

    def _evaluate_utility(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        retain_loader: DataLoader,
        test_loader: Optional[DataLoader],
    ) -> Dict[str, Any]:
        """Evaluate utility using standard metrics."""
        # Retain set
        retain_before = calculate_retain_quality(
            model_before, retain_loader, self.device, self.criterion
        )
        retain_after = calculate_retain_quality(
            model_after, retain_loader, self.device, self.criterion
        )
        retain_preservation = calculate_utility_preservation(retain_before, retain_after)

        results = {
            "retain_before": retain_before,
            "retain_after": retain_after,
            "retain_preservation": retain_preservation,
        }

        # Test set (if provided)
        if test_loader is not None:
            test_before = calculate_test_performance(model_before, test_loader, self.device)
            test_after = calculate_test_performance(model_after, test_loader, self.device)
            test_preservation = calculate_utility_preservation(test_before, test_after)

            results["test_before"] = test_before
            results["test_after"] = test_after
            results["test_preservation"] = test_preservation

        # Benchmark evaluation (if exists)
        benchmark_path = self.config.benchmark_dir / "benchmark_real_only.csv"
        if benchmark_path.exists():
            print(f"  Loading benchmark: {benchmark_path}")
            benchmark_loader = self._create_benchmark_loader(benchmark_path)

            if benchmark_loader is not None:
                benchmark_before = calculate_test_performance(
                    model_before, benchmark_loader, self.device
                )
                benchmark_after = calculate_test_performance(
                    model_after, benchmark_loader, self.device
                )
                benchmark_preservation = calculate_utility_preservation(
                    benchmark_before, benchmark_after
                )

                results["benchmark_before"] = benchmark_before
                results["benchmark_after"] = benchmark_after
                results["benchmark_preservation"] = benchmark_preservation

        return results

    def _create_benchmark_loader(self, benchmark_path: Path) -> Optional[DataLoader]:
        """Create DataLoader for benchmark dataset."""
        try:
            from src.data import NewsDataset, collate_fn

            use_glove = "glove" in self.config.model_name.lower()
            tokenizer = None
            if not use_glove:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            benchmark_dataset = NewsDataset(benchmark_path, self.config, tokenizer)
            benchmark_loader = DataLoader(
                benchmark_dataset,
                batch_size=self.config.val_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=2,
            )
            return benchmark_loader
        except Exception as e:
            print(f"  ⚠️  Failed to create benchmark loader: {e}")
            return None

    def _evaluate_efficiency(
        self, model_before: nn.Module, model_after: nn.Module, unlearning_time: Optional[float]
    ) -> Dict[str, Any]:
        """Evaluate efficiency using standard metrics."""
        # Parameter changes
        param_changes = calculate_parameter_changes(model_before, model_after)

        # Memory usage
        memory_usage = calculate_memory_usage(model_after, self.device)

        # Combine into efficiency metrics
        if unlearning_time is not None:
            efficiency_metrics = calculate_efficiency_metrics(
                unlearning_time, param_changes, memory_usage
            )
            return efficiency_metrics
        else:
            return {
                "param_changes": param_changes,
                "memory_usage": memory_usage,
            }

    def _calculate_overall_score(
        self,
        forget_results: Dict[str, Any],
        utility_results: Dict[str, Any],
        efficiency_results: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate overall unlearning score."""
        # Forget quality score
        forget_score = forget_results["efficacy"]

        # Utility score
        retain_preserved = 1.0 if utility_results["retain_preservation"]["is_preserved"] else 0.5
        retain_acc = utility_results["retain_after"]["accuracy"]

        if "benchmark_after" in utility_results:
            benchmark_acc = utility_results["benchmark_after"]["accuracy"]
            utility_score = 0.5 * retain_acc + 0.3 * benchmark_acc + 0.2 * retain_preserved
        else:
            utility_score = 0.7 * retain_acc + 0.3 * retain_preserved

        # Efficiency score
        params_changed_pct = efficiency_results.get("param_changes", {}).get(
            "params_changed_pct", 0.0
        )
        efficiency_score = 1.0 - min(params_changed_pct / 100.0, 1.0)

        # Overall weighted score
        overall_score = 0.4 * forget_score + 0.4 * utility_score + 0.2 * efficiency_score

        return {
            "forget_score": forget_score,
            "utility_score": utility_score,
            "efficiency_score": efficiency_score,
            "overall_score": overall_score,
            "grade": self._score_to_grade(overall_score),
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Very Good)"
        elif score >= 0.7:
            return "B (Good)"
        elif score >= 0.6:
            return "C (Acceptable)"
        elif score >= 0.5:
            return "D (Poor)"
        else:
            return "F (Failed)"

    def _print_forget_summary(self, results: Dict[str, Any]):
        """Print forget quality summary."""
        before = results["before"]
        after = results["after"]
        delta = results["delta"]
        efficacy = results["efficacy"]

        print(f"  Forget Set Metrics:")
        print(f"    Before - Loss: {before['loss']:.4f}, Acc: {before['accuracy']:.4f}")
        print(f"    After  - Loss: {after['loss']:.4f}, Acc: {after['accuracy']:.4f}")
        print(f"    Delta  - Loss: {delta['loss_delta']:+.4f}, Acc: {delta['accuracy_delta']:+.4f}")
        print(f"    Efficacy: {efficacy:.4f}")

    def _print_utility_summary(self, results: Dict[str, Any]):
        """Print utility preservation summary."""
        retain_pres = results["retain_preservation"]

        print(f"  Retain Set Metrics:")
        print(f"    Before - Acc: {retain_pres['accuracy_before']:.4f}")
        print(f"    After  - Acc: {retain_pres['accuracy_after']:.4f}")
        print(
            f"    Drop   - {retain_pres['accuracy_drop']:.4f} ({retain_pres['accuracy_drop_pct']:.2f}%)"
        )
        print(
            f"    Preserved: {retain_pres['is_preserved']} (tolerance: {retain_pres['tolerance']:.2%})"
        )

        if "test_preservation" in results:
            test_pres = results["test_preservation"]
            print(f"  Test Set Metrics:")
            print(f"    Before - Acc: {test_pres['accuracy_before']:.4f}")
            print(f"    After  - Acc: {test_pres['accuracy_after']:.4f}")
            print(f"    Drop   - {test_pres['accuracy_drop']:.4f}")

        if "benchmark_preservation" in results:
            bench_pres = results["benchmark_preservation"]
            print(f"  Benchmark Metrics:")
            print(f"    Before - Acc: {bench_pres['accuracy_before']:.4f}")
            print(f"    After  - Acc: {bench_pres['accuracy_after']:.4f}")
            print(f"    Drop   - {bench_pres['accuracy_drop']:.4f}")

    def _print_efficiency_summary(self, results: Dict[str, Any]):
        """Print efficiency summary."""
        if "param_changes" in results:
            param_changes = results["param_changes"]
            print(f"  Parameter Changes:")
            print(f"    Total params: {param_changes['total_params']:,}")
            print(f"    Trainable params: {param_changes['trainable_params']:,}")
            print(
                f"    Changed: {param_changes['params_changed']:,} ({param_changes['params_changed_pct']:.2f}% of trainable)"
            )
            print(f"    Avg change: {param_changes['avg_change']:.6f}")
            print(f"    Max change: {param_changes['max_change']:.6f}")

        if "memory_usage" in results:
            memory = results["memory_usage"]
            print(f"  Memory:")
            print(f"    Model size: {memory['model_size_mb']:.2f} MB")
            if memory["gpu_allocated_mb"] > 0:
                print(f"    GPU memory: {memory['gpu_allocated_mb']:.2f} MB")

        if "time_seconds" in results:
            print(f"  Time:")
            print(
                f"    Unlearning: {results['time_seconds']:.2f}s ({results['time_minutes']:.2f}m)"
            )

    def save_results(
        self, results: Dict[str, Any], filename: str = "unlearning_evaluation.json"
    ) -> Path:
        """
        Save evaluation results to JSON.

        Args:
            results: Evaluation results dictionary
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.results_dir / filename

        # Convert non-serializable types
        serializable_results = self._make_serializable(results)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"✅ Results saved: {output_path}")
        return output_path

    def save_summary_csv(
        self, results: Dict[str, Any], filename: str = "unlearning_summary.csv"
    ) -> Path:
        """
        Save summary metrics to CSV.

        Args:
            results: Evaluation results
            filename: Output filename

        Returns:
            Path to saved file
        """
        # Extract key metrics
        summary_data = []

        # Forget metrics
        forget = results["forget_quality"]
        summary_data.append(
            {"metric": "forget_accuracy_before", "value": forget["before"]["accuracy"]}
        )
        summary_data.append(
            {"metric": "forget_accuracy_after", "value": forget["after"]["accuracy"]}
        )
        summary_data.append(
            {"metric": "forget_accuracy_delta", "value": forget["delta"]["accuracy_delta"]}
        )
        summary_data.append({"metric": "forget_efficacy", "value": forget["efficacy"]})

        # Utility metrics
        utility = results["utility"]
        summary_data.append(
            {"metric": "retain_accuracy_before", "value": utility["retain_before"]["accuracy"]}
        )
        summary_data.append(
            {"metric": "retain_accuracy_after", "value": utility["retain_after"]["accuracy"]}
        )
        summary_data.append(
            {
                "metric": "retain_preserved",
                "value": 1.0 if utility["retain_preservation"]["is_preserved"] else 0.0,
            }
        )

        # Overall
        overall = results["overall"]
        summary_data.append({"metric": "overall_score", "value": overall["overall_score"]})
        summary_data.append({"metric": "overall_grade", "value": overall["grade"]})

        # Save to CSV
        df = pd.DataFrame(summary_data)
        output_path = self.results_dir / filename
        df.to_csv(output_path, index=False)

        print(f"✅ Summary saved: {output_path}")
        return output_path

    def print_summary(self, results: Dict[str, Any]):
        """
        Print comprehensive summary of results.

        Args:
            results: Evaluation results
        """
        print(f"\n{'='*70}")
        print("UNLEARNING EVALUATION SUMMARY")
        print(f"{'='*70}")

        # Forget Quality
        print(f"\n{'─'*70}")
        print("FORGET QUALITY")
        print(f"{'─'*70}")
        self._print_forget_summary(results["forget_quality"])

        # Utility
        print(f"\n{'─'*70}")
        print("UTILITY PRESERVATION")
        print(f"{'─'*70}")
        self._print_utility_summary(results["utility"])

        # Efficiency
        print(f"\n{'─'*70}")
        print("EFFICIENCY")
        print(f"{'─'*70}")
        self._print_efficiency_summary(results["efficiency"])

        # Overall
        print(f"\n{'─'*70}")
        print("OVERALL ASSESSMENT")
        print(f"{'─'*70}")
        overall = results["overall"]
        print(f"  Forget Score: {overall['forget_score']:.4f}")
        print(f"  Utility Score: {overall['utility_score']:.4f}")
        print(f"  Efficiency Score: {overall['efficiency_score']:.4f}")
        print(f"  Overall Score: {overall['overall_score']:.4f}")
        print(f"  Grade: {overall['grade']}")

        # Verdict
        print(f"\n{'─'*70}")
        print("VERDICT")
        print(f"{'─'*70}")
        self._print_verdict(results)

        print(f"{'='*70}\n")

    def _print_verdict(self, results: Dict[str, Any]):
        """Print final verdict on unlearning quality."""
        forget = results["forget_quality"]
        utility = results["utility"]
        overall = results["overall"]

        # Check criteria
        is_forgotten = forget["completeness"]["is_forgotten"]
        utility_preserved = utility["retain_preservation"]["is_preserved"]
        overall_score = overall["overall_score"]

        print(f"  Forget set forgotten: {'✅ Yes' if is_forgotten else '❌ No'}")
        print(f"  Utility preserved: {'✅ Yes' if utility_preserved else '❌ No'}")
        print(f"  Overall quality: {overall['grade']}")

        if is_forgotten and utility_preserved and overall_score >= 0.7:
            print(f"\n  🎉 SUCCESSFUL UNLEARNING")
            print(f"     The model has effectively forgotten the forget set")
            print(f"     while maintaining utility on the retain set.")
        elif is_forgotten and overall_score >= 0.6:
            print(f"\n  ✅ ACCEPTABLE UNLEARNING")
            print(f"     Forgetting achieved, but some utility degradation.")
        elif utility_preserved and forget["efficacy"] >= 0.5:
            print(f"\n  ⚠️  PARTIAL UNLEARNING")
            print(f"     Utility maintained but incomplete forgetting.")
        else:
            print(f"\n  ❌ UNSUCCESSFUL UNLEARNING")
            print(f"     Consider adjusting unlearning parameters or trying")
            print(f"     a different method.")

    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            return str(obj)

    def compare_methods(
        self, results_dict: Dict[str, Dict[str, Any]], save: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple unlearning methods.

        Args:
            results_dict: Dict mapping method_name -> evaluation_results
            save: Whether to save comparison to CSV

        Returns:
            DataFrame with comparison

        Example:
            >>> comparison = evaluator.compare_methods({
            ...     'first_order': results1,
            ...     'gradient_ascent': results2
            ... })
        """
        comparison_data = []

        for method_name, results in results_dict.items():
            forget = results["forget_quality"]
            utility = results["utility"]
            efficiency = results["efficiency"]
            overall = results["overall"]

            comparison_data.append(
                {
                    "method": method_name,
                    "forget_acc_after": forget["after"]["accuracy"],
                    "forget_efficacy": forget["efficacy"],
                    "retain_acc_after": utility["retain_after"]["accuracy"],
                    "utility_preserved": utility["retain_preservation"]["is_preserved"],
                    "params_changed_pct": efficiency["param_changes"]["params_changed_pct"],
                    "time_seconds": efficiency.get("time_seconds", 0.0),
                    "overall_score": overall["overall_score"],
                    "grade": overall["grade"],
                }
            )

        df = pd.DataFrame(comparison_data)

        # Sort by overall score
        df = df.sort_values("overall_score", ascending=False)

        if save:
            output_path = self.results_dir / "method_comparison.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Comparison saved: {output_path}")

        return df
