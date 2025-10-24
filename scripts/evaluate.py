#!/usr/bin/env python3
"""
CLI for evaluating news recommendation models.

Supports:
- Single model evaluation
- Batch evaluation (all models)
- Flexible analysis selection
- Works with or without unlearned models
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_config
from src.evaluation import ModelEvaluator
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate news recommendation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models using existing config
  python scripts/evaluate.py --config configs/experiments/simple/bert_finetune.yaml
  
  # Evaluate specific model types only
  python scripts/evaluate.py --config configs/experiments/simple/bert_finetune.yaml --models clean poisoned
  
  # Run specific analyses
  python scripts/evaluate.py --config configs/experiments/simple/bert_finetune.yaml --analyses exposure failure
  
  # Evaluate on specific benchmarks
  python scripts/evaluate.py --config configs/experiments/simple/bert_finetune.yaml --benchmarks benchmark_mixed
  
  # Skip analyses, just get metrics
  python scripts/evaluate.py --config configs/experiments/simple/bert_finetune.yaml --no-analysis
  
  # Include unlearning analysis (if unlearned model exists)
  python scripts/evaluate.py --config configs/experiments/simple/bert_finetune.yaml --analyses exposure unlearning
        """,
    )

    # ========== Config ==========
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")

    # ========== Model Selection ==========
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Model types to evaluate (default: all in checkpoints_dir). Example: clean poisoned unlearned",
    )

    parser.add_argument(
        "--model-path", type=str, help="Evaluate single model checkpoint (overrides --models)"
    )

    # ========== Benchmark Selection ==========
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        help="Benchmark names to use (default: all in benchmarks_dir). Example: benchmark_mixed benchmark_real_only",
    )

    parser.add_argument(
        "--benchmark-filter",
        type=str,
        default="mixed",
        help="Filter benchmarks by name for analysis (default: mixed)",
    )

    # ========== Analysis Options ==========
    parser.add_argument(
        "--analyses",
        type=str,
        nargs="+",
        choices=["exposure", "truth_decay", "failure", "unlearning", "all"],
        help='Analyses to run (default: all except unlearning). Use "all" to include unlearning.',
    )

    parser.add_argument(
        "--no-analysis", action="store_true", help="Skip analysis phase, only calculate metrics"
    )

    parser.add_argument(
        "--top-k", type=int, default=10, help="Top-k for exposure analysis (default: 10)"
    )

    # ========== Output Options ==========
    parser.add_argument(
        "--output-dir", type=str, help="Override output directory (default: from config)"
    )

    parser.add_argument(
        "--generate-report", action="store_true", help="Generate comprehensive text report"
    )

    # ========== Other Options ==========
    parser.add_argument("--gpu", type=int, help="GPU device ID (default: auto-detect)")

    parser.add_argument("--seed", type=int, help="Random seed (default: from config)")

    parser.add_argument("--verbose", action="store_true", help="Print detailed information")

    args = parser.parse_args()

    # ========== Load Config ==========
    print(f"\n{'='*70}")
    print("LOADING CONFIGURATION")
    print(f"{'='*70}")

    config = load_config(args.config)

    # Apply overrides
    if args.seed:
        config.seed = args.seed

    if args.output_dir:
        config.base_dir = Path(args.output_dir)

    if args.verbose:
        config.print_config()

    # ========== Setup Device ==========
    import torch

    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nUsing device: {device}")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # ========== Determine Enabled Analyses ==========
    if args.no_analysis:
        enable_analyses = []
        print("\n⚠️  Analysis phase disabled")
    elif args.analyses:
        if "all" in args.analyses:
            enable_analyses = ["exposure", "truth_decay", "failure", "unlearning"]
        else:
            enable_analyses = args.analyses
    else:
        # Default: all except unlearning (since it may not be available)
        enable_analyses = ["exposure", "truth_decay", "failure"]

    if enable_analyses:
        print(f"\nEnabled analyses: {enable_analyses}")
        if "unlearning" in enable_analyses:
            print("⚠️  Note: Unlearning analysis requires clean, poisoned, and unlearned models")

    # ========== Initialize Evaluator ==========
    print(f"\n{'='*70}")
    print("INITIALIZING EVALUATOR")
    print(f"{'='*70}")

    evaluator = ModelEvaluator(config=config, device=device, enable_analyses=enable_analyses)

    # ========== Run Evaluation ==========
    try:
        if args.model_path:
            # Evaluate single model
            print(f"\n{'='*70}")
            print("EVALUATING SINGLE MODEL")
            print(f"{'='*70}")

            model_path = Path(args.model_path)
            if not model_path.exists():
                print(f"❌ Model not found: {model_path}")
                return 1

            # Find benchmarks
            benchmarks_dir = evaluator.paths["benchmarks_dir"]  # ← Now this works!

            if not benchmarks_dir.exists():
                print(f"❌ Benchmarks directory not found: {benchmarks_dir}")
                print(f"   Please create benchmarks in: {benchmarks_dir}")
                return 1

            if args.benchmarks:
                benchmark_files = []
                for benchmark_name in args.benchmarks:
                    # Try exact match first
                    files = list(benchmarks_dir.glob(f"{benchmark_name}.csv"))
                    if not files:
                        # Try with wildcard
                        files = list(benchmarks_dir.glob(f"*{benchmark_name}*.csv"))
                    benchmark_files.extend(files)
            else:
                benchmark_files = list(benchmarks_dir.glob("*.csv"))

            if not benchmark_files:
                print(f"❌ No benchmarks found in {benchmarks_dir}")
                print(f"   Available files: {list(benchmarks_dir.glob('*'))}")
                return 1

            print(f"Evaluating on {len(benchmark_files)} benchmarks")

            for benchmark_file in benchmark_files:
                evaluator.evaluate_model(model_path, benchmark_file)

        else:
            # Evaluate all models
            print(f"\n{'='*70}")
            print("EVALUATING ALL MODELS")
            print(f"{'='*70}")

            # Convert benchmark names to list if provided
            benchmark_names = args.benchmarks if args.benchmarks else None

            summary_df = evaluator.evaluate_all(
                model_types=args.models, benchmark_names=benchmark_names
            )

            if summary_df.empty:
                print("❌ No evaluations completed")
                return 1

        # ========== Run Analyses ==========
        if enable_analyses:
            print(f"\n{'='*70}")
            print("RUNNING ANALYSES")
            print(f"{'='*70}")

            analysis_results = evaluator.run_analyses(
                model_types=args.models, benchmark_filter=args.benchmark_filter, top_k=args.top_k
            )

            if not analysis_results:
                print("⚠️  No analyses completed")

        # ========== Generate Report ==========
        if args.generate_report:
            print(f"\n{'='*70}")
            print("GENERATING REPORT")
            print(f"{'='*70}")

            evaluator.generate_report()

        # ========== Summary ==========
        print(f"\n{'='*70}")
        print("EVALUATION COMPLETE")
        print(f"{'='*70}")
        print(f"\n📁 All results saved in: {evaluator.results_dir}")
        print("\n📊 Generated files:")
        print("  • evaluation_summary.csv - Overall metrics")
        print("  • *_detailed.csv - Detailed predictions per model")

        if enable_analyses:
            if "exposure" in enable_analyses:
                print("  • exposure_* - Fake news exposure analysis")
            if "truth_decay" in enable_analyses:
                print("  • truth_decay_* - Ranking pattern analysis")
            if "failure" in enable_analyses:
                print("  • failure_* - Failure case analysis")
            if "unlearning" in enable_analyses:
                print("  • unlearning_* - Unlearning effectiveness")
            print("  • *.png - Visualizations")

        if args.generate_report:
            print("  • evaluation_report.txt - Comprehensive report")

        print(f"\n{'='*70}")

        # ========== Unlearning Status Check ==========
        if "unlearning" in enable_analyses:
            # Check if unlearning analysis was actually run
            if "unlearning" in analysis_results:
                unlearning_results = analysis_results["unlearning"]

                if isinstance(unlearning_results, dict) and unlearning_results:
                    print("\n🔄 UNLEARNING ANALYSIS SUMMARY:")
                    print(f"{'='*70}")
                    print(f"{'Model':<40} {'Status':<20} {'Recovery':<10}")
                    print(f"{'-'*70}")

                    for model_name, results in unlearning_results.items():
                        if results.get("has_unlearned"):
                            effectiveness = results["effectiveness"]
                            status = effectiveness["status"]
                            recovery = effectiveness["recovery_rate"]

                            # Format model name - remove "unlearned" prefix
                            display_name = model_name.replace("unlearned", "").strip("-_")
                            if not display_name:
                                display_name = "Unlearned"

                            print(f"{display_name:<40} {status:<20} {recovery:>6.1f}%")

                    print(f"{'='*70}")

                    # Overall assessment
                    successful = sum(
                        1
                        for r in unlearning_results.values()
                        if r.get("has_unlearned") and r["effectiveness"]["is_effective"]
                    )
                    total = len(unlearning_results)

                    print(f"\n✅ Successful unlearning: {successful}/{total} models")

                    if successful == total:
                        print("🎉 All unlearned models are effective!")
                    elif successful > 0:
                        print("⚠️  Some models need improvement")
                    else:
                        print("❌ Unlearning was not effective")
                else:
                    print("\n⚠️  No unlearned models found")
            print(f"{'='*70}")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
