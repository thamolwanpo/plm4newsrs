#!/usr/bin/env python3
"""
CLI for evaluating news recommendation models on the stage6 benchmarks.

For every trained model (last-epoch checkpoint per model, when available) it evaluates:
  standard, standard_clean, standard_mixed, adversarial,
  pattern_probe(_realneg), poison_probe(_realneg)
plus poison_probe(_realneg)@<A set>: the poison probes restricted to the
fake items injected into that poisoned set (clean model: every set,
poisoned model: its own set).
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_config
from src.evaluation import ModelEvaluator


ANALYSES = ["exposure", "truth_decay", "failure"]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate news recommendation models on the stage6 benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate every trained model on every benchmark
  python scripts/evaluate.py --config configs/experiments/nrms/nrms_glove.yaml

  # Only the clean model and one poisoned model
  python scripts/evaluate.py --config configs/experiments/nrms/nrms_glove.yaml \\
      --models clean poisoned_A_r0.5_s1

  # Only some benchmarks, metrics only
  python scripts/evaluate.py --config configs/experiments/nrms/nrms_glove.yaml \\
      --benchmarks standard poison_probe_realneg --no-analysis

  # A specific checkpoint
  python scripts/evaluate.py --config configs/experiments/nrms/nrms_glove.yaml \\
      --model-path outputs/politifact/nrms_model/glove_300_frozen/checkpoints/clean-epoch=03-val_auc=0.7000.ckpt
        """,
    )

    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument(
        "--data-root",
        type=str,
        help="fakeun-dataprep output folder of the dataset (overrides config)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Models to evaluate: run names or prefixes, e.g. clean poisoned_A_r0.5_s1 "
        "(default: every model in checkpoints_dir)",
    )
    parser.add_argument(
        "--model-path", type=str, help="Evaluate a single checkpoint (overrides --models)"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        help="stage6 benchmark names, e.g. standard adversarial (default: all)",
    )
    parser.add_argument(
        "--analyses",
        type=str,
        nargs="+",
        choices=ANALYSES + ["all"],
        help="Analyses to run (default: all)",
    )
    parser.add_argument(
        "--analysis-benchmark",
        type=str,
        default="standard",
        help="Benchmark the analyses run on (default: standard)",
    )
    parser.add_argument(
        "--no-analysis", action="store_true", help="Skip analysis phase, only calculate metrics"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Top-k for exposure analysis (default: 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Override base output directory (default: from config)"
    )
    parser.add_argument(
        "--generate-report", action="store_true", help="Write evaluation_report.txt"
    )
    parser.add_argument("--gpu", type=int, help="GPU device ID (default: auto-detect)")
    parser.add_argument("--seed", type=int, help="Random seed (default: from config)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")

    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed:
        config.seed = args.seed
    if args.output_dir:
        config.base_dir = Path(args.output_dir)
    if args.data_root:
        config.data_root = Path(args.data_root)
    if args.verbose:
        config.print_config()

    import torch

    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if args.no_analysis:
        enable_analyses = []
    elif args.analyses and "all" not in args.analyses:
        enable_analyses = args.analyses
    else:
        enable_analyses = ANALYSES

    evaluator = ModelEvaluator(config=config, device=device, enable_analyses=enable_analyses)

    try:
        if args.model_path:
            model_path = Path(args.model_path)
            if not model_path.exists():
                print(f"❌ Model not found: {model_path}")
                return 1
            benchmark_files = evaluator.discover_benchmarks(args.benchmarks)
            if not benchmark_files:
                print(f"❌ No benchmarks found in {evaluator.paths['benchmarks_dir']}")
                return 1
            evaluator.evaluate_checkpoint(model_path, None, benchmark_files)
            summary_df = evaluator.save_summary()
        else:
            summary_df = evaluator.evaluate_all(
                model_names=args.models, benchmark_names=args.benchmarks
            )

        if summary_df.empty:
            print("❌ No evaluations completed")
            return 1

        if enable_analyses:
            evaluator.run_analyses(
                model_names=args.models, benchmark=args.analysis_benchmark, top_k=args.top_k
            )
            if args.analysis_benchmark == "standard" and "exposure" in enable_analyses:
                evaluated_benchmarks = set(summary_df["benchmark"])
                for benchmark in ("standard_clean", "standard_mixed"):
                    if benchmark in evaluated_benchmarks:
                        evaluator.run_analyses(
                            model_names=args.models,
                            benchmark=benchmark,
                            top_k=args.top_k,
                            analyzer_names=["exposure"],
                        )

        if args.generate_report:
            evaluator.generate_report()

        print(f"\n{'='*70}")
        print("EVALUATION COMPLETE")
        print(f"{'='*70}")
        cols = ["model", "benchmark", "n_queries", "AUC", "MRR", "NDCG@10", "HR@10"]
        print(summary_df[cols].to_string(index=False, float_format="%.4f"))
        print(f"\n📁 Results saved in: {evaluator.results_dir}")
        print("  • evaluation_summary.csv - metrics per model x benchmark")
        print("  • <model>__<benchmark>_detailed.parquet - per-candidate scores")
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
