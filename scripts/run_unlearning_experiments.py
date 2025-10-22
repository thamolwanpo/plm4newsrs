#!/usr/bin/env python3
"""
Run comprehensive unlearning experiments across multiple configurations.

Supports:
- Multi-ratio × multi-trial experiments
- Multiple models (architectures/embeddings)
- Multiple unlearning methods
- Config-based or command-line based experiments
- Automatic result aggregation
"""

import argparse
import sys
from pathlib import Path
import subprocess
import json
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive unlearning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: Single model, multiple ratios and trials
  python scripts/run_unlearning_experiments.py \\
      --model-configs configs/experiments/simple/bert_finetune.yaml \\
      --checkpoints outputs/simple_bert_finetune/checkpoints/poisoned.ckpt \\
      --data-path data/politifact/train_poisoned.csv \\
      --ratios 0.01 0.05 0.10 0.20 \\
      --num-trials 5

  # Advanced: Multiple models × multiple methods × multiple ratios × multiple trials
  python scripts/run_unlearning_experiments.py \\
      --model-configs configs/experiments/simple/bert_finetune.yaml \\
                      configs/experiments/simple/roberta_finetune.yaml \\
                      configs/experiments/simple/glove.yaml \\
      --checkpoints outputs/simple_bert_finetune/checkpoints/poisoned.ckpt \\
                    outputs/simple_roberta_finetune/checkpoints/poisoned.ckpt \\
                    outputs/simple_glove/checkpoints/poisoned.ckpt \\
      --data-path data/politifact/train_poisoned.csv \\
      --ratios 0.01 0.05 0.10 0.20 0.50 \\
      --num-trials 5 \\
      --methods first_order gradient_ascent

  # Using unlearning configs (recommended for reproducibility)
  python scripts/run_unlearning_experiments.py \\
      --model-configs configs/experiments/simple/bert_finetune.yaml \\
      --checkpoints outputs/simple_bert_finetune/checkpoints/poisoned.ckpt \\
      --unlearn-configs configs/experiments/unlearning/first_order_ratio.yaml \\
      --ratios 0.01 0.05 0.10 \\
      --num-trials 5

  # Quick test with fewer trials
  python scripts/run_unlearning_experiments.py \\
      --model-configs configs/experiments/simple/bert_finetune.yaml \\
      --checkpoints outputs/simple_bert_finetune/checkpoints/poisoned.ckpt \\
      --data-path data/politifact/train_poisoned.csv \\
      --ratios 0.05 0.10 \\
      --num-trials 2 \\
      --quick-test
        """,
    )

    # ========== Model Configuration ==========
    parser.add_argument(
        "--model-configs",
        nargs="+",
        required=True,
        help="List of model config paths (e.g., configs/experiments/simple/bert_finetune.yaml)",
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of checkpoint paths (must match model-configs order)",
    )

    # ========== Unlearning Configuration ==========
    unlearn_group = parser.add_mutually_exclusive_group()
    unlearn_group.add_argument(
        "--unlearn-configs",
        nargs="+",
        help="List of unlearning config paths (if provided, uses config-based approach)",
    )

    unlearn_group.add_argument(
        "--data-path", help="Path to training data CSV (for command-line based approach)"
    )

    # ========== Experiment Parameters ==========
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10],
        help="Forget ratios to test (default: 0.01 0.05 0.10)",
    )

    parser.add_argument(
        "--num-trials", type=int, default=3, help="Number of trials per ratio (default: 3)"
    )

    parser.add_argument(
        "--methods", nargs="+", default=["first_order"], help="Unlearning methods to test"
    )

    # ========== Removal Strategy ==========
    parser.add_argument(
        "--removal-strategy",
        type=str,
        choices=["complete", "positive_only", "fake_positive_history"],
        default="fake_positive_history",
        help="Data removal strategy (default: fake_positive_history)",
    )

    parser.add_argument(
        "--use-label-correction",
        action="store_true",
        help="Use label correction instead of data removal",
    )

    # ========== Hyperparameter Overrides ==========
    hyper_group = parser.add_argument_group("Hyperparameter Overrides (optional)")
    hyper_group.add_argument("--learning-rate", type=float, help="Override learning rate")
    hyper_group.add_argument("--num-steps", type=int, help="Override number of unlearning steps")
    hyper_group.add_argument("--damping", type=float, help="Override damping factor")

    # ========== Execution Options ==========
    parser.add_argument("--gpu", type=int, help="GPU device ID (omit for auto-detect)")

    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel (experimental, not yet implemented)",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining experiments if one fails",
    )

    # ========== Output Options ==========
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/unlearning_experiments",
        help="Output directory for results (default: outputs/unlearning_experiments)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name (default: auto-generated from timestamp)",
    )

    parser.add_argument("--save-summary", action="store_true", help="Save aggregated summary CSV")

    # ========== Testing ==========
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (single ratio, 2 trials, no evaluation)",
    )

    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()

    # ========== Validation ==========
    if len(args.model_configs) != len(args.checkpoints):
        print("❌ Number of model-configs must match number of checkpoints")
        return 1

    if args.unlearn_configs and len(args.unlearn_configs) != len(args.methods):
        print(
            "⚠️  Warning: Number of unlearn-configs doesn't match methods. Using first config for all."
        )

    if not args.unlearn_configs and not args.data_path:
        print("❌ Either --unlearn-configs or --data-path is required")
        return 1

    # ========== Quick Test Mode ==========
    if args.quick_test:
        print("⚡ QUICK TEST MODE")
        args.ratios = [args.ratios[0]] if args.ratios else [0.05]
        args.num_trials = 2
        print(f"  - Using single ratio: {args.ratios[0]}")
        print(f"  - Using 2 trials")
        print()

    # ========== Setup Experiment ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"unlearning_exp_{timestamp}"
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total experiments
    total_experiments = (
        len(args.model_configs) * len(args.methods) * len(args.ratios) * args.num_trials
    )

    # ========== Print Summary ==========
    print(f"\n{'='*70}")
    print("COMPREHENSIVE UNLEARNING EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Experiment Name: {experiment_name}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*70}")
    print(f"Models: {len(args.model_configs)}")
    for cfg in args.model_configs:
        print(f"  - {Path(cfg).stem}")
    print(f"Methods: {args.methods}")
    print(f"Ratios: {args.ratios} ({len(args.ratios)} ratios)")
    print(f"Trials per ratio: {args.num_trials}")
    print(f"Removal Strategy: {args.removal_strategy}")
    if args.use_label_correction:
        print(f"Label Correction: ENABLED")
    print(f"Total runs: {total_experiments}")
    print(f"{'='*70}\n")

    if args.dry_run:
        print("🔍 DRY RUN MODE - Commands will be printed but not executed\n")

    # ========== Save Experiment Config ==========
    experiment_config = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "model_configs": args.model_configs,
        "checkpoints": args.checkpoints,
        "unlearn_configs": args.unlearn_configs,
        "data_path": args.data_path,
        "ratios": args.ratios,
        "num_trials": args.num_trials,
        "methods": args.methods,
        "removal_strategy": args.removal_strategy,
        "use_label_correction": args.use_label_correction,
        "gpu": args.gpu,
        "seed": args.seed,
    }

    config_file = output_dir / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump(experiment_config, f, indent=2)

    print(f"📝 Experiment config saved: {config_file}\n")

    # ========== Run Experiments ==========
    experiment_count = 0
    failed_experiments = []
    results_summary = []

    for model_config, checkpoint in zip(args.model_configs, args.checkpoints):
        model_name = Path(model_config).stem

        # Verify checkpoint exists
        if not Path(checkpoint).exists():
            print(f"❌ Checkpoint not found: {checkpoint}")
            if not args.continue_on_error:
                return 1
            failed_experiments.append(f"{model_name}: checkpoint not found")
            continue

        for method_idx, method in enumerate(args.methods):
            experiment_count += 1

            print(f"\n{'#'*70}")
            print(f"EXPERIMENT {experiment_count}/{len(args.model_configs) * len(args.methods)}")
            print(f"Model: {model_name}")
            print(f"Method: {method}")
            print(f"Ratios: {args.ratios} ({len(args.ratios)} ratios)")
            print(f"Trials: {args.num_trials} per ratio")
            print(f"Total runs: {len(args.ratios) * args.num_trials}")
            print(f"{'#'*70}\n")

            # Build command
            cmd = [
                "python",
                "scripts/unlearn.py",
                "--model-config",
                model_config,
                "--model-checkpoint",
                checkpoint,
                "--method",
                method,
                "--mode",
                "multi-ratio",
                "--ratios",
            ] + [str(r) for r in args.ratios]

            # Add data path or unlearn config
            if args.unlearn_configs:
                unlearn_config = (
                    args.unlearn_configs[method_idx]
                    if method_idx < len(args.unlearn_configs)
                    else args.unlearn_configs[0]
                )
                cmd += ["--unlearn-config", unlearn_config]
            else:
                cmd += ["--data-path", args.data_path]

            # Add other parameters
            cmd += [
                "--num-trials",
                str(args.num_trials),
                "--removal-strategy",
                args.removal_strategy,
                "--seed",
                str(args.seed),
            ]

            # Add GPU/CPU options
            if args.cpu:
                # Don't pass --gpu when CPU is requested
                pass
            elif args.gpu is not None:
                cmd += ["--gpu", str(args.gpu)]
            # If neither specified, let unlearn.py auto-detect

            # Add hyperparameter overrides
            if args.learning_rate:
                cmd += ["--learning-rate", str(args.learning_rate)]
            if args.num_steps:
                cmd += ["--num-steps", str(args.num_steps)]
            if args.damping:
                cmd += ["--damping", str(args.damping)]

            # Add label correction flag
            if args.use_label_correction:
                cmd.append("--use-label-correction")

            # Print command
            print(f"Command: {' '.join(cmd)}\n")

            if args.dry_run:
                print("⏭️  Skipping execution (dry run)\n")
                continue

            # Run
            result = subprocess.run(cmd)

            if result.returncode != 0:
                error_msg = f"{model_name} × {method}"
                print(f"⚠️  Experiment failed: {error_msg}")
                failed_experiments.append(error_msg)

                if not args.continue_on_error:
                    print("\n❌ Stopping due to error (use --continue-on-error to continue)")
                    return 1
            else:
                print(f"✅ Experiment complete: {model_name} × {method}")

                # Record success
                results_summary.append(
                    {
                        "model": model_name,
                        "method": method,
                        "checkpoint": checkpoint,
                        "ratios": args.ratios,
                        "num_trials": args.num_trials,
                        "status": "success",
                    }
                )

    # ========== Final Summary ==========
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"Total experiments: {len(args.model_configs) * len(args.methods)}")
    print(f"Successful: {len(results_summary)}")
    print(f"Failed: {len(failed_experiments)}")
    print(f"{'='*70}\n")

    if failed_experiments:
        print("❌ Failed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")
        print()

    print(f"📁 Experiment results saved in: {output_dir}")
    print(f"📝 Experiment config: {config_file}")

    # ========== Save Summary ==========
    if args.save_summary and results_summary:
        summary_file = output_dir / "experiments_summary.csv"
        df = pd.DataFrame(results_summary)
        df.to_csv(summary_file, index=False)
        print(f"📊 Summary saved: {summary_file}")

    print(f"{'='*70}\n")

    return 1 if failed_experiments and not args.continue_on_error else 0


if __name__ == "__main__":
    sys.exit(main())
