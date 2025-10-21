# scripts/run_unlearning_experiments.py

#!/usr/bin/env python3
"""
Run comprehensive unlearning experiments across multiple configurations.
"""

import argparse
import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive unlearning experiments")

    parser.add_argument(
        "--model-configs", nargs="+", required=True, help="List of model config paths"
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of checkpoint paths (must match model-configs order)",
    )

    parser.add_argument("--data-path", required=True, help="Path to training data CSV")

    parser.add_argument(
        "--ratios", type=float, nargs="+", default=[0.01, 0.05, 0.10], help="Forget ratios to test"
    )

    parser.add_argument("--num-trials", type=int, default=3, help="Number of trials per ratio")

    parser.add_argument(
        "--methods", nargs="+", default=["first_order"], help="Unlearning methods to test"
    )

    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")

    args = parser.parse_args()

    if len(args.model_configs) != len(args.checkpoints):
        print("❌ Number of model-configs must match number of checkpoints")
        return 1

    print(f"\n{'='*70}")
    print("COMPREHENSIVE UNLEARNING EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Models: {len(args.model_configs)}")
    print(f"Methods: {args.methods}")
    print(f"Ratios: {args.ratios}")
    print(f"Trials per ratio: {args.num_trials}")
    print(
        f"Total runs: {len(args.model_configs) * len(args.methods) * len(args.ratios) * args.num_trials}"
    )
    print(f"{'='*70}\n")

    # Run experiments
    experiment_count = 0
    total_experiments = len(args.model_configs) * len(args.methods)

    for model_config, checkpoint in zip(args.model_configs, args.checkpoints):
        model_name = Path(model_config).stem

        for method in args.methods:
            experiment_count += 1

            print(f"\n{'#'*70}")
            print(f"EXPERIMENT {experiment_count}/{total_experiments}")
            print(f"Model: {model_name}")
            print(f"Method: {method}")
            print(f"{'#'*70}\n")

            # Build command
            cmd = (
                [
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
                    "--data-path",
                    args.data_path,
                    "--ratios",
                ]
                + [str(r) for r in args.ratios]
                + ["--num-trials", str(args.num_trials), "--gpu", str(args.gpu)]
            )

            # Run
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f"⚠️  Experiment failed: {model_name} × {method}")
            else:
                print(f"✅ Experiment complete: {model_name} × {method}")

    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
