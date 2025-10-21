# scripts/unlearn.py

#!/usr/bin/env python3
"""
CLI for machine unlearning.

Supports:
- Single unlearning run
- Multiple trials
- Multiple ratios
- Both manual and ratio modes
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_config
from configs.unlearning import FirstOrderConfig
from src.unlearning import (
    unlearn_model,
    unlearn_multiple_trials,
    list_available_methods,
    create_ratio_split,
)
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Machine Unlearning for News Recommendation Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ========== MANUAL MODE (explicit paths) ==========
  
  # Single unlearning run
  python scripts/unlearn.py \\
      --model-config configs/experiments/simple/bert_finetune.yaml \\
      --model-checkpoint checkpoints/poisoned.ckpt \\
      --method first_order \\
      --forget-set data/forget.csv \\
      --retain-set data/retain.csv \\
      --learning-rate 0.0005 \\
      --num-steps 3
  
  # ========== RATIO MODE (pre-generated splits) ==========
  
  # Single trial with specific ratio
  python scripts/unlearn.py \\
      --model-config configs/experiments/simple/bert_finetune.yaml \\
      --model-checkpoint checkpoints/poisoned.ckpt \\
      --method first_order \\
      --mode ratio \\
      --splits-dir data/politifact/unlearning_splits/ratio_0_05 \\
      --trial-idx 0 \\
      --learning-rate 0.0005
  
  # Multiple trials (0, 1, 2)
  python scripts/unlearn.py \\
      --model-config configs/experiments/simple/bert_finetune.yaml \\
      --model-checkpoint checkpoints/poisoned.ckpt \\
      --method first_order \\
      --mode ratio \\
      --splits-dir data/politifact/unlearning_splits/ratio_0_05 \\
      --num-trials 3 \\
      --learning-rate 0.0005
  
  # ========== MULTI-RATIO MODE ==========
  
  # Multiple ratios with multiple trials each
  python scripts/unlearn.py \\
      --model-config configs/experiments/simple/bert_finetune.yaml \\
      --model-checkpoint checkpoints/poisoned.ckpt \\
      --method first_order \\
      --mode multi-ratio \\
      --data-path data/politifact/train_poisoned.csv \\
      --ratios 0.01 0.05 0.10 \\
      --num-trials 3 \\
      --learning-rate 0.0005
  
  # ========== CREATE SPLITS ONLY ==========
  
  # Create ratio-based splits without unlearning
  python scripts/unlearn.py \\
      --create-splits-only \\
      --data-path data/politifact/train_poisoned.csv \\
      --ratios 0.01 0.05 0.10 0.20 \\
      --num-trials 5
  
  # ========== LIST AVAILABLE METHODS ==========
  
  python scripts/unlearn.py --list-methods
        """,
    )

    # ========== Mode Selection ==========
    parser.add_argument(
        "--mode",
        type=str,
        choices=["manual", "ratio", "multi-ratio"],
        default="manual",
        help="Unlearning mode: manual (explicit paths), ratio (single ratio), multi-ratio (multiple ratios)",
    )

    # ========== Model & Method ==========
    parser.add_argument("--model-config", type=str, help="Path to model config YAML file")

    parser.add_argument("--model-checkpoint", type=str, help="Path to model checkpoint (.ckpt)")

    parser.add_argument(
        "--method", type=str, default="first_order", help="Unlearning method (default: first_order)"
    )

    # ========== Manual Mode ==========
    manual_group = parser.add_argument_group("Manual Mode Options")
    manual_group.add_argument("--forget-set", type=str, help="Path to forget set CSV (manual mode)")

    manual_group.add_argument("--retain-set", type=str, help="Path to retain set CSV (manual mode)")

    # ========== Ratio Mode ==========
    ratio_group = parser.add_argument_group("Ratio Mode Options")
    ratio_group.add_argument(
        "--splits-dir",
        type=str,
        help="Path to splits directory (ratio mode). E.g., data/politifact/unlearning_splits/ratio_0_05",
    )

    ratio_group.add_argument(
        "--trial-idx", type=int, default=0, help="Trial index for ratio mode (default: 0)"
    )

    ratio_group.add_argument(
        "--num-trials",
        type=int,
        help="Number of trials to run (ratio/multi-ratio mode). Overrides --trial-idx.",
    )

    # ========== Multi-Ratio Mode ==========
    multi_ratio_group = parser.add_argument_group("Multi-Ratio Mode Options")
    multi_ratio_group.add_argument(
        "--data-path", type=str, help="Path to source data CSV (multi-ratio mode)"
    )

    multi_ratio_group.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        help="Forget ratios to test (multi-ratio mode). E.g., 0.01 0.05 0.10",
    )

    multi_ratio_group.add_argument(
        "--create-splits-only", action="store_true", help="Only create splits, don't run unlearning"
    )

    # ========== Unlearning Hyperparameters ==========
    hyper_group = parser.add_argument_group("Unlearning Hyperparameters")
    hyper_group.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
        help="Learning rate for unlearning (default: 0.0005)",
    )

    hyper_group.add_argument(
        "--num-steps", type=int, default=3, help="Number of unlearning steps (default: 3)"
    )

    hyper_group.add_argument(
        "--damping", type=float, default=0.01, help="Damping factor (default: 0.01)"
    )

    # ========== Other Options ==========
    parser.add_argument("--gpu", type=int, help="GPU device ID (default: auto-detect)")

    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after unlearning")

    parser.add_argument("--no-save", action="store_true", help="Don't save unlearned model")

    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    parser.add_argument(
        "--list-methods", action="store_true", help="List available unlearning methods and exit"
    )

    args = parser.parse_args()

    # ========== List Methods ==========
    if args.list_methods:
        list_available_methods()
        return 0

    # ========== Create Splits Only ==========
    if args.create_splits_only:
        return create_splits_command(args)

    # ========== Validate Required Arguments ==========
    if not args.model_config or not args.model_checkpoint:
        parser.error("--model-config and --model-checkpoint are required")

    # ========== Setup Device ==========
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print("MACHINE UNLEARNING CLI")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Method: {args.method}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # ========== Execute Based on Mode ==========
    try:
        if args.mode == "manual":
            return manual_mode(args, device)

        elif args.mode == "ratio":
            return ratio_mode(args, device)

        elif args.mode == "multi-ratio":
            return multi_ratio_mode(args, device)

        else:
            parser.error(f"Unknown mode: {args.mode}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def manual_mode(args, device):
    """Execute manual mode unlearning."""
    # Validate
    if not args.forget_set or not args.retain_set:
        print("❌ Manual mode requires --forget-set and --retain-set")
        return 1

    print("Running MANUAL mode (explicit forget/retain paths)")

    # Load configs
    model_config = load_config(args.model_config)

    unlearn_config = FirstOrderConfig(
        method=args.method,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        damping=args.damping,
        mode="manual",
        forget_set_path=Path(args.forget_set),
        retain_set_path=Path(args.retain_set),
    )

    # Run unlearning
    results = unlearn_model(
        model_checkpoint=Path(args.model_checkpoint),
        model_config=model_config,
        unlearn_config=unlearn_config,
        device=device,
        evaluate=not args.no_eval,
        save_unlearned=not args.no_save,
    )

    print_final_summary([results])
    return 0


def ratio_mode(args, device):
    """Execute ratio mode unlearning (single ratio, single or multiple trials)."""
    # Validate
    if not args.splits_dir:
        print("❌ Ratio mode requires --splits-dir")
        return 1

    splits_dir = Path(args.splits_dir)
    if not splits_dir.exists():
        print(f"❌ Splits directory not found: {splits_dir}")
        return 1

    # Load model config
    model_config = load_config(args.model_config)

    # Check if running multiple trials
    if args.num_trials:
        print(f"Running RATIO mode with {args.num_trials} trials")

        # Base unlearn config
        unlearn_config = FirstOrderConfig(
            method=args.method,
            learning_rate=args.learning_rate,
            num_steps=args.num_steps,
            damping=args.damping,
            mode="ratio",
            unlearning_splits_dir=splits_dir,
            trial_idx=0,  # Will be updated in unlearn_multiple_trials
        )

        # Run multiple trials
        all_results = unlearn_multiple_trials(
            model_checkpoint=Path(args.model_checkpoint),
            model_config=model_config,
            unlearn_config=unlearn_config,
            num_trials=args.num_trials,
            device=device,
        )

        # Print summary
        print_trial_summary(all_results)
        return 0

    else:
        print(f"Running RATIO mode (single trial: {args.trial_idx})")

        unlearn_config = FirstOrderConfig(
            method=args.method,
            learning_rate=args.learning_rate,
            num_steps=args.num_steps,
            damping=args.damping,
            mode="ratio",
            unlearning_splits_dir=splits_dir,
            trial_idx=args.trial_idx,
        )

        # Run single trial
        results = unlearn_model(
            model_checkpoint=Path(args.model_checkpoint),
            model_config=model_config,
            unlearn_config=unlearn_config,
            device=device,
            evaluate=not args.no_eval,
            save_unlearned=not args.no_save,
        )

        print_final_summary([results])
        return 0


def multi_ratio_mode(args, device):
    """Execute multi-ratio mode (multiple ratios × multiple trials)."""
    # Validate
    if not args.data_path:
        print("❌ Multi-ratio mode requires --data-path")
        return 1

    if not args.ratios:
        print("❌ Multi-ratio mode requires --ratios")
        return 1

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return 1

    num_trials = args.num_trials if args.num_trials else 3

    print(f"Running MULTI-RATIO mode")
    print(f"  Ratios: {args.ratios}")
    print(f"  Trials per ratio: {num_trials}")
    print(f"  Total runs: {len(args.ratios) * num_trials}")

    # Load model config
    model_config = load_config(args.model_config)

    # Create splits directory structure
    output_dir = data_path.parent / "unlearning_splits"

    all_results = {}

    # For each ratio
    for ratio in args.ratios:
        print(f"\n{'#'*70}")
        print(f"RATIO: {ratio} ({ratio*100:.1f}%)")
        print(f"{'#'*70}\n")

        # Create splits for this ratio
        print(f"Creating {num_trials} splits for ratio {ratio}...")
        trial_dirs = create_ratio_split(
            data_path=data_path,
            ratio=ratio,
            num_trials=num_trials,
            output_dir=output_dir,
            seed=args.seed,
        )

        # Get splits directory for this ratio
        ratio_str = f"ratio_{ratio:.2f}".replace(".", "_")
        splits_dir = output_dir / ratio_str

        # Base unlearn config
        unlearn_config = FirstOrderConfig(
            method=args.method,
            learning_rate=args.learning_rate,
            num_steps=args.num_steps,
            damping=args.damping,
            mode="ratio",
            unlearning_splits_dir=splits_dir,
            ratio=ratio,
            trial_idx=0,  # Will be updated
        )

        # Run multiple trials for this ratio
        ratio_results = unlearn_multiple_trials(
            model_checkpoint=Path(args.model_checkpoint),
            model_config=model_config,
            unlearn_config=unlearn_config,
            num_trials=num_trials,
            device=device,
        )

        all_results[ratio] = ratio_results

    # Print comprehensive summary
    print_multi_ratio_summary(all_results, args.ratios, num_trials)
    return 0


def create_splits_command(args):
    """Create splits without unlearning."""
    if not args.data_path:
        print("❌ --data-path is required for creating splits")
        return 1

    if not args.ratios:
        print("❌ --ratios is required for creating splits")
        return 1

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return 1

    num_trials = args.num_trials if args.num_trials else 3

    print(f"\n{'='*70}")
    print("CREATING RATIO-BASED SPLITS")
    print(f"{'='*70}")
    print(f"Source: {data_path}")
    print(f"Ratios: {args.ratios}")
    print(f"Trials per ratio: {num_trials}")
    print(f"{'='*70}\n")

    output_dir = data_path.parent / "unlearning_splits"

    for ratio in args.ratios:
        print(f"\nCreating splits for ratio {ratio}...")
        trial_dirs = create_ratio_split(
            data_path=data_path,
            ratio=ratio,
            num_trials=num_trials,
            output_dir=output_dir,
            seed=args.seed,
        )
        print(f"✅ Created {len(trial_dirs)} trials for ratio {ratio}")

    print(f"\n{'='*70}")
    print("SPLITS CREATION COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(
        f"Total splits: {len(args.ratios)} ratios × {num_trials} trials = {len(args.ratios) * num_trials}"
    )
    print(f"{'='*70}\n")

    return 0


def print_final_summary(results_list):
    """Print summary for single or manual mode runs."""
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    for i, results in enumerate(results_list, 1):
        if len(results_list) > 1:
            print(f"\nRun {i}:")

        if results["evaluation_results"]:
            eval_results = results["evaluation_results"]
            overall = eval_results["overall"]
            forget = eval_results["forget_quality"]
            utility = eval_results["utility"]

            print(f"  Overall Score: {overall['overall_score']:.4f} ({overall['grade']})")
            print(f"  Forget Efficacy: {forget['efficacy']:.4f}")
            print(f"  Retain Accuracy: {utility['retain_after']['accuracy']:.4f}")
            print(f"  Time: {results['unlearning_time']:.2f}s")

        if results["unlearned_checkpoint_path"]:
            print(f"  Saved: {results['unlearned_checkpoint_path']}")

    print(f"{'='*70}\n")


def print_trial_summary(all_results):
    """Print summary for multiple trials (single ratio)."""
    print(f"\n{'='*70}")
    print(f"TRIALS SUMMARY ({len(all_results)} trials)")
    print(f"{'='*70}")

    scores = []
    forget_efficacies = []
    retain_accs = []
    times = []

    for trial_idx, results in all_results.items():
        if results["evaluation_results"]:
            eval_results = results["evaluation_results"]
            scores.append(eval_results["overall"]["overall_score"])
            forget_efficacies.append(eval_results["forget_quality"]["efficacy"])
            retain_accs.append(eval_results["utility"]["retain_after"]["accuracy"])
            times.append(results["unlearning_time"])

    print(f"\nOverall Score:")
    print(f"  Mean: {sum(scores)/len(scores):.4f} ± {_std(scores):.4f}")
    print(f"  Range: [{min(scores):.4f}, {max(scores):.4f}]")

    print(f"\nForget Efficacy:")
    print(
        f"  Mean: {sum(forget_efficacies)/len(forget_efficacies):.4f} ± {_std(forget_efficacies):.4f}"
    )

    print(f"\nRetain Accuracy:")
    print(f"  Mean: {sum(retain_accs)/len(retain_accs):.4f} ± {_std(retain_accs):.4f}")

    print(f"\nTime per Trial:")
    print(f"  Mean: {sum(times)/len(times):.2f}s ± {_std(times):.2f}s")

    print(f"{'='*70}\n")


def print_multi_ratio_summary(all_results, ratios, num_trials):
    """Print comprehensive summary for multi-ratio mode."""
    print(f"\n{'='*70}")
    print(f"MULTI-RATIO SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {len(ratios)} ratios × {num_trials} trials = {len(ratios) * num_trials} runs")
    print(f"{'='*70}\n")

    # Create summary table
    print(f"{'Ratio':<10} {'Score':<15} {'Forget Eff':<15} {'Retain Acc':<15} {'Time (s)':<10}")
    print(f"{'-'*70}")

    for ratio in ratios:
        ratio_results = all_results[ratio]

        scores = []
        forget_efficacies = []
        retain_accs = []
        times = []

        for trial_idx, results in ratio_results.items():
            if results["evaluation_results"]:
                eval_results = results["evaluation_results"]
                scores.append(eval_results["overall"]["overall_score"])
                forget_efficacies.append(eval_results["forget_quality"]["efficacy"])
                retain_accs.append(eval_results["utility"]["retain_after"]["accuracy"])
                times.append(results["unlearning_time"])

        mean_score = sum(scores) / len(scores)
        mean_forget = sum(forget_efficacies) / len(forget_efficacies)
        mean_retain = sum(retain_accs) / len(retain_accs)
        mean_time = sum(times) / len(times)

        print(
            f"{ratio:<10.2f} {mean_score:.4f}±{_std(scores):.4f}  "
            f"{mean_forget:.4f}±{_std(forget_efficacies):.4f}  "
            f"{mean_retain:.4f}±{_std(retain_accs):.4f}  "
            f"{mean_time:>8.2f}"
        )

    print(f"{'='*70}\n")

    # Find best ratio
    best_ratio = None
    best_score = 0.0

    for ratio in ratios:
        ratio_results = all_results[ratio]
        scores = [
            r["evaluation_results"]["overall"]["overall_score"]
            for r in ratio_results.values()
            if r["evaluation_results"]
        ]
        mean_score = sum(scores) / len(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_ratio = ratio

    print(f"🏆 Best Ratio: {best_ratio} (Score: {best_score:.4f})")
    print(f"{'='*70}\n")


def _std(values):
    """Calculate standard deviation."""
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance**0.5


if __name__ == "__main__":
    sys.exit(main())
