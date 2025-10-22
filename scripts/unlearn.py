#!/usr/bin/env python3
"""
CLI for machine unlearning.

Supports:
- Loading unlearning config from YAML file
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

from configs import load_config, load_unlearning_config
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
  # ========== CONFIG-BASED UNLEARNING (NEW!) ==========
  
  # Load both model and unlearning configs from YAML
  python scripts/unlearn.py \\
      --model-config configs/experiments/simple/bert_finetune.yaml \\
      --unlearn-config configs/experiments/unlearning/first_order_manual.yaml \\
      --model-checkpoint checkpoints/poisoned.ckpt
  
  # Config-based with overrides
  python scripts/unlearn.py \\
      --model-config configs/experiments/simple/bert_finetune.yaml \\
      --unlearn-config configs/experiments/unlearning/first_order_ratio.yaml \\
      --model-checkpoint checkpoints/poisoned.ckpt \\
      --learning-rate 0.001 \\
      --num-steps 5
  
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

    # ========== Configuration Loading ==========
    config_group = parser.add_argument_group("Configuration (Recommended)")
    config_group.add_argument(
        "--unlearn-config",
        type=str,
        help="Path to unlearning config YAML file (loads all unlearning parameters)",
    )

    # ========== Mode Selection ==========
    parser.add_argument(
        "--mode",
        type=str,
        choices=["manual", "ratio", "multi-ratio"],
        help="Unlearning mode (overrides config if provided)",
    )

    # ========== Model & Method ==========
    parser.add_argument("--model-config", type=str, help="Path to model config YAML file")

    parser.add_argument("--model-checkpoint", type=str, help="Path to model checkpoint (.ckpt)")

    parser.add_argument(
        "--method",
        type=str,
        help="Unlearning method (overrides config if provided, default: first_order)",
    )

    # ========== Manual Mode ==========
    manual_group = parser.add_argument_group("Manual Mode Options")
    manual_group.add_argument(
        "--forget-set", type=str, help="Path to forget set CSV (manual mode, overrides config)"
    )

    manual_group.add_argument(
        "--retain-set", type=str, help="Path to retain set CSV (manual mode, overrides config)"
    )

    # ========== Ratio Mode ==========
    ratio_group = parser.add_argument_group("Ratio Mode Options")
    ratio_group.add_argument(
        "--splits-dir",
        type=str,
        help="Path to splits directory (ratio mode, overrides config). E.g., data/politifact/unlearning_splits/ratio_0_05",
    )

    ratio_group.add_argument(
        "--trial-idx",
        type=int,
        help="Trial index for ratio mode (overrides config, default from config or 0)",
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
        "--removal-strategy",
        type=str,
        choices=["complete", "positive_only", "fake_positive_history"],
        default="fake_positive_history",
        help="Strategy for creating forget/retain splits: "
        "complete (remove all rows), positive_only (remove positive samples only), "
        "fake_positive_history (remove fake-positive contexts)",
    )

    multi_ratio_group.add_argument(
        "--create-splits-only", action="store_true", help="Only create splits, don't run unlearning"
    )

    multi_ratio_group.add_argument(
        "--use-label-correction",
        action="store_true",
        help="Use label correction instead of data removal",
    )

    # ========== Unlearning Hyperparameters (Override Config) ==========
    hyper_group = parser.add_argument_group("Unlearning Hyperparameters (Override Config)")
    hyper_group.add_argument(
        "--learning-rate", type=float, help="Learning rate for unlearning (overrides config)"
    )

    hyper_group.add_argument(
        "--num-steps", type=int, help="Number of unlearning steps (overrides config)"
    )

    hyper_group.add_argument("--damping", type=float, help="Damping factor (overrides config)")

    # ========== Other Options ==========
    parser.add_argument("--gpu", type=int, help="GPU device ID (default: auto-detect)")

    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after unlearning")

    parser.add_argument("--no-save", action="store_true", help="Don't save unlearned model")

    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")

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
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # ========== Load Configuration ==========
    try:
        # Load model config
        model_config = load_config(args.model_config)

        # Load unlearning config (from YAML or create from args)
        unlearn_config = load_or_create_unlearn_config(args, model_config)

        # Determine mode (from args or config)
        mode = args.mode if args.mode else unlearn_config.mode
        method = args.method if args.method else unlearn_config.method

        print(f"Mode: {mode}")
        print(f"Method: {method}")
        print(f"{'='*70}\n")

        # Execute based on mode
        if mode == "manual":
            return manual_mode(args, model_config, unlearn_config, device)

        elif mode == "ratio":
            return ratio_mode(args, model_config, unlearn_config, device)

        elif mode == "multi-ratio":
            return multi_ratio_mode(args, model_config, unlearn_config, device)

        else:
            parser.error(f"Unknown mode: {mode}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def load_or_create_unlearn_config(args, model_config):
    """
    Load unlearning config from YAML or create from command-line args.

    Priority:
    1. Load from --unlearn-config YAML
    2. Apply command-line overrides
    3. If no YAML, create from command-line args
    """
    if args.unlearn_config:
        print(f"Loading unlearning config from: {args.unlearn_config}")
        unlearn_config = load_unlearning_config(args.unlearn_config)
        print(f"✅ Loaded config for method: {unlearn_config.method}\n")

        # Apply command-line overrides
        if args.mode:
            unlearn_config.mode = args.mode
        if args.method:
            unlearn_config.method = args.method
        if args.learning_rate:
            unlearn_config.learning_rate = args.learning_rate
        if args.num_steps:
            unlearn_config.num_steps = args.num_steps
        if args.damping:
            unlearn_config.damping = args.damping
        if args.seed:
            unlearn_config.seed = args.seed

        # Override data paths if provided
        if args.forget_set:
            unlearn_config.forget_set_path = Path(args.forget_set)
        if args.retain_set:
            unlearn_config.retain_set_path = Path(args.retain_set)
        if args.splits_dir:
            unlearn_config.unlearning_splits_dir = Path(args.splits_dir)
        if args.trial_idx is not None:
            unlearn_config.trial_idx = args.trial_idx

        if args.mode or args.learning_rate or args.num_steps:
            print("Applied command-line overrides to config\n")

    else:
        # Create config from command-line args
        print("Creating unlearning config from command-line arguments\n")

        method = args.method if args.method else "first_order"
        mode = args.mode if args.mode else "manual"

        config_kwargs = {
            "method": method,
            "mode": mode,
            "learning_rate": args.learning_rate if args.learning_rate else 0.0005,
            "num_steps": args.num_steps if args.num_steps else 3,
            "damping": args.damping if args.damping else 0.01,
            "seed": args.seed if args.seed else model_config.seed,
        }

        # Add mode-specific parameters
        if mode == "manual":
            if not args.forget_set or not args.retain_set:
                raise ValueError("Manual mode requires --forget-set and --retain-set")
            config_kwargs["forget_set_path"] = Path(args.forget_set)
            config_kwargs["retain_set_path"] = Path(args.retain_set)

        elif mode == "ratio":
            if not args.splits_dir:
                raise ValueError("Ratio mode requires --splits-dir")
            config_kwargs["unlearning_splits_dir"] = Path(args.splits_dir)
            config_kwargs["trial_idx"] = args.trial_idx if args.trial_idx is not None else 0

        unlearn_config = FirstOrderConfig(**config_kwargs)

    return unlearn_config


def manual_mode(args, model_config, unlearn_config, device):
    """Execute manual mode unlearning."""
    # Validate
    if not unlearn_config.forget_set_path or not unlearn_config.retain_set_path:
        print("❌ Manual mode requires forget_set_path and retain_set_path in config")
        return 1

    print("Running MANUAL mode (explicit forget/retain paths)")

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


def ratio_mode(args, model_config, unlearn_config, device):
    """Execute ratio mode unlearning (single ratio, single or multiple trials)."""
    # Validate
    if not unlearn_config.unlearning_splits_dir:
        print("❌ Ratio mode requires unlearning_splits_dir in config")
        return 1

    splits_dir = unlearn_config.unlearning_splits_dir
    if not splits_dir.exists():
        print(f"❌ Splits directory not found: {splits_dir}")
        return 1

    # Check if running multiple trials
    if args.num_trials:
        print(f"Running RATIO mode with {args.num_trials} trials")

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
        trial_idx = unlearn_config.trial_idx
        print(f"Running RATIO mode (single trial: {trial_idx})")

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


def multi_ratio_mode(args, model_config, unlearn_config, device):
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
    removal_strategy = (
        args.removal_strategy if hasattr(args, "removal_strategy") else "fake_positive_history"
    )

    print(f"Running MULTI-RATIO mode")
    print(f"  Ratios: {args.ratios}")
    print(f"  Trials per ratio: {num_trials}")
    print(f"  Removal strategy: {removal_strategy}")
    print(f"  Total runs: {len(args.ratios) * num_trials}")

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
            use_label_correction=args.use_label_correction,  # ADD THIS
        )

        # Get splits directory for this ratio
        ratio_str = f"ratio_{ratio:.2f}".replace(".", "_")
        splits_dir = output_dir / ratio_str

        # Update config for this ratio
        import copy

        ratio_config = copy.deepcopy(unlearn_config)
        ratio_config.mode = "ratio"
        ratio_config.unlearning_splits_dir = splits_dir
        ratio_config.ratio = ratio
        ratio_config.trial_idx = 0  # Will be updated in unlearn_multiple_trials

        # Run multiple trials for this ratio
        ratio_results = unlearn_multiple_trials(
            model_checkpoint=Path(args.model_checkpoint),
            model_config=model_config,
            unlearn_config=ratio_config,
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
    seed = args.seed if args.seed else 42
    removal_strategy = (
        args.removal_strategy if hasattr(args, "removal_strategy") else "fake_positive_history"
    )

    print(f"\n{'='*70}")
    print("CREATING RATIO-BASED SPLITS")
    print(f"{'='*70}")
    print(f"Source: {data_path}")
    print(f"Ratios: {args.ratios}")
    print(f"Trials per ratio: {num_trials}")
    print(f"Removal strategy: {removal_strategy}")
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
            use_label_correction=args.use_label_correction,  # ADD THIS
        )
        print(f"✅ Created {len(trial_dirs)} trials for ratio {ratio}")

    print(f"\n{'='*70}")
    print("SPLITS CREATION COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(
        f"Total splits: {len(args.ratios)} ratios × {num_trials} trials = {len(args.ratios) * num_trials}"
    )
    print(f"Removal strategy: {removal_strategy}")
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
