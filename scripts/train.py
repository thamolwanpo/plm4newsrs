#!/usr/bin/env python3
# ============================================================
# scripts/train.py
# Main training script with YAML config support
# ============================================================

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_config, create_config
from src.training import train_model, train_all_models, train_reference_models
from src.training.trainer import skip_or_prepare_run
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Train news recommendation model with YAML configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean model (stage3/train_clean_model.csv)
  python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml --model-type clean

  # One poisoned model (stage4/A_r0.5_s1/train_model.csv)
  python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml --poison-set A_r0.5_s1

  # Clean models on three consecutive seeds + every poisoned set found in stage4/
  python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml --train-all

  # Only the clean references of every poisoned set (no clean / poisoned model):
  # oracle + retrain-without-forget-set per flag level, three seeds each
  python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml --train-references

  # Only the oracle of three poisoned sets
  python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml --train-references \\
      --reference-types oracle --poison-sets A_r0.5_s1 A_r0.5_s2 A_r0.5_s3

  # One retrain-without-forget-set model (flags/flag_050/clean_flagged_model.csv)
  python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml \\
      --model-type retrain --poison-set A_r0.5_s1 --flag-level 50 --seed 42

  # Point at the fakeun-dataprep output and override hyperparameters
  python scripts/train.py --config configs/experiments/nrms/nrms_glove.yaml --train-all \\
      --data-root ../fakeun-dataprep/output/problem2/politifact --epochs 5
        """,
    )

    # ========== Config Selection ==========
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", type=str, help="Path to YAML config file")
    config_group.add_argument(
        "--architecture",
        type=str,
        choices=["simple", "naml", "nrms", "lstur", "lightgcn"],
        help="Model architecture (if not using config file)",
    )

    # ========== Basic Options ==========
    parser.add_argument(
        "--model-name", type=str, help="Model name: bert-base-uncased, roberta-base, glove, etc."
    )
    parser.add_argument("--dataset", type=str, help="Dataset name (overrides config)")
    parser.add_argument("--experiment-name", type=str, help="Experiment name (overrides config)")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["clean", "poisoned", "oracle", "retrain"],
        help="clean (stage3/train_clean_model.csv), poisoned (stage4/<poison-set>/train_model.csv), "
        "oracle (stage4/<poison-set>/clean_oracle_model.csv) or retrain "
        "(stage4/<poison-set>/flags/<flag-level>/clean_flagged_model.csv)",
    )
    parser.add_argument(
        "--poison-set",
        type=str,
        help="Poisoned training set, e.g. A_r0.5_s1 (with --model-type poisoned)",
    )
    parser.add_argument(
        "--poison-sets",
        type=str,
        nargs="+",
        help="Poisoned sets trained by --train-all / --train-references "
        "(default: every stage4/A_* folder)",
    )
    parser.add_argument(
        "--flag-level",
        type=str,
        help="Flag level of a retrain model: 10, 20, 50, 80, 100 or flag_050",
    )
    parser.add_argument(
        "--flag-levels",
        type=str,
        nargs="+",
        help="Flag levels trained by --train-references (default: every level found; "
        "flag_100 only with --include-flag-100)",
    )

    # ========== Training Options ==========
    parser.add_argument("--epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, help="Training batch size (overrides config)")
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        dest="learning_rate",
        help="Learning rate (overrides config)",
    )
    parser.add_argument("--patience", type=int, help="Early stopping patience (overrides config)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")

    # ========== Path Options ==========
    parser.add_argument(
        "--base-dir", type=str, help="Base directory for experiment (overrides config)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        help="fakeun-dataprep output folder of the dataset, e.g. output/problem2/politifact",
    )

    # ========== GloVe Options ==========
    parser.add_argument(
        "--glove-path", type=str, help="Path to GloVe embeddings file (overrides config)"
    )
    parser.add_argument(
        "--glove-dim",
        type=int,
        choices=[50, 100, 200, 300],
        help="GloVe embedding dimension (overrides config)",
    )

    # ========== Model-Specific Options ==========
    parser.add_argument(
        "--fine-tune", action="store_true", help="Fine-tune language model (overrides config)"
    )
    parser.add_argument(
        "--freeze", action="store_true", help="Freeze language model weights (overrides config)"
    )

    # ========== Training Mode ==========
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train the clean model on three consecutive seeds, plus one model per poisoned set",
    )
    parser.add_argument(
        "--train-references",
        action="store_true",
        help="Train only the clean references of every poisoned set: the oracle and one "
        "retrain model per flag level, on --num-seeds seeds each (no clean / poisoned model)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Seeds per training file for --train-references, starting at seed (default: 3)",
    )
    parser.add_argument(
        "--reference-types",
        type=str,
        nargs="+",
        choices=["oracle", "retrain"],
        default=["oracle", "retrain"],
        help="--train-references: which references to train (default: both), "
        "e.g. --reference-types oracle",
    )
    parser.add_argument(
        "--include-flag-100",
        action="store_true",
        help="--train-references: also train flag_100, whose file is identical to the oracle's",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name of a single run's outputs (default: 'clean' or 'poisoned_{set}'); "
        "e.g. clean_seed43 to train one of --train-all's runs on its own",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already finished (last checkpoint + presentation log exist); "
        "unfinished runs are retrained from scratch. Rerun the same command after a crash",
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")

    # ========== Device Options ==========
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU device ID (default: auto-detect)"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")

    # ========== Other Options ==========
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--dry-run", action="store_true", help="Print config without training")

    args = parser.parse_args()

    # ========== Load or Create Config ==========
    if args.config:
        if args.verbose:
            print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    elif args.architecture:
        if not args.model_name:
            parser.error("--model-name is required when using --architecture")
        if args.verbose:
            print(f"Creating {args.architecture} config for {args.model_name}")

        config_kwargs = {"model_name": args.model_name}
        if args.dataset:
            config_kwargs["dataset"] = args.dataset

        config = create_config(args.architecture, **config_kwargs)
    else:
        parser.error("Either --config or --architecture is required")

    # ========== Apply Overrides ==========
    overrides = {}

    # Basic options
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.experiment_name:
        overrides["experiment_name"] = args.experiment_name
    if args.model_type:
        overrides["model_type"] = args.model_type
    if args.poison_set:
        overrides["poison_set"] = args.poison_set
        overrides.setdefault("model_type", "poisoned")
    if args.poison_sets:
        overrides["poison_sets"] = args.poison_sets
    if args.flag_level:
        overrides["flag_level"] = args.flag_level
    if args.flag_levels:
        overrides["flag_levels"] = args.flag_levels

    # Training options
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["train_batch_size"] = args.batch_size
    if args.learning_rate:
        overrides["learning_rate"] = args.learning_rate
    if args.patience:
        overrides["early_stopping_patience"] = args.patience
    if args.seed:
        overrides["seed"] = args.seed

    # Path options
    if args.base_dir:
        overrides["base_dir"] = Path(args.base_dir)
    if args.data_root:
        overrides["data_root"] = Path(args.data_root)

    # GloVe options
    if args.glove_path:
        overrides["glove_file_path"] = args.glove_path
    if args.glove_dim:
        overrides["glove_dim"] = args.glove_dim

    # Fine-tune/freeze
    if args.fine_tune and args.freeze:
        parser.error("Cannot use both --fine-tune and --freeze")
    if args.fine_tune:
        overrides["fine_tune_lm"] = True
    if args.freeze:
        overrides["fine_tune_lm"] = False

    # Apply overrides
    if overrides:
        if args.verbose:
            print(f"\nApplying overrides: {overrides}")
        if "dataset" in overrides:
            # Re-derive default paths that were based on the old dataset name
            if config.base_dir == Path(f"./outputs/{config.dataset}"):
                config.base_dir = None
            if config.data_root == Path(f"data/problem2/{config.dataset}"):
                config.data_root = None
        for key, value in overrides.items():
            setattr(config, key, value)
        config.__post_init__()

    # ========== Setup Device ==========
    if args.cpu:
        device = torch.device("cpu")
    elif args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"Using device: {device}")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    print(f"{'='*70}")

    # ========== Print Config ==========
    config.print_config()

    # ========== Dry Run ==========
    if args.dry_run:
        print("\n[DRY RUN] Config validated. Exiting without training.")
        return 0

    # ========== Train ==========
    try:
        if args.resume:
            print(f"\n⚠️ Resume training not yet implemented")
            print(f"Checkpoint: {args.resume}")
            return 1

        elif args.train_all and args.train_references:
            parser.error("Use --train-all or --train-references, not both")

        elif args.train_references:
            if args.run_name:
                parser.error("--run-name names a single run; it cannot be used with --train-references")
            print("\n" + "=" * 70)
            print("TRAINING CLEAN REFERENCES (oracle + retrain per flag level)")
            print("=" * 70)
            trained_models = train_reference_models(
                config,
                device,
                skip_existing=args.skip_existing,
                num_seeds=args.num_seeds,
                include_flag_100=args.include_flag_100,
                reference_types=args.reference_types,
            )
            if not trained_models:
                return 1

            print("\n" + "=" * 70)
            print("TRAINING SUMMARY")
            print("=" * 70)
            for run_name, checkpoint_path in trained_models.items():
                print(f"✅ {run_name}: {checkpoint_path}")

        elif args.train_all:
            if args.run_name:
                parser.error("--run-name names a single run; it cannot be used with --train-all")
            print("\n" + "=" * 70)
            print("TRAINING ALL MODEL TYPES")
            print("=" * 70)
            trained_models = train_all_models(config, device, skip_existing=args.skip_existing)

            print("\n" + "=" * 70)
            print("TRAINING SUMMARY")
            print("=" * 70)
            for run_name, checkpoint_path in trained_models.items():
                print(f"✅ {run_name}: {checkpoint_path}")

        else:
            print("\n" + "=" * 70)
            print("STARTING TRAINING")
            print("=" * 70)
            run_name = args.run_name or config.run_name
            checkpoint_path = None
            if args.skip_existing:
                checkpoint_path = skip_or_prepare_run(config, run_name)
            if checkpoint_path is None:
                checkpoint_path = train_model(config, device, run_name=run_name)

            if checkpoint_path:
                print("\n" + "=" * 70)
                print("TRAINING COMPLETE")
                print("=" * 70)
                print(f"✅ Model saved: {checkpoint_path}")
            else:
                print("\n❌ Training failed or was skipped")
                return 1

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
