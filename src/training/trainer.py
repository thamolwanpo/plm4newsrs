import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import lightning as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.data import NewsDataset
from src.data.collate import collate_fn
from src.data.preprocessing import build_user_map, read_model_csv
from src.models.registry import get_model, get_model_class
from src.training.dynamics import TrainingDynamicsTracker
from src.training.presentation_log import PresentationLogger
from src.utils.seed import set_seed


class LastEpochCheckpoint(Callback):
    """Save the final model state under a stable, run-specific checkpoint name."""

    def __init__(self, checkpoints_dir: Path, run_name: str):
        self.checkpoint_path = Path(checkpoints_dir) / f"{run_name}-last.ckpt"

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.checkpoint_path))


def train_model(
    config,
    device: Optional[torch.device] = None,
    return_trainer: bool = False,
    run_name: Optional[str] = None,
) -> str:
    """
    Train a single model.

    Args:
        config: Model configuration (ModelConfig instance)
        device: Device to train on (auto-detected if None)
        return_trainer: If True, return trainer instead of checkpoint path
        run_name: Optional name for this training run (defaults to config.run_name)

    Returns:
        Path to the last-epoch checkpoint (or trainer if return_trainer=True)
    """
    # Set seed for reproducibility
    set_seed(config.seed)
    run_name = run_name or config.run_name  # validates model_type / poison_set

    # Print configuration
    config.print_config()

    # Get paths
    paths = config.get_paths()

    # Check if data files exist
    if not paths["train_csv"].exists() or not paths["val_csv"].exists():
        print(f"\n❌ Missing data files:")
        print(f"  Train: {paths['train_csv']} (exists: {paths['train_csv'].exists()})")
        print(f"  Val: {paths['val_csv']} (exists: {paths['val_csv'].exists()})")
        return None

    # Determine if using GloVe
    use_glove = "glove" in config.model_name.lower()

    # Load tokenizer (only for transformer models)
    if use_glove:
        print("\nUsing GloVe embeddings - no tokenizer needed")
        tokenizer = None
    else:
        print(f"\nLoading tokenizer: {config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # LSTUR: index the users of this training set (unknown users -> 0)
    user_map = None
    if config.architecture == "lstur":
        user_map = build_user_map(read_model_csv(paths["train_csv"])["user_id"])
        map_path = user_map_path(paths["checkpoints_dir"], run_name)
        with open(map_path, "w") as f:
            json.dump(user_map, f)
        print(f"  Saved LSTUR user map ({len(user_map)} users): {map_path}")
        config.num_users = len(user_map) + 1  # + index 0 for unknown users

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = NewsDataset(paths["train_csv"], config, tokenizer, user_map=user_map)
    val_dataset = NewsDataset(paths["val_csv"], config, tokenizer, user_map=user_map)

    # Create dataloaders
    num_workers = min(4, os.cpu_count() // 2) if os.cpu_count() else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Same batch size as training, unshuffled: only used for the EL2N pass after epoch 1
    el2n_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )

    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Initialize model
    print("\nInitializing model...")
    # Select appropriate Lightning module based on architecture
    lit_model = get_model(config.architecture, config)

    # Setup logging
    logger = TensorBoardLogger(save_dir=paths["logs_dir"], name=f"logs_{run_name}")

    # Setup callbacks
    last_epoch_checkpoint = LastEpochCheckpoint(paths["checkpoints_dir"], run_name)
    dynamics_tracker = TrainingDynamicsTracker(el2n_loader)
    presentation_logger = PresentationLogger(train_dataset.item_ids, train_dataset.item_is_fake)

    early_stopping = EarlyStopping(
        monitor="val_auc", patience=config.early_stopping_patience, verbose=True, mode="max"
    )

    # Auto-detect device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    callbacks = [last_epoch_checkpoint, early_stopping, dynamics_tracker, presentation_logger]
    if config.model_type == "clean":
        # Clean models keep a checkpoint of every epoch (all are evaluated)
        callbacks.append(epoch_checkpoints(paths["checkpoints_dir"], run_name))

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=config.gradient_clip_val,
        deterministic=True,  # For reproducibility
    )

    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    trainer.fit(lit_model, train_loader, val_loader)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"✅ Last-epoch model saved: {last_epoch_checkpoint.checkpoint_path}")

    # Per-instance loss, C-proxy and EL2N ledgers, keyed by training CSV row
    dynamics_path = dynamics_tracker.save(training_dynamics_path(paths["results_dir"], run_name))
    print(f"✅ Training dynamics saved: {dynamics_path}")

    # Every presentation of every group, plus per-article gradient sums by role
    log_path = presentation_logger.save(presentation_log_path(paths["results_dir"], run_name))
    print(f"✅ Presentation log saved: {log_path}")

    if return_trainer:
        return trainer

    return str(last_epoch_checkpoint.checkpoint_path)


def epoch_checkpoints(checkpoints_dir: Path, run_name: str) -> ModelCheckpoint:
    """Save every epoch as {run_name}-epoch=NN-val_auc=X.XXXX.ckpt (epoch is 0-based)."""
    return ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=f"{run_name}-epoch={{epoch:02d}}-val_auc={{val_auc:.4f}}",
        auto_insert_metric_name=False,
        monitor="val_auc",
        mode="max",
        save_top_k=-1,
        every_n_epochs=1,
    )


def user_map_path(checkpoints_dir: Path, run_name: str) -> Path:
    """Where the LSTUR user map of a trained model is stored."""
    return Path(checkpoints_dir) / f"{run_name}_user_map.json"


def training_dynamics_path(results_dir: Path, run_name: str) -> Path:
    """Where the training dynamics ledgers of a training run are stored."""
    return Path(results_dir) / f"{run_name}_training_dynamics.pt"


def presentation_log_path(results_dir: Path, run_name: str) -> Path:
    """Where the per-presentation training log of a training run is stored."""
    return Path(results_dir) / f"{run_name}_presentation_log.pt"


def is_run_complete(paths: Dict[str, Path], run_name: str) -> bool:
    """True if a run finished: its last checkpoint and its end-of-training
    presentation log (the last file train_model writes) both exist."""
    return (Path(paths["checkpoints_dir"]) / f"{run_name}-last.ckpt").exists() and (
        presentation_log_path(paths["results_dir"], run_name).exists()
    )


def remove_partial_checkpoints(paths: Dict[str, Path], run_name: str) -> None:
    """Delete the checkpoints an interrupted run left behind, so retraining it
    from scratch does not leave stale epoch checkpoints for the evaluator."""
    for ckpt in sorted(Path(paths["checkpoints_dir"]).glob(f"{run_name}-*.ckpt")):
        print(f"  Removing checkpoint of unfinished run: {ckpt}")
        ckpt.unlink()


def skip_or_prepare_run(config, run_name: str) -> Optional[str]:
    """For --skip-existing: the last checkpoint if the run already finished
    (skip it), else None after clearing the run's leftover checkpoints."""
    paths = config.get_paths()
    if is_run_complete(paths, run_name):
        checkpoint = str(Path(paths["checkpoints_dir"]) / f"{run_name}-last.ckpt")
        print(f"⏭️  Skipping {run_name}: already trained ({checkpoint})")
        return checkpoint
    remove_partial_checkpoints(paths, run_name)
    return None


def get_training_runs(config) -> List[dict]:
    """Runs of --train-all: three clean seeds + one model per poisoned set.

    Each run is a dict of config overrides plus its run_name.
    """
    poison_sets = config.poison_sets or config.discover_poison_sets()
    clean_runs = [
        {"model_type": "clean", "poison_set": None, "flag_level": None, "seed": seed,
         "run_name": f"clean_seed{seed}"}
        for seed in range(config.seed, config.seed + 3)
    ]
    poisoned_runs = [
        {"model_type": "poisoned", "poison_set": poison_set, "flag_level": None,
         "seed": config.seed, "run_name": f"poisoned_{poison_set}"}
        for poison_set in poison_sets
    ]
    return clean_runs + poisoned_runs


def get_reference_runs(config, num_seeds: int = 3, include_flag_100: bool = False) -> List[dict]:
    """Runs of --train-references, per poisoned set: the oracle, then the
    retrain model of each flag level, each on num_seeds consecutive seeds."""
    poison_sets = config.poison_sets or config.discover_poison_sets()
    seeds = range(config.seed, config.seed + num_seeds)
    runs = []
    for poison_set in poison_sets:
        overrides = []
        if (config.poison_set_dir(poison_set) / "clean_oracle_model.csv").exists():
            overrides.append({"model_type": "oracle", "poison_set": poison_set,
                              "flag_level": None})
        else:
            print(f"⚠️ {poison_set}: no clean_oracle_model.csv - oracle skipped")
        available = config.discover_flag_levels(poison_set)
        for flag_level in config.flag_levels or available:
            if flag_level == "flag_100" and not include_flag_100:
                continue  # identical to the oracle's training file
            if flag_level not in available:
                print(f"⚠️ {poison_set}: no {flag_level}/clean_flagged_model.csv - skipped")
                continue
            overrides.append({"model_type": "retrain", "poison_set": poison_set,
                              "flag_level": flag_level})
        for override in overrides:
            for seed in seeds:
                run_config = copy.copy(config)
                for key, value in override.items():
                    setattr(run_config, key, value)
                run_config.seed = seed
                runs.append({**override, "seed": seed, "run_name": run_config.run_name})
    return runs


def train_all_models(
    config, device: Optional[torch.device] = None, skip_existing: bool = False
) -> Dict[str, str]:
    """
    Train three clean seeds and every poisoned model.

    Args:
        config: Base configuration (copied for each run)
        device: Device to train on (auto-detected if None)
        skip_existing: Skip runs that already finished; an unfinished run has
            its leftover checkpoints removed and is trained again from scratch

    Returns:
        Dictionary mapping run name to checkpoint path
    """
    runs = get_training_runs(config)
    if not any(run["model_type"] == "poisoned" for run in runs):
        print(f"⚠️ No poisoned sets found in {config.data_root / 'stage4'}")
    return train_runs(config, runs, device, skip_existing)


def train_reference_models(
    config,
    device: Optional[torch.device] = None,
    skip_existing: bool = False,
    num_seeds: int = 3,
    include_flag_100: bool = False,
) -> Dict[str, str]:
    """
    Train the clean references of every poisoned set, on num_seeds seeds:
    the oracle (clean_oracle_model.csv) and one retrain-without-forget-set
    model per flag level (clean_flagged_model.csv). Clean and poisoned
    models are not trained.

    Args:
        config: Base configuration (copied for each run)
        device: Device to train on (auto-detected if None)
        skip_existing: see train_all_models
        num_seeds: seeds per training file, config.seed, config.seed + 1, ...
        include_flag_100: also train flag_100, whose file is identical to the
            oracle's (skipped by default: the oracle is the flag_100 retrain)

    Returns:
        Dictionary mapping run name to checkpoint path
    """
    runs = get_reference_runs(config, num_seeds, include_flag_100)
    if not runs:
        print(
            f"⚠️ No clean references (clean_oracle_model.csv / clean_flagged_model.csv) "
            f"found in {config.data_root / 'stage4'}; rerun fakeun-dataprep"
        )
        return {}
    return train_runs(config, runs, device, skip_existing)


def train_runs(
    config, runs: List[dict], device: Optional[torch.device] = None, skip_existing: bool = False
) -> Dict[str, str]:
    """Train each run (config overrides + run_name) one after the other."""
    set_seed(config.seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_names = [run["run_name"] for run in runs]
    print("\n" + "=" * 70)
    print(f"TRAINING ALL MODELS: {', '.join(run_names)}")
    print("=" * 70)

    trained_models = {}
    skipped_models = []
    failed_models = []

    for i, run in enumerate(runs, 1):
        model_config = copy.deepcopy(config)
        for key, value in run.items():
            if key != "run_name":
                setattr(model_config, key, value)
        run_name = run["run_name"]

        print(f"\n{'='*70}")
        print(f"TRAINING [{i}/{len(runs)}]: {run_name}")
        print(f"{'='*70}")

        if skip_existing:
            existing = skip_or_prepare_run(model_config, run_name)
            if existing:
                trained_models[run_name] = existing
                skipped_models.append(run_name)
                continue

        try:
            model_path = train_model(model_config, device, run_name=run_name)

            if model_path:
                trained_models[run_name] = model_path
                print(f"✅ Successfully trained {run_name}")
            else:
                print(f"⚠️ Skipped {run_name} (missing data files)")
                failed_models.append(run_name)

        except Exception as e:
            print(f"❌ Failed to train {run_name}: {e}")
            import traceback

            traceback.print_exc()
            failed_models.append(run_name)
            continue

    # Summary
    print("\n" + "=" * 70)
    print("ALL MODELS TRAINING COMPLETE")
    print("=" * 70)
    print(f"Successfully trained: {len(trained_models)}/{len(runs)} models\n")
    if skipped_models:
        print(f"Already trained (skipped): {skipped_models}\n")

    for run_name, path in trained_models.items():
        print(f"  ✅ {run_name}: {path}")

    if failed_models:
        print(f"\n⚠️  Failed/Skipped models: {failed_models}")

    return trained_models


def resume_training(
    checkpoint_path: str,
    config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    additional_epochs: int = 5,
) -> str:
    """
    Resume training from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        train_loader: Training dataloader
        val_loader: Validation dataloader
        additional_epochs: Number of additional epochs to train

    Returns:
        Path to the checkpoint from the last epoch
    """
    print(f"\nResuming training from: {checkpoint_path}")

    # Select appropriate Lightning module based on architecture
    lit_model = get_model_class(config.architecture).load_from_checkpoint(
        checkpoint_path, config=config
    )

    # Get paths
    paths = config.get_paths()

    # Setup logging
    logger = TensorBoardLogger(save_dir=paths["logs_dir"], name=f"logs_{config.run_name}_resumed")

    # Setup callbacks
    last_epoch_checkpoint = LastEpochCheckpoint(
        paths["checkpoints_dir"], f"{config.run_name}-resumed"
    )

    early_stopping = EarlyStopping(
        monitor="val_auc", patience=config.early_stopping_patience, verbose=True, mode="max"
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=additional_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[last_epoch_checkpoint, early_stopping],
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Resume training
    trainer.fit(lit_model, train_loader, val_loader)

    print(f"✅ Resumed training complete. Last-epoch model: {last_epoch_checkpoint.checkpoint_path}")
    return str(last_epoch_checkpoint.checkpoint_path)
