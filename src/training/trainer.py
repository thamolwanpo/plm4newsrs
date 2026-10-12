import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from src.data import NewsDataset
from src.data.collate import collate_fn
from src.data.preprocessing import build_user_map, read_model_csv
from src.models.registry import get_model, get_model_class
from src.utils.seed import set_seed


def train_model(config, device: Optional[torch.device] = None, return_trainer: bool = False) -> str:
    """
    Train a single model.

    Args:
        config: Model configuration (ModelConfig instance)
        device: Device to train on (auto-detected if None)
        return_trainer: If True, return trainer instead of checkpoint path

    Returns:
        Path to best model checkpoint (or trainer if return_trainer=True)
    """
    # Set seed for reproducibility
    set_seed(config.seed)
    _ = config.run_name  # validates model_type / poison_set

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
        map_path = user_map_path(paths["checkpoints_dir"], config.run_name)
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
    logger = TensorBoardLogger(save_dir=paths["logs_dir"], name=f"logs_{config.run_name}")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        dirpath=paths["checkpoints_dir"],
        filename=f"{config.run_name}-{{epoch:02d}}-{{val_auc:.4f}}",
        save_top_k=1,
        mode="max",
        verbose=True,
        save_last=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_auc", patience=config.early_stopping_patience, verbose=True, mode="max"
    )

    # Auto-detect device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
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
    print(f"✅ Best model saved: {checkpoint_callback.best_model_path}")
    print(f"✅ Best val_auc: {checkpoint_callback.best_model_score:.4f}")

    if return_trainer:
        return trainer

    return checkpoint_callback.best_model_path


def user_map_path(checkpoints_dir: Path, run_name: str) -> Path:
    """Where the LSTUR user map of a trained model is stored."""
    return Path(checkpoints_dir) / f"{run_name}_user_map.json"


def get_training_runs(config) -> List[tuple]:
    """(model_type, poison_set) pairs trained by --train-all: the clean model
    plus one poisoned model per A set (config.poison_sets, or all found)."""
    poison_sets = config.poison_sets or config.discover_poison_sets()
    return [("clean", None)] + [("poisoned", a) for a in poison_sets]


def train_all_models(config, device: Optional[torch.device] = None) -> Dict[str, str]:
    """
    Train the clean model and every poisoned model.

    Args:
        config: Base configuration (copied for each run)
        device: Device to train on (auto-detected if None)

    Returns:
        Dictionary mapping run name to checkpoint path
    """
    set_seed(config.seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = get_training_runs(config)
    if len(runs) == 1:
        print(f"⚠️ No poisoned sets found in {config.data_root / 'stage4'}")

    print("\n" + "=" * 70)
    print(f"TRAINING ALL MODELS: {', '.join(p or m for m, p in runs)}")
    print("=" * 70)

    trained_models = {}
    failed_models = []

    for i, (model_type, poison_set) in enumerate(runs, 1):
        model_config = copy.deepcopy(config)
        model_config.model_type = model_type
        model_config.poison_set = poison_set
        run_name = model_config.run_name

        print(f"\n{'='*70}")
        print(f"TRAINING [{i}/{len(runs)}]: {run_name}")
        print(f"{'='*70}")

        try:
            model_path = train_model(model_config, device)

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
        Path to new best checkpoint
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
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        dirpath=paths["checkpoints_dir"],
        filename=f"{config.run_name}-resumed-{{epoch:02d}}-{{val_auc:.4f}}",
        save_top_k=1,
        mode="max",
        verbose=True,
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
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Resume training
    trainer.fit(lit_model, train_loader, val_loader)

    print(f"✅ Resumed training complete. Best model: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path
