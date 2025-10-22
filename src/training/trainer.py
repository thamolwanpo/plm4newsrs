import torch
from pathlib import Path
from typing import Dict, Optional
import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from src.data import NewsDataset
from src.data.collate import collate_fn
from src.models.simple import LitRecommender
from src.models.nrms import LitNRMSRecommender
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

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = NewsDataset(paths["train_csv"], config, tokenizer)
    val_dataset = NewsDataset(paths["val_csv"], config, tokenizer)

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
    if config.architecture == "nrms":
        lit_model = LitNRMSRecommender(config)
    elif config.architecture == "simple":
        lit_model = LitRecommender(config)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")

    # Setup logging
    logger = TensorBoardLogger(save_dir=paths["logs_dir"], name=f"logs_{config.model_type}")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        dirpath=paths["checkpoints_dir"],
        filename=f"{config.model_type}-{{epoch:02d}}-{{val_auc:.4f}}",
        save_top_k=1,
        mode="max",
        verbose=True,
        save_last=True,  # Also save last checkpoint
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
        gradient_clip_val=1.0,  # Gradient clipping for stability
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


def train_all_models(config, device: Optional[torch.device] = None) -> Dict[str, str]:
    """
    Train all model types (clean and poisoned).

    Args:
        config: Base configuration (will be modified for each model type)
        device: Device to train on (auto-detected if None)

    Returns:
        Dictionary mapping model_type to checkpoint path
    """
    # Set seed once at the start
    set_seed(config.seed)

    print("\n" + "=" * 70)
    print("TRAINING ALL MODEL TYPES")
    print("=" * 70)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define model types to train
    model_types = ["clean", "poisoned"]

    trained_models = {}
    failed_models = []

    for i, model_type in enumerate(model_types, 1):
        print(f"\n{'='*70}")
        print(f"TRAINING [{i}/{len(model_types)}]: {model_type.upper()}")
        print(f"{'='*70}")

        try:
            # Create a copy of config for this model type
            import copy

            model_config = copy.deepcopy(config)
            model_config.model_type = model_type

            # Update paths to reflect new model type
            model_config.__post_init__()

            # Train
            model_path = train_model(model_config, device)

            if model_path:
                trained_models[model_type] = model_path
                print(f"✅ Successfully trained {model_type}")
            else:
                print(f"⚠️ Skipped {model_type} (missing data files)")
                failed_models.append(model_type)

        except Exception as e:
            print(f"❌ Failed to train {model_type}: {e}")
            import traceback

            traceback.print_exc()
            failed_models.append(model_type)
            continue

    # Summary
    print("\n" + "=" * 70)
    print("ALL MODELS TRAINING COMPLETE")
    print("=" * 70)
    print(f"Successfully trained: {len(trained_models)}/{len(model_types)} models\n")

    for model_type, path in trained_models.items():
        print(f"  ✅ {model_type}: {path}")

    if failed_models:
        print(f"\n⚠️  Failed/Skipped models: {failed_models}")
        print(f"\nNote: Make sure the following files exist in your data directory:")
        for model_type in failed_models:
            print(f"  - train_{model_type}.csv")
            print(f"  - val_{model_type}.csv")

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
    if config.architecture == "nrms":
        lit_model = LitNRMSRecommender.load_from_checkpoint(checkpoint_path, config=config)
    elif config.architecture == "simple":
        lit_model = LitRecommender.load_from_checkpoint(checkpoint_path, config=config)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")

    # Get paths
    paths = config.get_paths()

    # Setup logging
    logger = TensorBoardLogger(save_dir=paths["logs_dir"], name=f"logs_{config.model_type}_resumed")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        dirpath=paths["checkpoints_dir"],
        filename=f"{config.model_type}-resumed-{{epoch:02d}}-{{val_auc:.4f}}",
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
