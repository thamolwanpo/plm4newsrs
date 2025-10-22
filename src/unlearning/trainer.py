# src/unlearning/trainer.py

"""
Main orchestrator for machine unlearning workflow.

Handles the complete unlearning pipeline:
1. Load model from checkpoint
2. Load forget/retain data
3. Apply unlearning method
4. Evaluate results
5. Save unlearned model and metrics
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import copy

from configs import BaseConfig
from configs.unlearning import BaseUnlearningConfig
from src.models.simple import LitRecommender
from src.models.nrms import LitNRMSRecommender
from src.data import NewsDataset, collate_fn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.unlearning import (
    ForgetSet,
    get_unlearning_method,
    list_unlearning_methods,
    UnlearningEvaluator,
    UnlearningTimer,
)
from src.utils.seed import set_seed


def unlearn_model(
    model_checkpoint: Path,
    model_config: BaseConfig,
    unlearn_config: BaseUnlearningConfig,
    device: Optional[torch.device] = None,
    evaluate: bool = True,
    save_unlearned: bool = True,
) -> Dict[str, Any]:
    """
    Main function to unlearn from a trained model.

    Complete workflow:
    1. Load model from checkpoint (using model_config)
    2. Load forget/retain sets (using unlearn_config)
    3. Get unlearning method
    4. Execute unlearning
    5. Evaluate results
    6. Save unlearned model and metrics

    Args:
        model_checkpoint: Path to trained model checkpoint
        model_config: Model configuration (architecture, dataset, etc.)
        unlearn_config: Unlearning configuration (method, hyperparameters, data)
        device: Device to run on (auto-detect if None)
        evaluate: Whether to run evaluation
        save_unlearned: Whether to save unlearned model

    Returns:
        Dictionary with:
        - unlearned_model: The unlearned model
        - unlearned_checkpoint_path: Path to saved checkpoint
        - evaluation_results: Evaluation metrics (if evaluate=True)
        - unlearning_time: Time taken for unlearning

    Example:
        >>> from configs import load_config
        >>> from configs.unlearning import FirstOrderConfig
        >>>
        >>> model_config = load_config("configs/experiments/simple/bert_finetune.yaml")
        >>> unlearn_config = FirstOrderConfig(
        ...     method="first_order",
        ...     learning_rate=0.0005,
        ...     num_steps=3,
        ...     mode="manual",
        ...     forget_set_path="data/forget.csv",
        ...     retain_set_path="data/retain.csv"
        ... )
        >>>
        >>> results = unlearn_model(
        ...     model_checkpoint=Path("checkpoints/poisoned.ckpt"),
        ...     model_config=model_config,
        ...     unlearn_config=unlearn_config,
        ...     device=torch.device("cuda")
        ... )
        >>>
        >>> print(f"Unlearned model saved: {results['unlearned_checkpoint_path']}")
        >>> print(f"Overall score: {results['evaluation_results']['overall']['overall_score']:.4f}")
    """
    print(f"\n{'='*70}")
    print("MACHINE UNLEARNING PIPELINE")
    print(f"{'='*70}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set seed
    set_seed(model_config.seed)
    print(f"Random seed: {model_config.seed}\n")

    # Print configs
    print(f"{'─'*70}")
    print("MODEL CONFIGURATION")
    print(f"{'─'*70}")
    print(f"Architecture: {model_config.architecture}")
    print(f"Model name: {model_config.model_name}")
    print(f"Dataset: {model_config.dataset}")
    print(f"Checkpoint: {model_checkpoint}")

    print(f"\n{'─'*70}")
    print("UNLEARNING CONFIGURATION")
    print(f"{'─'*70}")
    unlearn_config.print_config()

    # Step 1: Load model from checkpoint
    print(f"{'='*70}")
    print("STEP 1/6: LOAD MODEL FROM CHECKPOINT")
    print(f"{'='*70}")
    model_before, lit_model = _load_model(model_checkpoint, model_config, device)
    print(f"✅ Model loaded successfully\n")

    # Step 2: Load forget/retain sets
    print(f"{'='*70}")
    print("STEP 2/6: LOAD FORGET/RETAIN SETS")
    print(f"{'='*70}")
    forget_set, forget_loader, retain_loader, corrected_loader = _load_data(
        unlearn_config, model_config, device
    )
    forget_set.print_summary()
    print(f"✅ Data loaded successfully\n")

    # Step 3: Get unlearning method
    print(f"{'='*70}")
    print("STEP 3/6: INITIALIZE UNLEARNING METHOD")
    print(f"{'='*70}")
    unlearning_method = get_unlearning_method(
        name=unlearn_config.method,
        model=lit_model.model,  # Get the actual model (not Lightning wrapper)
        model_config=model_config,
        unlearn_config=unlearn_config,
        device=device,
    )
    print(f"✅ Method initialized: {unlearn_config.method}\n")

    # Step 4: Execute unlearning
    print(f"{'='*70}")
    print("STEP 4/6: EXECUTE UNLEARNING")
    print(f"{'='*70}")
    with UnlearningTimer() as timer:
        if corrected_loader is not None:
            print("Using label correction mode (paper's formula)\n")
            unlearned_model = unlearning_method.unlearn(
                forget_loader=forget_loader,
                retain_loader=retain_loader,
                validation_loader=retain_loader,
                corrected_loader=corrected_loader,
            )
        else:
            print("Using data removal mode (heuristic)\n")
            unlearned_model = unlearning_method.unlearn(
                forget_loader=forget_loader,
                retain_loader=retain_loader,
                validation_loader=retain_loader,
            )

    unlearning_time = timer.get_elapsed()
    print(f"✅ Unlearning complete in {unlearning_time:.2f}s ({unlearning_time/60:.2f}m)\n")

    # Step 5: Evaluate
    evaluation_results = None
    if evaluate:
        print(f"{'='*70}")
        print("STEP 5/6: EVALUATE UNLEARNING")
        print(f"{'='*70}")

        evaluator = UnlearningEvaluator(model_config, device)

        # Wrap models back for evaluation
        model_before_eval = model_before.model
        model_after_eval = unlearned_model

        evaluation_results = evaluator.evaluate_full(
            model_before=model_before_eval,
            model_after=model_after_eval,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            test_loader=None,  # Can add benchmark evaluation
            unlearning_time=unlearning_time,
        )

        # Save evaluation results
        evaluator.save_results(evaluation_results)
        evaluator.save_summary_csv(evaluation_results)
        evaluator.print_summary(evaluation_results)

        print(f"✅ Evaluation complete\n")

    # Step 6: Save unlearned model
    unlearned_checkpoint_path = None
    if save_unlearned:
        print(f"{'='*70}")
        print("STEP 6/6: SAVE UNLEARNED MODEL")
        print(f"{'='*70}")
        unlearned_checkpoint_path = _save_unlearned_model(
            unlearned_model=unlearned_model,
            model_config=model_config,
            unlearn_config=unlearn_config,
            evaluation_results=evaluation_results,
            unlearning_time=unlearning_time,
        )
        print(f"✅ Model saved: {unlearned_checkpoint_path}\n")

    # Summary
    print(f"{'='*70}")
    print("UNLEARNING PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Method: {unlearn_config.method}")
    print(f"Time: {unlearning_time:.2f}s")
    if evaluation_results:
        overall = evaluation_results["overall"]
        print(f"Overall score: {overall['overall_score']:.4f} ({overall['grade']})")
    if unlearned_checkpoint_path:
        print(f"Saved to: {unlearned_checkpoint_path}")
    print(f"{'='*70}\n")

    return {
        "unlearned_model": unlearned_model,
        "unlearned_checkpoint_path": unlearned_checkpoint_path,
        "evaluation_results": evaluation_results,
        "unlearning_time": unlearning_time,
        "model_config": model_config,
        "unlearn_config": unlearn_config,
    }


def _load_model(checkpoint_path: Path, config: BaseConfig, device: torch.device):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        device: Device to load on

    Returns:
        Tuple of (model_before_copy, model_for_unlearning)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load model
    if config.architecture == "nrms":
        lit_model = LitNRMSRecommender.load_from_checkpoint(
            str(checkpoint_path), config=config, map_location=device
        )
    elif config.architecture == "simple":
        lit_model = LitRecommender.load_from_checkpoint(
            str(checkpoint_path), config=config, map_location=device
        )
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")
    lit_model.to(device)
    lit_model.eval()

    print(f"Model architecture: {config.architecture}")
    print(f"Model name: {config.model_name}")

    # Create a copy for "before" comparison
    model_before = copy.deepcopy(lit_model)

    return model_before, lit_model


def _load_data(
    unlearn_config: BaseUnlearningConfig, model_config: BaseConfig, device: torch.device
) -> Tuple:
    if unlearn_config.mode == "manual":
        print(f"Loading data in MANUAL mode")

        corrected_path = getattr(unlearn_config, "corrected_set_path", None)

        forget_set = ForgetSet.from_manual(
            forget_path=unlearn_config.forget_set_path,
            retain_path=unlearn_config.retain_set_path,
            corrected_path=corrected_path,
        )

    elif unlearn_config.mode == "ratio":
        print(f"Loading data in RATIO mode")
        forget_set = ForgetSet.from_ratio(
            splits_dir=unlearn_config.unlearning_splits_dir, trial_idx=unlearn_config.trial_idx
        )

    else:
        raise ValueError(f"Unknown mode: {unlearn_config.mode}")

    if forget_set.is_label_correction():
        print(f"\n🔄 AUTO-DETECTED: Label Correction Mode")
        forget_loader, corrected_loader, retain_loader = _create_dataloaders_label_correction(
            forget_set, model_config
        )
        return forget_set, forget_loader, retain_loader, corrected_loader
    else:
        print(f"\n🔄 AUTO-DETECTED: Data Removal Mode")
        forget_loader, retain_loader = _create_dataloaders(forget_set, model_config)
        return forget_set, forget_loader, retain_loader, None


def _create_dataloaders_label_correction(
    forget_set: ForgetSet, model_config: BaseConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    import pandas as pd

    temp_dir = Path("./temp_unlearning_data")
    temp_dir.mkdir(exist_ok=True)

    forget_temp_path = temp_dir / "forget_temp.csv"
    corrected_temp_path = temp_dir / "corrected_temp.csv"
    retain_temp_path = temp_dir / "retain_temp.csv"

    forget_set.forget_df.to_csv(forget_temp_path, index=False)
    forget_set.corrected_df.to_csv(corrected_temp_path, index=False)
    forget_set.retain_df.to_csv(retain_temp_path, index=False)

    use_glove = "glove" in model_config.model_name.lower()
    tokenizer = None if use_glove else AutoTokenizer.from_pretrained(model_config.model_name)

    forget_dataset = NewsDataset(forget_temp_path, model_config, tokenizer)
    corrected_dataset = NewsDataset(corrected_temp_path, model_config, tokenizer)
    retain_dataset = NewsDataset(retain_temp_path, model_config, tokenizer)

    print(len(forget_dataset), len(corrected_dataset), len(retain_dataset))

    forget_loader = DataLoader(
        forget_dataset,
        batch_size=model_config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    corrected_loader = DataLoader(
        corrected_dataset,
        batch_size=model_config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    retain_loader = DataLoader(
        retain_dataset,
        batch_size=model_config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Created loaders:")
    print(f"  Z (wrong labels): {len(forget_loader)} batches")
    print(f"  Z̃ (correct labels): {len(corrected_loader)} batches")
    print(f"  Retain: {len(retain_loader)} batches")

    return forget_loader, corrected_loader, retain_loader


def _create_dataloaders(
    forget_set: ForgetSet, model_config: BaseConfig
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for data removal mode (TWO loaders)."""
    import pandas as pd

    temp_dir = Path("./temp_unlearning_data")
    temp_dir.mkdir(exist_ok=True)

    forget_temp_path = temp_dir / "forget_temp.csv"
    retain_temp_path = temp_dir / "retain_temp.csv"

    forget_set.forget_df.to_csv(forget_temp_path, index=False)
    forget_set.retain_df.to_csv(retain_temp_path, index=False)

    use_glove = "glove" in model_config.model_name.lower()
    tokenizer = None if use_glove else AutoTokenizer.from_pretrained(model_config.model_name)

    forget_dataset = NewsDataset(forget_temp_path, model_config, tokenizer)
    retain_dataset = NewsDataset(retain_temp_path, model_config, tokenizer)

    forget_loader = DataLoader(
        forget_dataset,
        batch_size=model_config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    retain_loader = DataLoader(
        retain_dataset,
        batch_size=model_config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Forget loader: {len(forget_loader)} batches")
    print(f"Retain loader: {len(retain_loader)} batches")

    return forget_loader, retain_loader


def _save_unlearned_model(
    unlearned_model: nn.Module,
    model_config: BaseConfig,
    unlearn_config: BaseUnlearningConfig,
    evaluation_results: Optional[Dict[str, Any]],
    unlearning_time: float,
) -> Path:
    """
    Save unlearned model as proper Lightning checkpoint (matching training checkpoint format).

    Args:
        unlearned_model: The unlearned model (raw nn.Module from unlearning method)
        model_config: Model configuration
        unlearn_config: Unlearning configuration
        evaluation_results: Evaluation results
        unlearning_time: Time taken for unlearning

    Returns:
        Path to saved checkpoint
    """
    paths = model_config.get_paths()
    checkpoints_dir = paths["checkpoints_dir"]

    # Create filename with evaluation metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if evaluation_results:
        forget_flip = evaluation_results["forget_quality"]["after"]["positive_flip_rate"]
        retain_auc = evaluation_results["utility"]["retain_after"].get(
            "auc", evaluation_results["utility"]["retain_after"]["accuracy"]
        )
        filename = f"unlearned-{unlearn_config.method}-flip={forget_flip:.4f}-retain_auc={retain_auc:.4f}-{timestamp}.ckpt"
    else:
        filename = f"unlearned-{unlearn_config.method}-{timestamp}.ckpt"

    checkpoint_path = checkpoints_dir / filename

    # Wrap raw model back into LitRecommender to get proper state_dict format
    from src.models.simple import LitRecommender
    from src.models.nrms import LitNRMSRecommender

    if model_config.architecture == "nrms":
        lit_model = LitNRMSRecommender(model_config)
    elif model_config.architecture == "simple":
        lit_model = LitRecommender(model_config)
    else:
        raise ValueError(f"Unknown architecture: {model_config.architecture}")
    lit_model.model.load_state_dict(unlearned_model.state_dict())

    # Create checkpoint matching PyTorch Lightning format
    import pytorch_lightning as pl

    checkpoint = {
        # Standard Lightning keys
        "state_dict": lit_model.state_dict(),
        "epoch": 0,  # Unlearning happens post-training
        "global_step": 0,
        "pytorch-lightning_version": pl.__version__,
        "hyper_parameters": {
            # Model config as hyper_parameters (Lightning expects this)
            **model_config.__dict__,
        },
        # Custom unlearning metadata (won't interfere with Lightning loading)
        "unlearning_metadata": {
            "method": unlearn_config.method,
            "unlearn_config": {
                k: str(v) if isinstance(v, Path) else v for k, v in unlearn_config.__dict__.items()
            },
            "unlearning_time": unlearning_time,
            "timestamp": timestamp,
        },
    }

    # Add evaluation results if available
    if evaluation_results:
        checkpoint["unlearning_metadata"]["evaluation_summary"] = {
            "forget_efficacy": evaluation_results["forget_quality"]["efficacy"],
            "forget_flip_rate": evaluation_results["forget_quality"]["after"]["positive_flip_rate"],
            "forget_auc": evaluation_results["forget_quality"]["after"]["auc"],
            "retain_auc": evaluation_results["utility"]["retain_after"].get(
                "auc", evaluation_results["utility"]["retain_after"]["accuracy"]
            ),
            "overall_score": evaluation_results["overall"]["overall_score"],
            "grade": evaluation_results["overall"]["grade"],
        }

        # Also save at checkpoint level for easy access
        checkpoint["val_auc"] = evaluation_results["utility"]["retain_after"].get(
            "auc", evaluation_results["utility"]["retain_after"]["accuracy"]
        )
        checkpoint["forget_flip_rate"] = evaluation_results["forget_quality"]["after"][
            "positive_flip_rate"
        ]

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    print(f"✅ Checkpoint saved (Lightning-compatible): {checkpoint_path}")
    print(f"  Method: {unlearn_config.method}")
    print(f"  Time: {unlearning_time:.2f}s")
    if evaluation_results:
        eval_summary = checkpoint["unlearning_metadata"]["evaluation_summary"]
        print(f"  Overall score: {eval_summary['overall_score']:.4f} ({eval_summary['grade']})")
        print(f"  Forget flip rate: {eval_summary['forget_flip_rate']:.4f}")
        print(f"  Forget AUC: {eval_summary['forget_auc']:.4f}")
        print(f"  Retain AUC: {eval_summary['retain_auc']:.4f}")

    return checkpoint_path


def unlearn_multiple_trials(
    model_checkpoint: Path,
    model_config: BaseConfig,
    unlearn_config: BaseUnlearningConfig,
    num_trials: int = 3,
    device: Optional[torch.device] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Run unlearning for multiple trials (useful for ratio-based splits).

    Args:
        model_checkpoint: Path to trained model checkpoint
        model_config: Model configuration
        unlearn_config: Base unlearning configuration
        num_trials: Number of trials to run
        device: Device to run on

    Returns:
        Dictionary mapping trial_idx -> results

    Example:
        >>> results = unlearn_multiple_trials(
        ...     model_checkpoint=Path("checkpoints/poisoned.ckpt"),
        ...     model_config=model_config,
        ...     unlearn_config=unlearn_config,
        ...     num_trials=3
        ... )
        >>> for trial, result in results.items():
        ...     print(f"Trial {trial}: {result['evaluation_results']['overall']['overall_score']:.4f}")
    """
    if unlearn_config.mode != "ratio":
        raise ValueError("Multiple trials only supported for ratio mode")

    print(f"\n{'='*70}")
    print(f"RUNNING {num_trials} UNLEARNING TRIALS")
    print(f"{'='*70}\n")

    all_results = {}

    for trial_idx in range(num_trials):
        print(f"\n{'#'*70}")
        print(f"TRIAL {trial_idx + 1}/{num_trials}")
        print(f"{'#'*70}\n")

        # Update trial index in config
        trial_config = copy.deepcopy(unlearn_config)
        trial_config.trial_idx = trial_idx

        # Run unlearning
        results = unlearn_model(
            model_checkpoint=model_checkpoint,
            model_config=model_config,
            unlearn_config=trial_config,
            device=device,
            evaluate=True,
            save_unlearned=True,
        )

        all_results[trial_idx] = results

    # Summary across trials
    print(f"\n{'='*70}")
    print(f"SUMMARY ACROSS {num_trials} TRIALS")
    print(f"{'='*70}")

    scores = [r["evaluation_results"]["overall"]["overall_score"] for r in all_results.values()]
    forget_efficacies = [
        r["evaluation_results"]["forget_quality"]["efficacy"] for r in all_results.values()
    ]
    retain_accs = [
        r["evaluation_results"]["utility"]["retain_after"]["accuracy"] for r in all_results.values()
    ]

    print(f"Overall Score:")
    print(f"  Mean: {sum(scores)/len(scores):.4f}")
    print(
        f"  Std:  {(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5:.4f}"
    )
    print(f"  Min:  {min(scores):.4f}")
    print(f"  Max:  {max(scores):.4f}")

    print(f"\nForget Efficacy:")
    print(f"  Mean: {sum(forget_efficacies)/len(forget_efficacies):.4f}")

    print(f"\nRetain Accuracy:")
    print(f"  Mean: {sum(retain_accs)/len(retain_accs):.4f}")

    print(f"{'='*70}\n")

    return all_results


def list_available_methods():
    """Print available unlearning methods."""
    print(f"\n{'='*70}")
    print("AVAILABLE UNLEARNING METHODS")
    print(f"{'='*70}")

    methods = list_unlearning_methods()

    if not methods:
        print("No methods registered.")
        print("Available after importing src.unlearning.methods")
    else:
        for i, method in enumerate(methods, 1):
            print(f"{i}. {method}")

    print(f"{'='*70}\n")
