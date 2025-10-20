"""Path management utilities."""

from pathlib import Path
from typing import Dict


def get_experiment_paths(
    base_dir: Path, experiment_name: str, model_folder: str
) -> Dict[str, Path]:
    """Generate all necessary paths for an experiment."""
    experiment_dir = base_dir / experiment_name
    model_dir = experiment_dir / model_folder

    paths = {
        "experiment_dir": experiment_dir,
        "model_dir": model_dir,
        "checkpoints_dir": model_dir / "checkpoints",
        "logs_dir": model_dir / "logs",
        "results_dir": model_dir / "results",
    }

    # Create directories
    for key in ["checkpoints_dir", "logs_dir", "results_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths
