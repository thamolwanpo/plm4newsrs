"""Abstract base class for all analyzers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import json


class BaseAnalyzer(ABC):
    """
    Abstract base class for evaluation analyzers.

    All analyzers must implement:
    - analyze(): Run analysis and return results
    - save_results(): Save results to file
    - get_summary(): Get human-readable summary

    Optional to override:
    - validate_data(): Check if data is suitable for analysis
    - cleanup(): Clean up resources after analysis
    """

    def __init__(self, results_dir: Path, name: Optional[str] = None):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory to save analysis results
            name: Name of analyzer (defaults to class name)
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.name = name or self.__class__.__name__.replace("Analyzer", "").lower()
        self.analysis_results = None

    @abstractmethod
    def analyze(self, results_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Run analysis on evaluation results.

        Args:
            results_df: DataFrame with evaluation results
                Required columns: ['user_id', 'news_id', 'score', 'label']
                Optional columns: ['is_fake', 'model_type', etc.]
            **kwargs: Additional analysis parameters

        Returns:
            Dictionary containing analysis results

        Example:
            >>> analyzer = MyAnalyzer(results_dir)
            >>> results = analyzer.analyze(results_df)
            >>> print(results.keys())
        """
        pass

    @abstractmethod
    def save_results(self, results: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save analysis results to file.

        Args:
            results: Analysis results (uses self.analysis_results if None)

        Returns:
            Path to saved results file

        Example:
            >>> analyzer.analyze(results_df)
            >>> csv_path = analyzer.save_results()
            >>> print(f"Results saved to: {csv_path}")
        """
        pass

    @abstractmethod
    def get_summary(self, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Get human-readable summary of analysis results.

        Args:
            results: Analysis results (uses self.analysis_results if None)

        Returns:
            Formatted summary string

        Example:
            >>> analyzer.analyze(results_df)
            >>> print(analyzer.get_summary())
        """
        pass

    def validate_data(self, results_df: pd.DataFrame) -> bool:
        """
        Validate that data has required columns for this analysis.

        Args:
            results_df: DataFrame to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If data is invalid with detailed error message
        """
        required_columns = self.get_required_columns()
        missing_columns = set(required_columns) - set(results_df.columns)

        if missing_columns:
            raise ValueError(
                f"{self.name} analyzer requires columns: {required_columns}\n"
                f"Missing: {missing_columns}"
            )

        if len(results_df) == 0:
            raise ValueError(f"Empty DataFrame provided to {self.name} analyzer")

        return True

    def get_required_columns(self) -> List[str]:
        """
        Get list of required columns for this analyzer.

        Override this method to specify required columns.

        Returns:
            List of required column names
        """
        return ["user_id", "news_id", "score", "label"]

    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """
        Save analysis metadata (parameters, timestamps, etc.).

        Args:
            metadata: Dictionary of metadata to save

        Returns:
            Path to metadata file
        """
        metadata_path = self.results_dir / f"{self.name}_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return metadata_path

    def load_results(self, results_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load previously saved analysis results.

        Args:
            results_path: Path to results file (auto-detects if None)

        Returns:
            Dictionary of loaded results
        """
        if results_path is None:
            # Try to find the most recent results file
            results_path = self.results_dir / f"{self.name}_results.csv"

        if not results_path.exists():
            raise FileNotFoundError(f"No results found at: {results_path}")

        # Load based on file extension
        if results_path.suffix == ".csv":
            return {"dataframe": pd.read_csv(results_path)}
        elif results_path.suffix == ".json":
            with open(results_path, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {results_path.suffix}")

    def cleanup(self):
        """
        Clean up resources after analysis.

        Override this method if analyzer needs cleanup
        (e.g., closing connections, deleting temp files).
        """
        pass

    def run(self, results_df: pd.DataFrame, save: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Convenience method: validate, analyze, and optionally save.

        Args:
            results_df: DataFrame with evaluation results
            save: Whether to save results to file
            **kwargs: Additional parameters for analyze()

        Returns:
            Analysis results dictionary

        Example:
            >>> analyzer = ExposureAnalyzer(results_dir)
            >>> results = analyzer.run(results_df, k=10)
            >>> print(results['avg_fake_count'])
        """
        # Validate
        self.validate_data(results_df)

        # Analyze
        print(f"\nRunning {self.name} analysis...")
        results = self.analyze(results_df, **kwargs)
        self.analysis_results = results

        # Save if requested
        if save:
            results_path = self.save_results(results)
            print(f"✅ Results saved: {results_path}")

        # Print summary
        summary = self.get_summary(results)
        print(f"\n{summary}")

        return results

    def __repr__(self) -> str:
        """String representation of analyzer."""
        return f"{self.__class__.__name__}(name='{self.name}', results_dir='{self.results_dir}')"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.name.title()} Analyzer"
