"""Main evaluator orchestrating the evaluation pipeline."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import ModelConfig
from src.data.preprocessing import load_news_items
from src.models.registry import get_model_class
from src.training.trainer import user_map_path
from src.utils.seed import set_seed

from .analyzers import ExposureAnalyzer, TruthDecayAnalyzer, FailureAnalyzer
from .benchmark_dataset import BenchmarkDataset, benchmark_collate_fn
from .metrics.misinformation import calculate_fake_exposure, calculate_mc_at_k
from .metrics.ranking import query_ranking_metrics
from .visualizer import EvaluationVisualizer

# stage6 probe benchmarks that cover the whole poisoning pool; for a given
# poisoned set they are also evaluated on that set's injected items only.
POISON_PROBES = ("poison_probe", "poison_probe_realneg")

# Order used in summaries (any other stage6 csv is appended alphabetically)
BENCHMARK_ORDER = [
    "standard",
    "standard_clean",
    "standard_mixed",
    "adversarial",
    "pattern_probe_realneg",
    "pattern_probe",
    "poison_probe_realneg",
    "poison_probe",
]

CKPT_PATTERN = re.compile(r"^(?P<run>.+?)-epoch=(?P<epoch>\d+)-val_auc=(?P<auc>[0-9.]+?)(-v\d+)?$")


def parse_run_name(run_name: str) -> Tuple[str, Optional[str]]:
    """Parse 'clean', 'clean_seed42', or 'poisoned_A_r0.5_s1' run names."""
    if run_name == "clean" or re.fullmatch(r"clean_seed\d+", run_name):
        return "clean", None
    if run_name.startswith("poisoned_"):
        return "poisoned", run_name[len("poisoned_") :]
    return run_name, None


class ModelEvaluator:
    """
    Evaluates trained models on the stage6 benchmarks.

    - One query = one target + its negatives; metrics are averaged over queries.
    - poison_probe*.csv cover the whole poisoning pool. They are also evaluated
      per poisoned set, restricted to that set's injected items
      (stage4/A_.../A_fake_items.csv), as "<benchmark>@<A set>": the clean
      model on every poisoned set (its reference), a poisoned model on its own.
    """

    def __init__(
        self,
        config: ModelConfig,
        device: Optional[torch.device] = None,
        enable_analyses: Optional[List[str]] = None,
        results_dir: Optional[Path] = None,
    ):
        """
        Args:
            config: Model configuration
            device: Device to run on (auto-detect if None)
            enable_analyses: Analyses to run: 'exposure', 'truth_decay', 'failure'
                (None = all)
            results_dir: Override results directory
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.paths = config.get_paths()
        self.results_dir = Path(results_dir) if results_dir else self.paths["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.analyzers = self._initialize_analyzers(enable_analyses)
        self.visualizer = EvaluationVisualizer(self.results_dir)

        # (model, benchmark) -> metrics dict / detailed results
        self.evaluation_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.detailed_results: Dict[Tuple[str, str], pd.DataFrame] = {}

        self._news_items = None
        self._benchmark_cache: Dict[str, BenchmarkDataset] = {}

        print(f"\n{'='*70}")
        print("MODEL EVALUATOR INITIALIZED")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Benchmarks: {self.paths['benchmarks_dir']}")
        print(f"Results directory: {self.results_dir}")
        print(f"Enabled analyses: {list(self.analyzers.keys())}")
        print(f"{'='*70}\n")

    def _initialize_analyzers(self, enable_analyses: Optional[List[str]]) -> Dict:
        available_analyzers = {
            "exposure": ExposureAnalyzer,
            "truth_decay": TruthDecayAnalyzer,
            "failure": FailureAnalyzer,
        }
        if enable_analyses is None:
            enable_analyses = list(available_analyzers.keys())

        analyzers = {}
        for name in enable_analyses:
            if name in available_analyzers:
                analyzers[name] = available_analyzers[name](self.results_dir)
            else:
                print(f"⚠️  Warning: Unknown analyzer '{name}' - skipping")
        return analyzers

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    def discover_models(self, model_names: Optional[List[str]] = None) -> Dict[str, Path]:
        """Last-epoch checkpoint per model, falling back to best legacy checkpoint.

        Args:
            model_names: run names or prefixes to keep, e.g. ["clean",
                "poisoned_A_r0.5_s1"] or ["poisoned"] (default: all)
        """
        selected: Dict[str, Tuple[int, float, Path]] = {}
        for path in sorted(self.paths["checkpoints_dir"].glob("*.ckpt")):
            is_last = path.stem.endswith("-last")
            if is_last:
                run = path.stem[:-5]
                if run.endswith("-resumed"):
                    run = run[: -len("-resumed")]
                rank = (1, path.stat().st_mtime)
            else:
                match = CKPT_PATTERN.match(path.stem)
                run, auc = (
                    (match["run"], float(match["auc"]))
                    if match
                    else (path.stem, float("-inf"))
                )
                rank = (0, auc)
            if run.endswith("-resumed"):
                run = run[: -len("-resumed")]
            if model_names and not any(run == m or run.startswith(m) for m in model_names):
                continue
            candidate = (*rank, path)
            if run not in selected or candidate[:2] > selected[run][:2]:
                selected[run] = candidate
        return {run: path for run, (_, _, path) in sorted(selected.items())}

    def discover_benchmarks(self, benchmark_names: Optional[List[str]] = None) -> List[Path]:
        """stage6 benchmark CSVs (exact names, default: all)."""
        files = {p.stem: p for p in self.paths["benchmarks_dir"].glob("*.csv")}
        if benchmark_names:
            missing = [b for b in benchmark_names if b not in files]
            if missing:
                print(f"⚠️  Benchmarks not found: {missing} (available: {sorted(files)})")
            names = [b for b in benchmark_names if b in files]
        else:
            names = [b for b in BENCHMARK_ORDER if b in files]
            names += sorted(set(files) - set(BENCHMARK_ORDER))
        return [files[n] for n in names]

    def _injected_items(self, poison_set: str) -> Set[str]:
        path = self.config.poison_set_dir(poison_set) / "A_fake_items.csv"
        return set(pd.read_csv(path, dtype={"item_id": str})["item_id"])

    def benchmark_plan(
        self, run_name: str, benchmark_files: List[Path]
    ) -> List[Tuple[str, Path, Optional[str]]]:
        """(benchmark name, csv, poisoned set to filter targets by) for one model."""
        model_type, own_set = parse_run_name(run_name)
        if model_type == "clean":
            filter_sets = self.config.poison_sets or self.config.discover_poison_sets()
        else:
            filter_sets = [own_set]

        plan = []
        for path in benchmark_files:
            plan.append((path.stem, path, None))
            if path.stem in POISON_PROBES:
                for a_set in filter_sets:
                    if (self.config.poison_set_dir(a_set) / "A_fake_items.csv").exists():
                        plan.append((f"{path.stem}@{a_set}", path, a_set))
        return plan

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate_all(
        self, model_names: Optional[List[str]] = None, benchmark_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Evaluate the last checkpoint of every trained model on every benchmark."""
        set_seed(self.config.seed)

        models = self.discover_models(model_names)
        benchmark_files = self.discover_benchmarks(benchmark_names)
        if not models or not benchmark_files:
            print(f"❌ No models ({self.paths['checkpoints_dir']}) or "
                  f"benchmarks ({self.paths['benchmarks_dir']}) found")
            return pd.DataFrame()

        print(f"Found {len(models)} models and {len(benchmark_files)} benchmarks.")
        for run_name, model_path in models.items():
            print(f"  - {run_name}: {model_path.name}")

        for run_name, model_path in models.items():
            try:
                self.evaluate_checkpoint(model_path, run_name, benchmark_files)
            except Exception as e:
                print(f"❌ Error evaluating model {run_name}: {e}")
                import traceback

                traceback.print_exc()

        return self.save_summary()

    def evaluate_checkpoint(
        self, model_path: Path, run_name: Optional[str], benchmark_files: List[Path]
    ):
        """Load one checkpoint and evaluate it on its benchmark plan."""
        model_path = Path(model_path)
        if run_name is None:
            if model_path.stem.endswith("-last"):
                run_name = model_path.stem[:-5]
                if run_name.endswith("-resumed"):
                    run_name = run_name[: -len("-resumed")]
            else:
                match = CKPT_PATTERN.match(model_path.stem)
                run_name = match["run"] if match else model_path.stem
        print(f"\n🔄 Loading model: {run_name} ({model_path.name})")

        tokenizer = None
        if "glove" not in self.config.model_name.lower():
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        model_class = get_model_class(self.config.architecture)
        model = model_class.load_from_checkpoint(str(model_path), weights_only=False)
        model.to(self.device)
        model.eval()

        user_map = None
        if self.config.architecture == "lstur":
            map_path = user_map_path(model_path.parent, run_name)
            with open(map_path) as f:
                user_map = json.load(f)

        for bench_name, bench_path, filter_set in self.benchmark_plan(run_name, benchmark_files):
            try:
                self.evaluate_benchmark(
                    model, tokenizer, run_name, bench_name, bench_path, filter_set, user_map
                )
            except Exception as e:
                print(f"❌ Error evaluating {run_name} on {bench_name}: {e}")
                import traceback

                traceback.print_exc()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_benchmark(self, bench_name, bench_path, filter_set, tokenizer, user_map):
        """Benchmark datasets don't depend on the model, so build each once."""
        if bench_name not in self._benchmark_cache:
            if self._news_items is None:
                self._news_items = load_news_items(self.config.news_items_path)
            target_filter = self._injected_items(filter_set) if filter_set else None
            self._benchmark_cache[bench_name] = BenchmarkDataset(
                bench_path,
                tokenizer,
                self.config,
                user_map=user_map,
                target_filter=target_filter,
                news_items=self._news_items,
            )
        dataset = self._benchmark_cache[bench_name]
        dataset.user_to_idx = user_map
        return dataset

    def evaluate_benchmark(
        self, model, tokenizer, run_name, bench_name, bench_path, filter_set=None, user_map=None
    ) -> Dict[str, Any]:
        print(f"   📍 Benchmark: {bench_name}")
        dataset = self._get_benchmark(bench_name, bench_path, filter_set, tokenizer, user_map)
        if len(dataset) == 0:
            print("      (no queries - skipped)")
            return {}

        loader = DataLoader(
            dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            collate_fn=benchmark_collate_fn,
            num_workers=0,
        )
        detailed = self._run_inference(model, loader)
        metrics = self._calculate_metrics(detailed)

        model_type, poison_set = parse_run_name(run_name)
        key = (run_name, bench_name)
        self.evaluation_results[key] = {
            "model": run_name,
            "model_type": model_type,
            "poison_set": poison_set,
            "benchmark": bench_name,
            **metrics,
        }
        self.detailed_results[key] = detailed

        safe_name = bench_name.replace("@", "__")
        detailed.to_parquet(
            self.results_dir / f"{run_name}__{safe_name}_detailed.parquet", index=False
        )
        print(
            f"      queries={metrics['n_queries']}  AUC={metrics['AUC']:.4f}  "
            f"MRR={metrics['MRR']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}"
        )
        return metrics

    def _run_inference(self, model, dataloader) -> pd.DataFrame:
        """Score every candidate; padded candidates are masked out."""
        columns = {k: [] for k in ("query_id", "user_id", "news_id", "score", "label", "is_fake")}
        use_amp = self.device.type == "cuda"

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Inference", leave=False):
                if batch is None:
                    continue

                batch_tensors = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    scores = model(batch_tensors)

                scores = scores.float()
                n_cand = batch_tensors["num_candidates"]
                pad_mask = torch.arange(scores.size(1), device=scores.device)[None, :] >= n_cand[
                    :, None
                ]
                scores = torch.softmax(scores.masked_fill(pad_mask, float("-inf")), dim=1)
                scores = scores.cpu().numpy()

                for i, impression in enumerate(batch["impression_data"]):
                    news_ids, _, labels, is_fakes = zip(*impression)
                    n_items = len(news_ids)
                    columns["query_id"].extend([batch["query_id"][i]] * n_items)
                    columns["user_id"].extend([batch["user_id"][i]] * n_items)
                    columns["news_id"].extend(news_ids)
                    columns["score"].extend(scores[i][:n_items])
                    columns["label"].extend(labels)
                    columns["is_fake"].extend(is_fakes)

        df = pd.DataFrame(columns)
        df["score"] = df["score"].astype(float)
        df["label"] = df["label"].astype(int)
        df["is_fake"] = df["is_fake"].astype(int)
        return df

    def _calculate_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Ranking metrics of the target + fake exposure, averaged over queries.

        For standard*, a higher target rank is better. For adversarial and the
        probes the target is fake, so a LOWER target rank (lower AUC / MRR /
        HR@k) means the model pushes that fake news less.
        """
        per_query = query_ranking_metrics(results_df, ks=(5, 10))
        metrics = {
            "n_queries": int(len(per_query)),
            "n_users": int(results_df["user_id"].nunique()),
            "avg_candidates": float(per_query["n_candidates"].mean()),
            "AUC": float(per_query["AUC"].mean()),
            "MRR": float(per_query["MRR"].mean()),
            "NDCG@5": float(per_query["NDCG@5"].mean()),
            "NDCG@10": float(per_query["NDCG@10"].mean()),
            "HR@5": float(per_query["HR@5"].mean()),
            "HR@10": float(per_query["HR@10"].mean()),
            "mean_rank": float(per_query["rank"].mean()),
            "mean_percentile": float(per_query["percentile"].mean()),
        }

        # Fake exposure among the top-k of each query (ignores the target's label)
        for k in (5, 10, 20):
            metrics[f"MC@{k}"] = calculate_mc_at_k(results_df, k=k, group_by="query_id")
        for k in (10, 20):
            exposure = calculate_fake_exposure(results_df, k=k, group_by="query_id")
            metrics[f"avg_fake_in_top_{k}"] = exposure["avg_fake_count"]
            metrics[f"queries_with_fake_pct_top_{k}"] = exposure["users_with_fake_pct"]
        return metrics

    def save_summary(self) -> pd.DataFrame:
        """Write evaluation_summary.csv (one row per model x benchmark)."""
        if not self.evaluation_results:
            return pd.DataFrame()
        summary_df = pd.DataFrame(list(self.evaluation_results.values()))
        summary_path = self.results_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ Saved summary: {summary_path}")
        return summary_df

    # ------------------------------------------------------------------ #
    # Analyses
    # ------------------------------------------------------------------ #
    def run_analyses(
        self,
        model_names: Optional[List[str]] = None,
        benchmark: str = "standard",
        top_k: int = 10,
        analyzer_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the enabled analyzers on one benchmark for every evaluated model.

        Args:
            model_names: run names or prefixes to include (default: all)
            benchmark: exact benchmark name (default: "standard")
            top_k: Top-k for exposure analysis
            analyzer_names: analyzers to run (default: all enabled)
        """
        print(f"\n{'='*70}")
        print(f"RUNNING ANALYSES on {benchmark}")
        print(f"{'='*70}\n")

        # Analyzers group impressions by "user_id"; give them one group per
        # query (users appear in several queries of the same benchmark).
        model_results = {
            model: df.drop(columns="user_id").rename(columns={"query_id": "user_id"})
            for (model, bench), df in self.detailed_results.items()
            if bench == benchmark
            and (not model_names or any(model.startswith(m) for m in model_names))
        }
        if not model_results:
            print(f"❌ No results found for benchmark: {benchmark}")
            return {}

        print(f"Analyzing {len(model_results)} models: {sorted(model_results)}")
        analysis_results = {}

        selected_analyzers = (
            self.analyzers.items()
            if analyzer_names is None
            else (
                (name, analyzer)
                for name, analyzer in self.analyzers.items()
                if name in analyzer_names
            )
        )
        for analyzer_name, analyzer in selected_analyzers:
            print(f"\n{'─'*70}")
            print(f"Running {analyzer_name} analysis...")
            print(f"{'─'*70}")

            try:
                analyzer_summary_data = []

                for model_name, results_df in model_results.items():
                    print(f"\n  Analyzing {model_name}...")
                    results = analyzer.run(results_df, save=False, top_k=top_k)
                    analysis_results[f"{analyzer_name}_{model_name}"] = results

                    model_summary = {"model": model_name, "benchmark": benchmark}
                    if analyzer_name == "exposure":
                        model_summary.update(
                            {
                                "avg_fake_in_top_k": results["exposure"]["avg_fake_count"],
                                "max_fake_in_top_k": results["exposure"]["max_fake_count"],
                                "queries_with_fake_pct": results["exposure"][
                                    "users_with_fake_pct"
                                ],
                                "avg_fake_ratio": results["exposure"]["avg_fake_ratio"],
                                "MC@10": results["mc_metrics"].get("MC@10", 0.0),
                            }
                        )
                    elif analyzer_name == "truth_decay":
                        median = results["median_ranks"].set_index("news_type")["median_rank"]
                        real_median = median.get("Real", float("nan"))
                        fake_median = median.get("Fake", float("nan"))
                        model_summary.update(
                            {
                                "real_median_rank": real_median,
                                "fake_median_rank": fake_median,
                                "rank_gap": real_median - fake_median,
                            }
                        )
                    elif analyzer_name == "failure":
                        model_summary.update(
                            {
                                "failure_rate_pct": results["overall"]["failure_rate_pct"],
                                "total_failures": results["overall"]["total_failures"],
                                "fake_involvement_pct": results["fake_involvement"][
                                    "fake_involvement_pct"
                                ],
                                "failures_with_fake": results["fake_involvement"][
                                    "fake_involved"
                                ],
                            }
                        )
                    analyzer_summary_data.append(model_summary)

                summary_path = self.results_dir / f"{analyzer_name}_{benchmark}_summary.csv"
                pd.DataFrame(analyzer_summary_data).to_csv(summary_path, index=False)
                print(f"\n✅ Saved {analyzer_name} summary: {summary_path.name}")

                if analyzer_name == "exposure":
                    comparison = analyzer.compare_models(model_results, top_k=top_k)
                    filename = (
                        "exposure_comparison.png"
                        if benchmark == "standard"
                        else f"exposure_comparison_{benchmark}.png"
                    )
                    self.visualizer.plot_exposure_comparison(
                        comparison, top_k=top_k, filename=filename
                    )
                elif analyzer_name == "truth_decay":
                    analyzer.compare_models(model_results, top_k=top_k)
                    rank_data = {}
                    for model_name, results_df in model_results.items():
                        df = results_df.copy()
                        df["rank"] = df.groupby("user_id")["score"].rank(
                            method="first", ascending=False
                        )
                        df["news_type"] = df["is_fake"].apply(lambda x: "Fake" if x else "Real")
                        rank_data[model_name] = df
                    self.visualizer.plot_rank_comparison(rank_data)
                elif analyzer_name == "failure":
                    comparison = analyzer.compare_models(model_results)
                    self.visualizer.plot_failure_analysis(comparison)

            except Exception as e:
                print(f"❌ Error in {analyzer_name} analysis: {e}")
                import traceback

                traceback.print_exc()

        print(f"\nResults saved in: {self.results_dir}")
        return analysis_results

    def generate_report(self, output_path: Optional[Path] = None):
        """Write a plain-text report of all metrics (model x benchmark)."""
        if output_path is None:
            output_path = self.results_dir / "evaluation_report.txt"

        lines = ["=" * 70, "EVALUATION REPORT", "=" * 70]
        lines.append(f"Dataset: {self.config.dataset}")
        lines.append(f"Data root: {self.config.data_root}")
        lines.append(f"Experiment: {self.config.experiment_name}")
        lines.append(f"Model: {self.config.architecture} / {self.config.model_name}")
        lines.append("")

        if self.evaluation_results:
            summary = pd.DataFrame(list(self.evaluation_results.values()))
            cols = ["model", "benchmark", "n_queries", "AUC", "MRR", "NDCG@10", "HR@10", "MC@10"]
            for bench, group in summary.groupby("benchmark", sort=False):
                lines.append(f"\n[{bench}]")
                lines.append(group[cols].to_string(index=False, float_format="%.4f"))

        lines += ["", "=" * 70, "END OF REPORT", "=" * 70]
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        print(f"✅ Report saved: {output_path}")
