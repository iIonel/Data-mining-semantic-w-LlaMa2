import os

import pandas as pd

from src.analysis import ExploratoryAnalysis
from src.data import DatasetSource, FoldFactory
from src.model import ModelBuilder
from src.plots import (
    ConfusionMatrixPlot,
    DatasetComparisonPlot,
    HyperparameterPlot,
    MetricsAggregator,
    ScalabilityPlot,
)
from src.training import (
    AblationSweep,
    HyperparameterSweep,
    ResourceCleaner,
    ScalabilitySweep,
    TrainingRun,
)
from src.utils import Reproducibility, ResultsIO, SummaryPrinter


class DatasetPipeline:
    def __init__(self, dataset_key, config, options, output_base="results"):
        self.dataset_key = dataset_key
        self.config = config
        self.options = options
        self.output_base = output_base
        self.output_dir = os.path.join(output_base, dataset_key)
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        print(f"\n{'#' * 70}\n# {self.dataset_key}\n{'#' * 70}")

        train_df, test_df, class_names = DatasetSource(
            self.dataset_key, self.config["data"][self.dataset_key],
        ).fetch()
        num_labels = len(class_names)
        print(f"  train={len(train_df)} test={len(test_df)} classes={num_labels}")

        eda_summary = ExploratoryAnalysis(
            train_df, test_df, class_names, self.dataset_key, self.output_base,
        ).run()

        if self.options.skip_training:
            return None

        full_df = pd.concat([train_df, test_df], ignore_index=True)
        runs = self._cv_runs(full_df, num_labels, class_names)

        metrics_summary = MetricsAggregator(self.output_dir).render(runs, self.dataset_key)
        SummaryPrinter.show(metrics_summary, self.dataset_key)

        hp_results = self._maybe_sweep(HyperparameterSweep, "no_hp", num_labels, full_df)
        if hp_results:
            HyperparameterPlot(self.output_dir).render(hp_results, self.dataset_key)

        ablation_results = self._maybe_sweep(AblationSweep, "no_ablation", num_labels, full_df)

        scale_results = self._maybe_sweep(ScalabilitySweep, "no_scale", num_labels, full_df)
        if scale_results:
            ScalabilityPlot(self.output_dir).render(scale_results, self.dataset_key)

        ResultsIO.dump({
            "eda": eda_summary,
            "metrics_summary": metrics_summary,
            "per_run_metrics": [{"seed": r.get("seed"), "fold_id": r.get("fold_id"),
                                 "metrics": r["metrics"], "train_time": r["train_time"]}
                                for r in runs],
            "hyperparameter_results": hp_results,
            "ablation_results": ablation_results,
            "scalability_results": scale_results,
        }, os.path.join(self.output_dir, "all_results.json"))

        return metrics_summary

    def _cv_runs(self, full_df, num_labels, class_names):
        seeds = self.config["project"]["seeds"]
        if self.options.runs:
            seeds = seeds[: self.options.runs]

        plotter = ConfusionMatrixPlot(self.output_dir)
        cv = self.config["cv"]
        runs = []
        for seed in seeds:
            Reproducibility.seed_all(seed)
            folds = FoldFactory(cv["n_splits"], seed, cv["val_ratio"]).make(full_df)
            for fold in folds:
                print(f"\n  seed={seed} fold={fold['fold_id']}")
                model, tok = ModelBuilder(self.config, num_labels, smoke=self.options.smoke).build()
                run_id = seed * 100 + fold["fold_id"]
                res = TrainingRun(model, tok, fold, self.config, self.output_dir, run_id).execute()
                res["seed"] = seed
                res["fold_id"] = fold["fold_id"]
                runs.append(res)
                plotter.render(res["confusion_matrix"], class_names, self.dataset_key, run_id)
                ResourceCleaner.release(model)
                if self.options.smoke:
                    return runs
        return runs

    def _maybe_sweep(self, sweep_cls, skip_flag, num_labels, full_df):
        if getattr(self.options, skip_flag) or self.options.smoke:
            return []
        print(f"\n  {sweep_cls.__name__}")
        cv = self.config["cv"]
        fold = FoldFactory(cv["n_splits"], 42, cv["val_ratio"]).make(full_df)[0]
        return sweep_cls(self.config, num_labels, fold, self.output_dir).run()


class FullPipeline:
    def __init__(self, config, options):
        self.config = config
        self.options = options

    def run(self):
        datasets = self._resolve_datasets()
        metrics_by_dataset = {}
        for ds in datasets:
            metrics = DatasetPipeline(ds, self.config, self.options, self.options.output).run()
            if metrics is not None:
                metrics_by_dataset[ds] = metrics

        if len(metrics_by_dataset) == 2:
            DatasetComparisonPlot(self.options.output).render(
                metrics_by_dataset["ag_news"],
                metrics_by_dataset["dbpedia_14"],
                names=("ag_news", "dbpedia_14"),
            )

        print(f"\nDone. Output: {self.options.output}")

    def _resolve_datasets(self):
        if self.options.dataset == "both":
            return ["ag_news", "dbpedia_14"]
        return [self.options.dataset]
