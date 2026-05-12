import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class PlotWriter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, name):
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, name), dpi=120)
        plt.close()


class ConfusionMatrixPlot(PlotWriter):
    def render(self, cm, class_names, dataset_name, run_id):
        size = max(6, len(class_names) * 0.6)
        fig, ax = plt.subplots(figsize=(size, size * 0.9))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{dataset_name} - confusion matrix (run {run_id})")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        self._save(f"confusion_matrix_run{run_id}.png")


class MetricsAggregator(PlotWriter):
    KEYS = ["test_accuracy", "test_precision_weighted", "test_recall_weighted",
            "test_f1_weighted", "test_precision_macro", "test_recall_macro", "test_f1_macro"]

    def render(self, runs, dataset_name):
        matrix = {k: [] for k in self.KEYS}
        for r in runs:
            for k in self.KEYS:
                if k in r["metrics"]:
                    matrix[k].append(r["metrics"][k])

        summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
                   for k, v in matrix.items() if v}

        labels = [k.replace("test_", "") for k in self.KEYS if matrix[k]]
        data = [matrix[k] for k in self.KEYS if matrix[k]]
        if data:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.boxplot(data, labels=labels)
            ax.set_title(f"{dataset_name} - metrics across {len(runs)} runs")
            ax.set_ylabel("Score")
            plt.xticks(rotation=30, ha="right")
            self._save("metrics_boxplot.png")
        return summary


class DatasetComparisonPlot(PlotWriter):
    def render(self, metrics_a, metrics_b, names):
        keys = [k for k in metrics_a if k in metrics_b]
        means_a = [metrics_a[k]["mean"] for k in keys]
        means_b = [metrics_b[k]["mean"] for k in keys]
        stds_a = [metrics_a[k]["std"] for k in keys]
        stds_b = [metrics_b[k]["std"] for k in keys]

        x = np.arange(len(keys))
        w = 0.4
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w / 2, means_a, w, yerr=stds_a, capsize=4, label=names[0])
        ax.bar(x + w / 2, means_b, w, yerr=stds_b, capsize=4, label=names[1])
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("test_", "") for k in keys], rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Dataset comparison")
        ax.legend()
        self._save("dataset_comparison.png")


class HyperparameterPlot(PlotWriter):
    def render(self, results, dataset_name):
        rows = [r for r in results if "error" not in r]
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.3)))
        labels = [f"lr={r['learning_rate']} r={r['lora_rank']} ep={r['epochs']}" for r in rows]
        ax.barh(labels, [r["accuracy"] for r in rows])
        ax.set_xlabel("Accuracy")
        ax.set_title(f"{dataset_name} - hyperparameter grid")
        self._save("hyperparameter_results.png")


class ScalabilityPlot(PlotWriter):
    def render(self, results, dataset_name):
        rows = [r for r in results if "error" not in r]
        if not rows:
            return
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for bs in sorted({r["batch_size"] for r in rows}):
            series = sorted([r for r in rows if r["batch_size"] == bs], key=lambda r: r["fraction"])
            if not series:
                continue
            axes[0].plot([r["fraction"] for r in series], [r["time"] for r in series],
                         marker="o", label=f"bs={bs}")
            axes[1].plot([r["fraction"] for r in series], [r["accuracy"] for r in series],
                         marker="o", label=f"bs={bs}")
        axes[0].set_xlabel("Fraction")
        axes[0].set_ylabel("Time (s)")
        axes[0].set_title(f"{dataset_name} - time vs fraction")
        axes[0].legend()
        axes[1].set_xlabel("Fraction")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{dataset_name} - accuracy vs fraction")
        axes[1].legend()
        self._save("scalability.png")
