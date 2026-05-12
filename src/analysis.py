import os
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns


class ExploratoryAnalysis:
    def __init__(self, train_df, test_df, class_names, dataset_name, output_base):
        self.train_df = train_df
        self.test_df = test_df
        self.class_names = class_names
        self.dataset_name = dataset_name
        self.out_dir = os.path.join(output_base, "eda", dataset_name)
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self):
        summary = {
            "dataset": self.dataset_name,
            "n_train": len(self.train_df),
            "n_test": len(self.test_df),
            "n_classes": len(self.class_names),
            "class_names": self.class_names,
        }
        summary.update(self._class_distribution())
        summary.update(self._length_stats())
        return summary

    def _class_distribution(self):
        train_counts = Counter(self.train_df["label"])
        test_counts = Counter(self.test_df["label"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        names = self.class_names
        sns.barplot(x=names,
                    y=[train_counts.get(i, 0) for i in range(len(names))], ax=axes[0])
        axes[0].set_title(f"{self.dataset_name} - train classes")
        axes[0].tick_params(axis="x", rotation=45)
        sns.barplot(x=names,
                    y=[test_counts.get(i, 0) for i in range(len(names))], ax=axes[1])
        axes[1].set_title(f"{self.dataset_name} - test classes")
        axes[1].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "class_distribution.png"), dpi=120)
        plt.close()

        return {
            "train_class_counts": {names[k]: int(v) for k, v in train_counts.items()},
            "test_class_counts": {names[k]: int(v) for k, v in test_counts.items()},
        }

    def _length_stats(self):
        train_lens = self.train_df["text"].str.split().str.len()
        test_lens = self.test_df["text"].str.split().str.len()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(train_lens, bins=50, alpha=0.6, label="train")
        ax.hist(test_lens, bins=50, alpha=0.6, label="test")
        ax.set_xlabel("Words")
        ax.set_ylabel("Count")
        ax.set_title(f"{self.dataset_name} - text length")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "text_lengths.png"), dpi=120)
        plt.close()

        return {
            "text_length_words": {
                "train_mean": float(train_lens.mean()),
                "train_median": float(train_lens.median()),
                "train_min": int(train_lens.min()),
                "train_max": int(train_lens.max()),
                "test_mean": float(test_lens.mean()),
                "test_median": float(test_lens.median()),
            }
        }
