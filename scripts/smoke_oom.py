import argparse
import sys

import pandas as pd
import torch

from src.data import TextDataset
from src.model import ModelBuilder
from src.utils import Config, Env


class DummyFrame:
    @staticmethod
    def build(n, num_labels):
        texts = ["dummy classification input"] * n
        labels = [i % num_labels for i in range(n)]
        return pd.DataFrame({"text": texts, "label": labels})


class OOMProbe:
    def __init__(self, cfg, n):
        self.cfg = cfg
        self.n = n
        self.num_labels = cfg["data"]["ag_news"]["num_classes"]

    def run(self):
        torch.cuda.reset_peak_memory_stats()
        model, tok = ModelBuilder(self.cfg, num_labels=self.num_labels).build()
        ds = TextDataset(
            DummyFrame.build(self.n, self.num_labels),
            tok,
            self.cfg["model"]["max_length"],
        )
        batch = self._collate(ds)
        model.train()
        model(**batch).loss.backward()
        return self._stats()

    def _collate(self, ds):
        bs = self.cfg["training"]["batch_size"]
        batch = {
            k: torch.stack([ds[i][k] for i in range(bs)]).to("cuda")
            for k in ("input_ids", "attention_mask")
        }
        batch["labels"] = torch.tensor([ds[i]["labels"] for i in range(bs)]).to("cuda")
        return batch

    @staticmethod
    def _stats():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        return peak_mb, total_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.local_4gb.yaml")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA")
        sys.exit(0)

    Env().load().hf_login()
    cfg = Config.from_yaml(args.config)
    peak, total = OOMProbe(cfg, args.n).run()
    print(f"peak {peak:.0f} MB / {total:.0f} MB ({100 * peak / total:.1f}%)")


if __name__ == "__main__":
    main()
