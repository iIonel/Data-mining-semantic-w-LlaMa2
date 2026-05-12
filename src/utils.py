import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


class Env:
    def __init__(self, path=".env"):
        self.path = Path(path)
        self.values = {}

    def load(self):
        if not self.path.exists():
            return self
        try:
            from dotenv import dotenv_values
            self.values = {k: v for k, v in dotenv_values(self.path).items() if v}
        except ImportError:
            with self.path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    if v:
                        self.values[k.strip()] = v
        for k, v in self.values.items():
            os.environ.setdefault(k, v)
        return self

    def hf_login(self):
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            return False
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        return True


class Reproducibility:
    @staticmethod
    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


class Config(dict):
    @classmethod
    def from_yaml(cls, path):
        with open(path) as fh:
            return cls(yaml.safe_load(fh))


class ResultsIO:
    @staticmethod
    def dump(payload, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=ResultsIO._default)

    @staticmethod
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)


class Device:
    @staticmethod
    def describe():
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"[device] CUDA {torch.cuda.get_device_name(0)} ({props.total_memory / 1e9:.1f} GB)")
            return "cuda"
        print("[device] CPU")
        return "cpu"

    @staticmethod
    def peak_memory_mb():
        return torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0


class SummaryPrinter:
    @staticmethod
    def show(summary, label):
        print(f"\n=== {label} ===")
        for k, v in summary.items():
            if isinstance(v, dict) and "mean" in v:
                print(f"  {k:30s}: {v['mean']:.4f} +/- {v['std']:.4f}")
