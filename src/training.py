import copy
import gc
import os
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import Trainer, TrainingArguments

from src.data import TextDataset
from src.model import ModelBuilder
from src.utils import Device


class MetricComputer:
    @staticmethod
    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision_weighted": p_w,
            "recall_weighted": r_w,
            "f1_weighted": f1_w,
            "precision_macro": p_m,
            "recall_macro": r_m,
            "f1_macro": f1_m,
        }


class TrainingRun:
    def __init__(self, model, tokenizer, frames, config, output_dir, run_id):
        self.model = model
        self.tokenizer = tokenizer
        self.frames = frames
        self.config = config
        self.output_dir = output_dir
        self.run_id = run_id

    def execute(self):
        max_len = self.config["model"]["max_length"]
        train_ds = TextDataset(self.frames["train"], self.tokenizer, max_len)
        val_ds = TextDataset(self.frames["val"], self.tokenizer, max_len)
        test_ds = TextDataset(self.frames["test"], self.tokenizer, max_len)

        args = self._build_args()
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            compute_metrics=MetricComputer.compute,
        )

        t0 = time.time()
        trainer.train()
        train_time = time.time() - t0

        out = trainer.predict(test_ds)
        preds = np.argmax(out.predictions, axis=-1)
        return {
            "metrics": out.metrics,
            "confusion_matrix": confusion_matrix(out.label_ids, preds),
            "train_time": train_time,
            "run_id": self.run_id,
            "predictions": preds,
            "labels": out.label_ids,
        }

    def _build_args(self):
        tcfg = self.config["training"]
        run_dir = os.path.join(self.output_dir, f"run_{self.run_id}")
        os.makedirs(run_dir, exist_ok=True)
        return TrainingArguments(
            output_dir=run_dir,
            num_train_epochs=tcfg["num_epochs"],
            per_device_train_batch_size=tcfg["batch_size"],
            per_device_eval_batch_size=tcfg["eval_batch_size"],
            gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
            learning_rate=tcfg["learning_rate"],
            warmup_ratio=tcfg["warmup_ratio"],
            weight_decay=tcfg["weight_decay"],
            max_grad_norm=tcfg["max_grad_norm"],
            fp16=tcfg["fp16"] and torch.cuda.is_available(),
            logging_steps=tcfg["logging_steps"],
            eval_strategy="steps",
            eval_steps=tcfg["eval_steps"],
            save_strategy=tcfg["save_strategy"],
            optim=tcfg["optim"] if torch.cuda.is_available() else "adamw_torch",
            report_to=tcfg["report_to"],
            remove_unused_columns=False,
            seed=self.config["project"]["seed"],
            gradient_checkpointing=tcfg.get("gradient_checkpointing", False)
            and torch.cuda.is_available(),
            dataloader_pin_memory=False,
        )


class ResourceCleaner:
    @staticmethod
    def release(*objs):
        for o in objs:
            del o
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


class HyperparameterSweep:
    def __init__(self, config, num_labels, fold, output_dir):
        self.config = config
        self.num_labels = num_labels
        self.fold = fold
        self.output_dir = output_dir

    def run(self):
        grid = self.config["hyperparameter_search"]
        results = []
        for lr in grid["learning_rates"]:
            for r in grid["lora_ranks"]:
                for ep in grid["epochs"]:
                    print(f"  hp: lr={lr} r={r} ep={ep}")
                    cfg = copy.deepcopy(self.config)
                    cfg["training"]["learning_rate"] = lr
                    cfg["training"]["num_epochs"] = ep
                    cfg["qlora"]["r"] = r
                    results.append(self._train_one(cfg, lr, r, ep))
        return results

    def _train_one(self, cfg, lr, r, ep):
        try:
            model, tok = ModelBuilder(cfg, self.num_labels).build()
            res = TrainingRun(model, tok, self.fold, cfg,
                              os.path.join(self.output_dir, "hp_search"), 0).execute()
            payload = {
                "learning_rate": lr, "lora_rank": r, "epochs": ep,
                "accuracy": res["metrics"].get("test_accuracy", 0),
                "f1_weighted": res["metrics"].get("test_f1_weighted", 0),
                "f1_macro": res["metrics"].get("test_f1_macro", 0),
                "train_time": res["train_time"],
            }
            ResourceCleaner.release(model)
            return payload
        except Exception as e:
            return {"learning_rate": lr, "lora_rank": r, "epochs": ep, "error": str(e)}


class AblationSweep:
    def __init__(self, config, num_labels, fold, output_dir):
        self.config = config
        self.num_labels = num_labels
        self.fold = fold
        self.output_dir = output_dir

    def run(self):
        results = []
        for abl in self.config["ablation"]["configs"]:
            print(f"  ablation: {abl['name']} -> {abl['target_modules']}")
            cfg = copy.deepcopy(self.config)
            cfg["qlora"]["target_modules"] = abl["target_modules"]
            results.append(self._train_one(cfg, abl))
        return results

    def _train_one(self, cfg, abl):
        try:
            model, tok = ModelBuilder(cfg, self.num_labels).build()
            res = TrainingRun(model, tok, self.fold, cfg,
                              os.path.join(self.output_dir, "ablation"), 0).execute()
            payload = {
                "name": abl["name"],
                "target_modules": abl["target_modules"],
                "accuracy": res["metrics"].get("test_accuracy", 0),
                "f1_weighted": res["metrics"].get("test_f1_weighted", 0),
                "f1_macro": res["metrics"].get("test_f1_macro", 0),
                "train_time": res["train_time"],
            }
            ResourceCleaner.release(model)
            return payload
        except Exception as e:
            return {"name": abl["name"], "error": str(e)}


class ScalabilitySweep:
    def __init__(self, config, num_labels, fold, output_dir):
        self.config = config
        self.num_labels = num_labels
        self.fold = fold
        self.output_dir = output_dir

    def run(self):
        spec = self.config["scalability"]
        out = []
        for frac in spec["dataset_fractions"]:
            n = max(1, int(len(self.fold["train"]) * frac))
            subset = self.fold["train"].sample(n=n, random_state=42).reset_index(drop=True)
            for bs in spec["batch_sizes"]:
                print(f"  scale: frac={frac} bs={bs} n={n}")
                out.append(self._train_one(subset, frac, bs, n))
        return out

    def _train_one(self, subset, frac, bs, n):
        cfg = copy.deepcopy(self.config)
        cfg["training"]["batch_size"] = bs
        cfg["training"]["num_epochs"] = 1
        t0 = time.time()
        try:
            model, tok = ModelBuilder(cfg, self.num_labels).build()
            mini = {"train": subset, "val": self.fold["val"], "test": self.fold["val"]}
            res = TrainingRun(model, tok, mini, cfg,
                              os.path.join(self.output_dir, "scale"), 0).execute()
            payload = {
                "fraction": frac, "batch_size": bs, "n_train": n,
                "time": time.time() - t0,
                "accuracy": res["metrics"].get("test_accuracy", 0.0),
                "max_memory_mb": Device.peak_memory_mb(),
            }
            ResourceCleaner.release(model)
            return payload
        except Exception as e:
            return {"fraction": frac, "batch_size": bs, "n_train": n, "error": str(e)}
