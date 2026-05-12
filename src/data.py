import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset


class DatasetSource:
    def __init__(self, key, spec):
        self.key = key
        self.spec = spec

    def fetch(self):
        raw = load_dataset(self.spec["hf_id"])
        train = self._to_frame(raw["train"])
        test = self._to_frame(raw["test"])

        if self.spec.get("max_samples_train"):
            train = train.sample(n=min(self.spec["max_samples_train"], len(train)),
                                 random_state=42).reset_index(drop=True)
        if self.spec.get("max_samples_test"):
            test = test.sample(n=min(self.spec["max_samples_test"], len(test)),
                               random_state=42).reset_index(drop=True)
        return train, test, self.spec["class_names"]

    def _to_frame(self, split):
        df = split.to_pandas()
        cols = self.spec["text_columns"]
        if len(cols) == 1:
            df["text"] = df[cols[0]].astype(str)
        else:
            df["text"] = df[cols[0]].astype(str)
            for c in cols[1:]:
                df["text"] = df["text"] + " " + df[c].astype(str)
        df["label"] = df[self.spec["label_column"]].astype(int)
        return df[["text", "label"]]


class FoldFactory:
    def __init__(self, n_splits, seed, val_ratio=0.1):
        self.n_splits = n_splits
        self.seed = seed
        self.val_ratio = val_ratio

    def make(self, df):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        folds = []
        for fold_id, (tr_idx, te_idx) in enumerate(skf.split(df["text"].values, df["label"].values)):
            train_part = df.iloc[tr_idx].reset_index(drop=True)
            test_part = df.iloc[te_idx].reset_index(drop=True)
            train_part, val_part = train_test_split(
                train_part, test_size=self.val_ratio,
                stratify=train_part["label"], random_state=self.seed,
            )
            folds.append({
                "fold_id": fold_id,
                "train": train_part.reset_index(drop=True),
                "val": val_part.reset_index(drop=True),
                "test": test_part.reset_index(drop=True),
            })
        return folds


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.texts = frame["text"].tolist()
        self.labels = frame["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
