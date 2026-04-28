"""
Yelp Review Full dataset loader with Dirichlet non-IID partitioning.

Supports alpha=0.5 (mild non-IID) and alpha=0.1 (hard non-IID).
Removes hallucination_rate metric — not meaningful for classification.
"""

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import List, Dict, Tuple


PROMPT_TEMPLATE = (
    "Analyze the review and classify the rating as: "
    "1 stars, 2 stars, 3 stars, 4 stars, or 5 stars.\n\n"
    "Review: {text}\n\nRating: {label}"
)

LABEL_MAP = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}


class YelpClientDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        sample = self.samples[idx]
        prompt = PROMPT_TEMPLATE.format(text=sample["text"][:400], label=LABEL_MAP[sample["label"]])
        enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        result = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": sample["label"],
            "prompt": prompt,
        }
        self._cache[idx] = result
        return result


class YelpEvalSample:
    """Lightweight eval container — no tokenization at load time."""
    def __init__(self, prompt: str, label: int):
        self.prompt = prompt
        self.label = label

    def __getitem__(self, key):
        return {"prompt": self.prompt, "label": self.label}[key]


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition sample indices across clients using Dirichlet distribution.

    Lower alpha = more heterogeneous (less IID).
    alpha=0.5: moderate non-IID; alpha=0.1: strongly non-IID.
    """
    rng = np.random.default_rng(seed)
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        proportions = rng.dirichlet(np.ones(num_clients) * alpha)
        proportions = (proportions / proportions.sum() * len(class_indices[c])).astype(int)
        # Fix rounding
        proportions[-1] = len(class_indices[c]) - proportions[:-1].sum()

        rng.shuffle(class_indices[c])
        start = 0
        for cid, count in enumerate(proportions):
            client_indices[cid].extend(class_indices[c][start:start + count].tolist())
            start += count

    return client_indices


def load_yelp(
    tokenizer,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    train_size: int = 650000,
    test_size: int = 5000,
    max_length: int = 256,
) -> Tuple[List[YelpClientDataset], List[Dict]]:
    """
    Load Yelp, partition for FL, return (client_datasets, eval_samples).
    """
    ds = load_dataset("yelp_review_full")
    train_data = ds["train"].shuffle(seed=seed).select(range(min(train_size, len(ds["train"]))))
    test_data = ds["test"].shuffle(seed=seed).select(range(min(test_size, len(ds["test"]))))

    labels = np.array(train_data["label"])
    all_samples = [{"text": t, "label": l} for t, l in zip(train_data["text"], train_data["label"])]

    # Partition
    partition = dirichlet_partition(labels, num_clients, 5, alpha, seed)

    client_datasets = []
    for cid in range(num_clients):
        client_samples = [all_samples[i] for i in partition[cid]]
        client_datasets.append(YelpClientDataset(client_samples, tokenizer, max_length))

    # Eval samples: prompt WITHOUT label so model predicts the label token
    EVAL_TEMPLATE = (
        "Analyze the review and classify the rating as: "
        "1 stars, 2 stars, 3 stars, 4 stars, or 5 stars.\n\n"
        "Review: {text}\n\nRating:"
    )
    eval_samples = []
    for row in test_data:
        prompt = EVAL_TEMPLATE.format(text=row["text"][:400])
        eval_samples.append({"prompt": prompt, "label": row["label"]})

    return client_datasets, eval_samples
