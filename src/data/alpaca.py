"""
Alpaca-52k instruction-following dataset.

Uses tatsu-lab/alpaca from HuggingFace.
Metrics: ROUGE-L, BLEU, perplexity, hallucination_rate (meaningful here — generation task).
"""

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import List, Dict, Tuple


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

EVAL_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


class AlpacaClientDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = PROMPT_TEMPLATE.format(
            instruction=s["instruction"],
            input=s.get("input", ""),
            output=s["output"],
        )
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def load_alpaca(
    tokenizer,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    train_size: int = 40000,
    test_size: int = 200,
    max_length: int = 512,
) -> Tuple[List[AlpacaClientDataset], List[Dict]]:
    """Load Alpaca, partition for FL."""
    ds = load_dataset("tatsu-lab/alpaca")
    data = ds["train"].shuffle(seed=seed)

    # Filter out empty outputs
    data = data.filter(lambda x: len(x["output"].strip()) > 5)

    train_data = data.select(range(min(train_size, len(data) - test_size)))
    test_data = data.select(range(len(data) - test_size, len(data)))

    # For Alpaca, create roughly IID partition (no class structure for Dirichlet)
    # Use simple random partition weighted by alpha (simulate heterogeneity via size imbalance)
    rng = np.random.default_rng(seed)
    all_samples = list(train_data)
    rng.shuffle(all_samples)

    # Random partition (uniform for instruction data — no natural class labels)
    indices = np.arange(len(all_samples))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)

    client_datasets = []
    for cid, split in enumerate(splits):
        samples = [all_samples[i] for i in split]
        client_datasets.append(AlpacaClientDataset(samples, tokenizer, max_length))

    # Eval samples
    eval_samples = []
    for row in test_data:
        prompt = EVAL_PROMPT_TEMPLATE.format(
            instruction=row["instruction"],
            input=row.get("input", ""),
        )
        eval_samples.append({"prompt": prompt, "output": row["output"]})

    return client_datasets, eval_samples
