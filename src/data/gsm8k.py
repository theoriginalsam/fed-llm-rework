"""
GSM8K math reasoning dataset.

Task: grade-school math word problems with chain-of-thought answers.
Metric: exact match on final numeric answer (extracted from "#### <number>" format).
"""

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import List, Dict, Tuple


PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "At the end, write the final answer as: #### <number>\n\n"
    "Problem: {question}\n\nSolution: {answer}"
)

EVAL_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "At the end, write the final answer as: #### <number>\n\n"
    "Problem: {question}\n\nSolution:"
)


class GSM8KClientDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = PROMPT_TEMPLATE.format(question=s["question"], answer=s["answer"])
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


def load_gsm8k(
    tokenizer,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    max_length: int = 512,
) -> Tuple[List[GSM8KClientDataset], List[Dict]]:
    """Load GSM8K and partition across clients."""
    ds = load_dataset("openai/gsm8k", "main")
    train_data = list(ds["train"])
    test_data = list(ds["test"])

    rng = np.random.default_rng(seed)
    rng.shuffle(train_data)

    # GSM8K is small (7473 train) — divide evenly, some clients may have few samples
    splits = np.array_split(np.arange(len(train_data)), num_clients)
    client_datasets = []
    for split in splits:
        samples = [train_data[i] for i in split]
        client_datasets.append(GSM8KClientDataset(samples, tokenizer, max_length))

    eval_samples = []
    for row in test_data[:100]:
        prompt = EVAL_PROMPT_TEMPLATE.format(question=row["question"])
        eval_samples.append({"prompt": prompt, "answer": row["answer"]})

    return client_datasets, eval_samples
