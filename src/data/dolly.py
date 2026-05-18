"""
Dolly-15K dataset for task-heterogeneous FL — mirrors FlexLoRA's evaluation setup.

Partition: 60 clients total (50 train + 10 unseen eval).
Each client is assigned a dominant task category from Dolly's 8 categories.
Unseen eval clients are held out entirely from training — zero-shot ROUGE-L
on their data is the primary metric (same as FlexLoRA paper).

Categories: brainstorming, classification, closed_qa, creative_writing,
            general_qa, information_extraction, open_qa, summarization
"""

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import List, Dict, Tuple


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{context}\n\n"
    "### Response:\n{response}"
)

EVAL_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{context}\n\n"
    "### Response:\n"
)

DOLLY_CATEGORIES = [
    "brainstorming", "classification", "closed_qa", "creative_writing",
    "general_qa", "information_extraction", "open_qa", "summarization",
]

NUM_TRAIN_CLIENTS = 50
NUM_UNSEEN_CLIENTS = 10
TOTAL_CLIENTS = NUM_TRAIN_CLIENTS + NUM_UNSEEN_CLIENTS


class DollyClientDataset(Dataset):
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
            context=s.get("context", ""),
            response=s["response"],
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


def load_dolly_federated(
    tokenizer,
    seed: int = 42,
    max_length: int = 512,
) -> Tuple[List[DollyClientDataset], List[List[Dict]]]:
    """
    Load Dolly-15K and partition into federated clients.

    Returns:
        train_datasets: list of 50 DollyClientDataset (for training)
        unseen_samples:  list of 10 raw sample lists (for zero-shot eval)
    """
    rng = np.random.default_rng(seed)
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    # Group by category
    by_cat: Dict[str, List[Dict]] = {cat: [] for cat in DOLLY_CATEGORIES}
    for ex in ds:
        cat = ex.get("category", "general_qa")
        if cat in by_cat:
            by_cat[cat].append(ex)

    # Shuffle each category
    for cat in by_cat:
        rng.shuffle(by_cat[cat])

    # Assign clients to categories (round-robin, 60 clients across 8 categories)
    client_categories = [DOLLY_CATEGORIES[i % len(DOLLY_CATEGORIES)] for i in range(TOTAL_CLIENTS)]
    rng.shuffle(client_categories)

    # Allocate samples per client within its category
    cat_cursors = {cat: 0 for cat in DOLLY_CATEGORIES}
    samples_per_client = []
    for cat in client_categories:
        pool = by_cat[cat]
        n = max(10, len(pool) // (TOTAL_CLIENTS // len(DOLLY_CATEGORIES) + 1))
        start = cat_cursors[cat]
        end = min(start + n, len(pool))
        samples_per_client.append(pool[start:end])
        cat_cursors[cat] = end

    # Split: clients 0-49 = train, 50-59 = unseen eval
    train_datasets = [
        DollyClientDataset(samples_per_client[i], tokenizer, max_length)
        for i in range(NUM_TRAIN_CLIENTS)
    ]
    unseen_samples = [samples_per_client[i] for i in range(NUM_TRAIN_CLIENTS, TOTAL_CLIENTS)]

    return train_datasets, unseen_samples
