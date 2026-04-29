"""
Evaluation metrics for all three tasks.

- Yelp: accuracy, F1-macro, perplexity
- Alpaca: ROUGE-L, BLEU, perplexity, hallucination_rate
- GSM8K: exact_match on final answer, perplexity
"""

import re
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.metrics import f1_score, accuracy_score
from peft import LoraConfig, get_peft_model, TaskType
from src.clients.lora_client import make_lora_model, inject_lora_weights
from config.base_config import TARGET_MODULES
import copy


def _load_eval_model(base_model, global_lora_weights, rank, device):
    # Deepcopy through CPU: base_model stays on CPU until eval model is deleted
    # (see evaluate_model). Having both on GPU simultaneously = OOM on 24GB.
    base_model.to("cpu")
    model = make_lora_model(copy.deepcopy(base_model), rank, TARGET_MODULES)
    model = model.to(device)
    if global_lora_weights:
        inject_lora_weights(model, global_lora_weights, device)
    model.eval()
    return model


# ──────────────────────────────────────────────
#  Perplexity
# ──────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, tokenizer, texts: List[str], device: str, max_length: int = 256) -> float:
    total_nll = 0.0
    total_tokens = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", max_length=max_length,
                        truncation=True, padding=False).to(device)
        out = model(**enc, labels=enc["input_ids"])
        n_tokens = enc["input_ids"].shape[1]
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens
    return float(np.exp(total_nll / max(total_tokens, 1)))


# ──────────────────────────────────────────────
#  Yelp: accuracy + F1
# ──────────────────────────────────────────────

YELP_LABEL_TOKENS = ["1", "2", "3", "4", "5"]

@torch.no_grad()
def evaluate_yelp(model, tokenizer, samples: List[Dict], device: str) -> Dict[str, float]:
    """
    Correct evaluation: prompt ends with 'Rating:' (no label).
    Score each label token at the final position — this is the model's
    prediction FOR the label, not after it.
    """
    # Pre-compute label token IDs once
    label_token_ids = []
    for tok in YELP_LABEL_TOKENS:
        ids = tokenizer(" " + tok, add_special_tokens=False)["input_ids"]
        label_token_ids.append(ids[-1])

    preds, golds = [], []
    for sample in samples:
        enc = tokenizer(sample["prompt"], return_tensors="pt",
                        max_length=256, truncation=True).to(device)
        out = model(**enc)
        logits = out.logits[0, -1, :]   # prediction at last position = label position

        scores = [logits[tid].item() for tid in label_token_ids]
        preds.append(int(np.argmax(scores)))
        golds.append(sample["label"])

    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1_macro": float(f1)}


# ──────────────────────────────────────────────
#  Alpaca: ROUGE-L + BLEU + hallucination rate
# ──────────────────────────────────────────────

def evaluate_alpaca(model, tokenizer, samples: List[Dict], device: str) -> Dict[str, float]:
    from rouge_score import rouge_scorer
    import sacrebleu

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores, bleu_scores, hallucinated = [], [], 0

    for sample in samples:
        prompt = sample["prompt"]
        reference = sample["output"]

        enc = tokenizer(prompt, return_tensors="pt", max_length=512,
                        truncation=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

        if len(gen_text.strip()) < 3:
            hallucinated += 1
            gen_text = ""

        r = scorer.score(reference, gen_text)["rougeL"].fmeasure
        rouge_scores.append(r)

        bl = sacrebleu.sentence_bleu(gen_text, [reference]).score
        bleu_scores.append(bl)

    n = len(samples)
    return {
        "rouge_l": float(np.mean(rouge_scores)),
        "bleu": float(np.mean(bleu_scores)),
        "hallucination_rate": float(hallucinated / n),
    }


# ──────────────────────────────────────────────
#  GSM8K: exact match on final numeric answer
# ──────────────────────────────────────────────

GSM8K_ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
GENERATED_ANSWER_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*$")


def extract_gsm8k_answer(text: str) -> Optional[str]:
    m = GSM8K_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: last number in text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def evaluate_gsm8k(model, tokenizer, samples: List[Dict], device: str) -> Dict[str, float]:
    correct = 0
    for sample in samples:
        prompt = sample["prompt"]
        gold_answer = extract_gsm8k_answer(sample["answer"])

        enc = tokenizer(prompt, return_tensors="pt", max_length=512,
                        truncation=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_answer = extract_gsm8k_answer(gen_text)

        if gold_answer and pred_answer and gold_answer.strip() == pred_answer.strip():
            correct += 1

    return {"exact_match": float(correct / max(len(samples), 1))}


# ──────────────────────────────────────────────
#  Unified evaluate_model
# ──────────────────────────────────────────────

def evaluate_model(
    base_model,
    tokenizer,
    global_lora_weights: Optional[Dict],
    rank: int,
    test_dataset,
    dataset_config: Dict,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Dispatch to correct evaluation function based on dataset task type.
    """
    model = _load_eval_model(base_model, global_lora_weights, rank, device)
    task = dataset_config["task"]
    samples = list(test_dataset)

    metrics = {}

    if task == "classification":
        metrics.update(evaluate_yelp(model, tokenizer, samples, device))

    elif task == "instruction_following":
        metrics.update(evaluate_alpaca(model, tokenizer, samples, device))

    elif task == "math_reasoning":
        metrics.update(evaluate_gsm8k(model, tokenizer, samples, device))

    # Perplexity for all tasks
    texts = [s.get("text", s.get("prompt", "")) for s in samples[:200]]
    metrics["perplexity"] = compute_perplexity(model, tokenizer, texts, device)

    del model
    torch.cuda.empty_cache()
    base_model.to(device)
    return metrics
