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
#  Batched generation helper
# ──────────────────────────────────────────────

def _batch_generate(
    model, tokenizer, prompts: List[str], device: str,
    max_new_tokens: int, batch_size: int = 4,
) -> List[str]:
    """Left-padded batched generation for decoder-only models (~4× faster than one-by-one)."""
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", max_length=512,
                        truncation=True, padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for j in range(len(batch)):
            results.append(tokenizer.decode(gen[j][input_len:], skip_special_tokens=True))
    tokenizer.padding_side = original_padding_side
    return results


# ──────────────────────────────────────────────
#  Alpaca: ROUGE-L + BLEU + hallucination rate
# ──────────────────────────────────────────────

def evaluate_alpaca(model, tokenizer, samples: List[Dict], device: str) -> Dict[str, float]:
    from rouge_score import rouge_scorer
    import sacrebleu

    prompts = [s["prompt"] for s in samples]
    references = [s["output"] for s in samples]
    gen_texts = _batch_generate(model, tokenizer, prompts, device, max_new_tokens=128)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores, bleu_scores, hallucinated = [], [], 0

    for gen_text, reference in zip(gen_texts, references):
        if len(gen_text.strip()) < 3:
            hallucinated += 1
            gen_text = ""
        rouge_scores.append(scorer.score(reference, gen_text)["rougeL"].fmeasure)
        bleu_scores.append(sacrebleu.sentence_bleu(gen_text, [reference]).score)

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
    prompts = [s["prompt"] for s in samples]
    gen_texts = _batch_generate(model, tokenizer, prompts, device, max_new_tokens=256)

    correct = 0
    for sample, gen_text in zip(samples, gen_texts):
        gold_answer = extract_gsm8k_answer(sample["answer"])
        pred_answer = extract_gsm8k_answer(gen_text)
        if gold_answer and pred_answer and gold_answer.strip() == pred_answer.strip():
            correct += 1

    return {"exact_match": float(correct / max(len(samples), 1))}


# ──────────────────────────────────────────────
#  Dolly unseen-client evaluation (FlexLoRA metric)
# ──────────────────────────────────────────────

def evaluate_unseen_clients(
    model,
    tokenizer,
    unseen_samples: List[List[Dict]],
    device: str,
    n_samples_per_client: int = 30,
) -> Dict[str, float]:
    """
    Zero-shot ROUGE-L on held-out clients never seen during training.
    Mirrors FlexLoRA's evaluation protocol exactly.

    unseen_samples: list of raw Dolly sample lists (one per unseen client)
    Returns weighted average ROUGE-L across all unseen clients.
    """
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    from src.data.dolly import EVAL_PROMPT_TEMPLATE
    all_rouge = []

    for client_samples in unseen_samples:
        subset = client_samples[:n_samples_per_client]
        if not subset:
            continue
        prompts = [
            EVAL_PROMPT_TEMPLATE.format(
                instruction=s["instruction"],
                context=s.get("context", ""),
            )
            for s in subset
        ]
        references = [s["response"] for s in subset]
        gen_texts = _batch_generate(model, tokenizer, prompts, device, max_new_tokens=128)

        client_rouge = [
            scorer.score(ref, gen)["rougeL"].fmeasure
            for ref, gen in zip(references, gen_texts)
        ]
        all_rouge.append(float(np.mean(client_rouge)))

    return {
        "rouge_l_unseen": float(np.mean(all_rouge)) if all_rouge else 0.0,
        "rouge_l_unseen_std": float(np.std(all_rouge)) if all_rouge else 0.0,
    }


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
