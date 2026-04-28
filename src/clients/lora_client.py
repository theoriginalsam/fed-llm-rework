"""
LoRA client: local training and weight extraction/injection.

Each federated round, a client:
  1. Receives global LoRA weights (projected to its rank).
  2. Trains locally for STEPS_PER_ROUND steps.
  3. Returns updated weights for aggregation.
"""

import torch
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Optional, Tuple
import copy


def make_lora_model(base_model, rank: int, target_modules, lora_alpha_multiplier: int = 2):
    """Wrap a base model with a fresh LoRA adapter."""
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * lora_alpha_multiplier,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return get_peft_model(base_model, config)


def inject_lora_weights(
    model,
    layer_weights: Dict[str, Dict[str, torch.Tensor]],
    device: str = "cuda",
):
    """
    Load (A, B) tensors from layer_weights dict into the model's LoRA parameters.

    Uses exact key construction to avoid substring false-matches
    (e.g. 'layers.1' incorrectly matching 'layers.10', 'layers.11', etc).
    """
    state = model.state_dict()
    for module_name, mats in layer_weights.items():
        # Try both PEFT key formats
        for suffix in ["lora_A.default.weight", "lora_A.weight"]:
            key = f"{module_name}.{suffix}"
            if key in state:
                A = mats["A"].to(device=device, dtype=state[key].dtype)
                if A.shape == state[key].shape:
                    state[key] = A
                break

        for suffix in ["lora_B.default.weight", "lora_B.weight"]:
            key = f"{module_name}.{suffix}"
            if key in state:
                B = mats["B"].to(device=device, dtype=state[key].dtype)
                if B.shape == state[key].shape:
                    state[key] = B
                break

    model.load_state_dict(state, strict=False)


def extract_lora_weights(model, method: str = "full_w") -> Dict[str, torch.Tensor]:
    """
    Extract LoRA weights from model for aggregation.

    method="full_w": return ΔW = B @ A for each layer (used by SPA, FlexLoRA, FLoRA, Homo).
    method="ab_pair": return {"A": ..., "B": ...} for each layer (used by HeteroPad).
    """
    result = {}
    lora_params = {n: p for n, p in model.named_parameters() if "lora_" in n}

    # Group by base module name
    modules = set()
    for name in lora_params:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p in ("lora_A", "lora_B"):
                module_key = ".".join(parts[:i])
                modules.add(module_key)

    for module_key in modules:
        A_key = f"{module_key}.lora_A.default.weight"
        B_key = f"{module_key}.lora_B.default.weight"

        if A_key not in lora_params or B_key not in lora_params:
            # Try alternate naming
            A_key = f"{module_key}.lora_A.weight"
            B_key = f"{module_key}.lora_B.weight"
            if A_key not in lora_params:
                continue

        A = lora_params[A_key].detach().cpu().float()   # (r, d_in)
        B = lora_params[B_key].detach().cpu().float()   # (d_out, r)

        if method == "full_w":
            result[module_key] = B @ A                  # (d_out, d_in)
        elif method == "ab_pair":
            result[module_key] = {"A": A, "B": B}

    return result


def train_client(
    base_model,
    tokenizer,
    rank: int,
    target_modules,
    global_weights: Optional[Dict],
    dataset,
    steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    device: str = "cuda",
    extract_method: str = "full_w",
    pbar_desc: str = "",
) -> Tuple[Dict, float]:
    """
    Full client training step. Returns (extracted_weights, avg_loss).
    """
    from tqdm import tqdm

    model = make_lora_model(copy.deepcopy(base_model), rank, target_modules)
    model = model.to(device)
    model.train()

    if global_weights is not None:
        inject_lora_weights(model, global_weights, device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    total_loss = 0.0
    step = 0
    optimizer.zero_grad()

    data_iter = iter(torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    ))

    pbar = tqdm(total=steps, desc=pbar_desc, leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, loss={postfix}]")

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            ))
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss / grad_accum
        loss.backward()
        total_loss += loss.item() * grad_accum

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        step += 1
        pbar.set_postfix_str(f"{total_loss/step:.4f}")
        pbar.update(1)

    pbar.close()

    weights = extract_lora_weights(model, method=extract_method)
    del model
    torch.cuda.empty_cache()

    return weights, total_loss / steps
