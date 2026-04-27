"""Lightweight experiment logger: writes JSON + prints to stdout."""

import json
import os
import time
from typing import Any, Dict, List


class ExperimentLogger:
    def __init__(self, method: str, seed: int, alpha: float, results_dir: str):
        self.method = method
        self.seed = seed
        self.alpha = alpha
        tag = f"{method}_seed{seed}_alpha{str(alpha).replace('.', '')}"
        self.out_path = os.path.join(results_dir, f"{tag}.json")
        os.makedirs(results_dir, exist_ok=True)
        self._start = time.time()

    def log(self, msg: str):
        elapsed = time.time() - self._start
        print(f"[{elapsed:8.1f}s] {msg}", flush=True)

    def save(self, round_results: List[Dict[str, Any]]):
        payload = {
            "method": self.method,
            "seed": self.seed,
            "alpha": self.alpha,
            "rounds": round_results,
        }
        with open(self.out_path, "w") as f:
            json.dump(payload, f, indent=2)
        self.log(f"Results saved to {self.out_path}")
