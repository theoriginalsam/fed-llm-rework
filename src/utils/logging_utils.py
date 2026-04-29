"""Lightweight experiment logger: writes JSON + tees to stdout and log file."""

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

        logs_dir = os.path.join(os.path.dirname(results_dir), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.log_path = os.path.join(logs_dir, f"{tag}.log")
        self._log_file = open(self.log_path, "w", buffering=1)

        self._start = time.time()
        self.log(f"Log file: {self.log_path}")

    def log(self, msg: str):
        elapsed = time.time() - self._start
        line = f"[{elapsed:8.1f}s] {msg}"
        print(line, flush=True)
        self._log_file.write(line + "\n")

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
        self._log_file.close()
