"""Central hyperparameter config. All experiments import from here."""
from typing import Dict

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# LoRA
TARGET_MODULES = ["q_proj", "v_proj"]
MAX_RANK = 32

# Client heterogeneity
# Reflects realistic device tiers: edge / mid / workstation / server
RANK_DISTRIBUTION: Dict[str, int] = {
    "r4":  20,  # edge devices (8GB VRAM)
    "r8":  20,  # mid-range (16GB VRAM)
    "r16":  5,  # workstation (24GB VRAM)
    "r32":  5,  # server (48GB VRAM)
}
NUM_CLIENTS = 50
CLIENTS_PER_ROUND = 10   # 2× original; 96GB VRAM handles this easily

# Training — tuned for 2× RTX PRO 6000 Blackwell (96GB VRAM each)
LR = 2e-4
BATCH_SIZE = 8            # fits comfortably in 96GB with 7B model
GRAD_ACCUM_STEPS = 4      # effective batch = 32
STEPS_PER_ROUND = 200     # more steps per round for stronger signal
NUM_ROUNDS = 30           # 3× original for better convergence curves

# Evaluation
EVAL_SAMPLES = 500
F1_SAMPLES = 1000

# Random seeds — 5 seeds for robust statistics
SEEDS = [42, 43, 44, 45, 46]

# Non-IID Dirichlet concentration
ALPHA_VALUES = [0.5, 0.1]  # 0.5=moderate, 0.1=hard non-IID

# Methods
METHODS = [
    "homo_r4",
    "homo_r8",
    "hetero_pad",
    "flexlora",
    "hetero_spa",
]

# GPU — use GPU 0 for training, GPU 1 for evaluation (parallel)
TRAIN_DEVICE = "cuda:0"
EVAL_DEVICE  = "cuda:1"

# Paths
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
