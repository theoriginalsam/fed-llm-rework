"""Per-dataset configurations."""

YELP_CONFIG = {
    "name": "yelp_review_full",
    "task": "classification",
    "num_classes": 5,
    "max_length": 256,
    "train_size": 650000,
    "test_size": 50000,
    "prompt_template": (
        "Analyze the review and classify the rating as: "
        "1 stars, 2 stars, 3 stars, 4 stars, or 5 stars.\n\n"
        "Review: {text}\n\nRating: {label}"
    ),
    "label_tokens": ["1", "2", "3", "4", "5"],
    "metrics": ["accuracy", "f1_macro", "perplexity"],
}

ALPACA_CONFIG = {
    "name": "tatsu-lab/alpaca",
    "task": "instruction_following",
    "max_length": 512,
    "train_size": 40000,
    "test_size": 12000,
    "prompt_template": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    ),
    "metrics": ["rouge_l", "bleu", "perplexity", "hallucination_rate"],
}

GSM8K_CONFIG = {
    "name": "openai/gsm8k",
    "subset": "main",
    "task": "math_reasoning",
    "max_length": 512,
    "train_size": 7473,
    "test_size": 1319,
    "prompt_template": (
        "Solve the following math problem step by step. "
        "At the end, state the final answer as a number.\n\n"
        "Problem: {question}\n\nSolution: {answer}"
    ),
    "metrics": ["exact_match", "perplexity"],
    "answer_extractor": r"####\s*(-?\d+(?:\.\d+)?)",  # GSM8K answer format
}
