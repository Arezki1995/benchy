import argparse
import time
import json
import csv
import hashlib
from typing import Dict, Any, List
import requests
import yaml
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"

# ==============================
# Configuration
# ==============================

DEFAULT_PARAMS: Dict[str, float] = {
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}

DETERMINISM_RUNS = 3


# ==============================
# Utility Functions
# ==============================

def load_eval_suite(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def call_model(model: str, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": params,
    }

    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload)
    latency = time.time() - start

    if response.status_code != 200:
        raise RuntimeError(f"Model call failed: {response.text}")

    data = response.json()

    output_text = data.get("response", "")
    total_tokens = data.get("eval_count", 0)

    tokens_per_sec = total_tokens / latency if latency > 0 else 0

    return {
        "output": output_text.strip(),
        "latency_sec": round(latency, 3),
        "tokens": total_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def hash_output(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ==============================
# Evaluation Logic
# ==============================

def evaluate_test(model: str, test: Dict[str, Any]) -> Dict[str, Any]:
    test_id = test["id"]
    test_type = test["type"]
    prompt = test["prompt"]

    result = call_model(model, prompt, DEFAULT_PARAMS)

    output = result["output"]

    evaluation: Dict[str, Any] = {
        "test_id": test_id,
        "model": model,
        "type": test_type,
        "latency_sec": result["latency_sec"],
        "tokens": result["tokens"],
        "tokens_per_sec": result["tokens_per_sec"],
        "json_valid": None,
        "hallucinated": None,
        "determinism_score": None,
    }

    # JSON Validation
    if test_type == "extraction":
        evaluation["json_valid"] = int(is_valid_json(output))

    # Hallucination trap
    if test_type == "hallucination":
        lower = output.lower()
        if "don't know" in lower or "do not know" in lower or "no information" in lower or "__wrong__" in lower:
            evaluation["hallucinated"] = 0
        else:
            evaluation["hallucinated"] = 1

    # Determinism test
    if test_type == "determinism":
        hashes: List[str] = []
        for _ in range(DETERMINISM_RUNS):
            run = call_model(model, prompt, DEFAULT_PARAMS)
            hashes.append(hash_output(run["output"]))

        identical = len(set(hashes)) == 1
        evaluation["determinism_score"] = int(identical)

    return evaluation


# ==============================
# Main Runner
# ==============================

def run(models: List[str], eval_suite_path: str):
    suite = load_eval_suite(eval_suite_path)
    tests = suite["tests"]

    results: List[Dict[str, Any]] = []

    for model in models:
        print(f"\nRunning model: {model}")
        for test in tests:
            print(f"  → Test: {test['id']}")
            try:
                eval_result = evaluate_test(model, test)
                results.append(eval_result)
            except Exception as e:
                print(f"    ERROR: {e}")

    save_results(results)


def save_results(results: List[Dict[str, Any]]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/eval_results_{timestamp}.csv"

    fieldnames = [
        "model",
        "test_id",
        "type",
        "latency_sec",
        "tokens",
        "tokens_per_sec",
        "json_valid",
        "hallucinated",
        "determinism_score",
    ]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {filename}")


# ==============================
# Entry Point
# ==============================

# LOCALLY AVAILABLE MODELS
###########################
# mistral:7b
# qwen2.5-coder:7b 
# llava:7b
# qwen3-embedding:4b
# qwen3:4b
# qwen3:0.6b    
# deepseek-r1:1.5b



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation suite.")
    parser.add_argument(
        "--suite",
        type=str,
        default="eval_suite.yaml",
        help="Path to the evaluation suite YAML file (default: eval_suite.yaml)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="deepseek-r1:1.5b,qwen3:0.6b",
        help="Comma-separated list of models to test (default: 'deepseek-r1:1.5b,qwen3:0.6b')",
    )
    args = parser.parse_args()

    models_to_test = [m.strip() for m in args.models.split(",") if m.strip()]

    run(models_to_test, args.suite)
