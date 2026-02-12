import csv
from collections import defaultdict
from typing import Dict, List, Any, DefaultDict
import statistics
import argparse


# ==============================
# Scoring Weights (Editable)
# ==============================

WEIGHTS = {
    "hallucination_resistance": 0.30,
    "determinism": 0.20,
    "json_validity": 0.15,
    "latency": 0.15,
    "tokens_per_sec": 0.10,
    "token_efficiency": 0.10,
}


# ==============================
# Load Results
# ==============================

def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ==============================
# Aggregation Per Model
# ==============================

def aggregate_by_model(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    models: DefaultDict[str, Dict[str, List[float]]] = defaultdict(lambda: {
        "hallucinated": [],
        "determinism_score": [],
        "json_valid": [],
        "latency_sec": [],
        "tokens_per_sec": [],
        "tokens": [],
    })

    for row in rows:
        model = row["model"]

        if row["hallucinated"]:
            models[model]["hallucinated"].append(int(row["hallucinated"]))

        if row["determinism_score"]:
            models[model]["determinism_score"].append(int(row["determinism_score"]))

        if row["json_valid"]:
            models[model]["json_valid"].append(int(row["json_valid"]))

        models[model]["latency_sec"].append(float(row["latency_sec"]))
        models[model]["tokens_per_sec"].append(float(row["tokens_per_sec"]))
        models[model]["tokens"].append(float(row["tokens"]))

    return models


# ==============================
# Normalization Helpers
# ==============================

def normalize_inverse(value: float, min_val: float, max_val: float) -> float:
    """
    Lower is better (e.g., latency)
    """
    if max_val == min_val:
        return 1.0
    return 1 - ((value - min_val) / (max_val - min_val))


def normalize_direct(value: float, min_val: float, max_val: float) -> float:
    """
    Higher is better
    """
    if max_val == min_val:
        return 1.0
    return (value - min_val) / (max_val - min_val)


# ==============================
# Compute Scores
# ==============================

def compute_scores(aggregated: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:

    summary: Dict[str, Dict[str, float]] = {}

    # Collect global min/max for normalization
    avg_latency: List[float] = []
    avg_tps: List[float] = []
    avg_tokens: List[float] = []

    for model, data in aggregated.items():
        avg_latency.append(statistics.mean(data["latency_sec"]))
        avg_tps.append(statistics.mean(data["tokens_per_sec"]))
        avg_tokens.append(statistics.mean(data["tokens"]))

    min_latency, max_latency = min(avg_latency), max(avg_latency)
    min_tps, max_tps = min(avg_tps), max(avg_tps)
    min_tokens, max_tokens = min(avg_tokens), max(avg_tokens)

    for model, data in aggregated.items():

        hallucination_resistance = 1 - statistics.mean(data["hallucinated"]) if data["hallucinated"] else 1
        determinism = statistics.mean(data["determinism_score"]) if data["determinism_score"] else 1
        json_validity = statistics.mean(data["json_valid"]) if data["json_valid"] else 1

        avg_lat = statistics.mean(data["latency_sec"])
        avg_tps_model = statistics.mean(data["tokens_per_sec"])
        avg_tokens_model = statistics.mean(data["tokens"])

        latency_score = normalize_inverse(avg_lat, min_latency, max_latency)
        tps_score = normalize_direct(avg_tps_model, min_tps, max_tps)
        efficiency_score = normalize_inverse(avg_tokens_model, min_tokens, max_tokens)

        final_score = (
            WEIGHTS["hallucination_resistance"] * hallucination_resistance +
            WEIGHTS["determinism"] * determinism +
            WEIGHTS["json_validity"] * json_validity +
            WEIGHTS["latency"] * latency_score +
            WEIGHTS["tokens_per_sec"] * tps_score +
            WEIGHTS["token_efficiency"] * efficiency_score
        )

        summary[model] = {
            "final_score": round(final_score, 4),
            "hallucination_resistance": round(hallucination_resistance, 3),
            "determinism": round(determinism, 3),
            "json_validity": round(json_validity, 3),
            "latency_score": round(latency_score, 3),
            "tokens_per_sec_score": round(tps_score, 3),
            "efficiency_score": round(efficiency_score, 3),
        }

    return summary


# ==============================
# Ranking Output
# ==============================

def rank_models(summary: Dict[str, Dict[str, float]]) -> None:

    ranked = sorted(summary.items(), key=lambda x: x[1]["final_score"], reverse=True)

    print("\n===== MODEL RANKING =====\n")
    print("Legend:")
    print("  Final Score: Weighted sum of all metrics (range: 0–1, higher is better)")
    print("  Hallucination Resistance: Fraction of non-hallucinated responses (range: 0–1, 1 = best)")
    print("  Determinism: Fraction of identical outputs for repeated runs (range: 0–1, 1 = best)")
    print("  JSON Validity: Fraction of valid JSON outputs (range: 0–1, 1 = best)")
    print("  Latency Score: Normalized (range: 0–1, lower latency = higher score, 1 = best)")
    print("  TPS Score: Normalized tokens per second (range: 0–1, higher = better, 1 = best)")
    print("  Efficiency Score: Normalized token usage (range: 0–1, lower = better, 1 = best)")
    print("")
    for rank, (model, metrics) in enumerate(ranked, start=1):
        print(f"{rank}. {model}")
        print(f"   Final Score: {metrics['final_score']}")
        print(f"   Hallucination Resistance: {metrics['hallucination_resistance']}")
        print(f"   Determinism: {metrics['determinism']}")
        print(f"   JSON Validity: {metrics['json_validity']}")
        print(f"   Latency Score: {metrics['latency_score']}")
        print(f"   TPS Score: {metrics['tokens_per_sec_score']}")
        print(f"   Efficiency Score: {metrics['efficiency_score']}")
        print("")


# ==============================
# Entry Point
# ==============================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate and rank models from results CSV.")
    parser.add_argument(
        "--file",
        type=str,
        default="results/eval_results_latest.csv",
        help="Path to the results CSV file (default: results/eval_results_latest.csv)",
    )
    args = parser.parse_args()

    results_file = args.file
    rows = load_results(results_file)
    aggregated = aggregate_by_model(rows)
    summary = compute_scores(aggregated)
    rank_models(summary)
