# Benchy: Model Evaluation Suite

## Project Goal

Benchy is designed to benchmark and compare the performance of different language models on a suite of evaluation tasks. It provides a reproducible framework to:
- Run multiple models on a set of prompts/tests
- Collect and aggregate results (latency, accuracy, determinism, hallucination, etc.)
- Score and rank models using customizable metrics
- Output results in a structured, analyzable format

## Project Structure

- `runner.py` — Main script to run models on the evaluation suite and collect results
- `metrics.py` — Aggregates, scores, and ranks models based on results CSV
- `eval_suite.yaml` — YAML file describing the evaluation tests/prompts
- `models_config.yaml` — (Optional) Configuration for available models
- `reporters.py` — (Optional) Custom reporting utilities
- `requirements.txt` — Python dependencies
- `results/` — Output directory for generated results CSV files

## How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Evaluation Suite
- Edit `eval_suite.yaml` to define your tests/prompts.
- (Optional) Edit `models_config.yaml` to configure models.

### 3. Run Model Evaluations
```bash
python3 runner.py --suite eval_suite.yaml --models "model1,model2,model3"
```
- `--suite` (optional): Path to the evaluation suite YAML file (default: `eval_suite.yaml`)
- `--models` (optional): Comma-separated list of models to test (default: `deepseek-r1:1.5b,qwen3:0.6b`)

Results are saved in the `results/` directory as timestamped CSV files.

### 4. Score and Rank Models
```bash
python3 metrics.py --file results/eval_results_<timestamp>.csv
```
- `--file` (optional): Path to the results CSV file (default: `results/eval_results_latest.csv`)

The script prints a ranking of models with a legend explaining each metric.

## Metrics Explained
- **Final Score**: Weighted sum of all metrics (range: 0–1, higher is better)
- **Hallucination Resistance**: Fraction of non-hallucinated responses (range: 0–1, 1 = best)
- **Determinism**: Fraction of identical outputs for repeated runs (range: 0–1, 1 = best)
- **JSON Validity**: Fraction of valid JSON outputs (range: 0–1, 1 = best)
- **Latency Score**: Normalized (range: 0–1, lower latency = higher score, 1 = best)
- **TPS Score**: Normalized tokens per second (range: 0–1, higher = better, 1 = best)
- **Efficiency Score**: Normalized token usage (range: 0–1, lower = better, 1 = best)

## Customization
- Adjust metric weights in `metrics.py` (`WEIGHTS` dictionary)
- Add or modify evaluation tests in `eval_suite.yaml`
- Extend reporting or add new metrics in `metrics.py` or `reporters.py`
