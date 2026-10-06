import json
import pickle
from pathlib import Path
from statistics import mean

from config import (
    CORPUS_DIR,
    EVALUATION_DIR,
    GROQ_API_KEY,
    GROQ_COMPLETION_PRICE_PER_1M,
    GROQ_PROMPT_PRICE_PER_1M,
    LLM_MODEL,
    MODELS_DIR,
)
from src.analyzer import QueryAnalyzer
from src.budget import BudgetSelector


def _load_json(path: Path) -> list[dict] | list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_documents() -> list[str]:
    doc_path = MODELS_DIR / "documents.pkl"
    if doc_path.exists():
        with open(doc_path, "rb") as f:
            return pickle.load(f)

    corpus_path = CORPUS_DIR / "hotpot_corpus.json"
    if corpus_path.exists():
        return _load_json(corpus_path)

    corpus_data_path = CORPUS_DIR / "corpus_data.json"
    if corpus_data_path.exists():
        data = _load_json(corpus_data_path)
        if data and isinstance(data[0], dict):
            return [item.get("text", "") for item in data]
        return data

    return []


def run_preflight(
    sample_size: int,
    latency_runs: int = 1,
    budget_config_path: Path | str = None,
    feature_stats_path: Path | str = None,
) -> dict:
    eval_path = EVALUATION_DIR / "hotpotqa_eval.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {eval_path}")

    dataset = _load_json(eval_path)
    eval_count = min(sample_size, len(dataset))
    estimate_count = sample_size
    latency_runs = max(1, latency_runs)

    analyzer = QueryAnalyzer()
    selector = BudgetSelector(config_path=budget_config_path, stats_path=feature_stats_path)
    scores = []
    features_by_query = []
    for item in dataset[:eval_count]:
        features = analyzer.analyze(item["query"])
        score = selector.estimate_score(features)
        scores.append(score)
        features_by_query.append(features)

    thresholds = None
    if selector.threshold_strategy == "percentile":
        thresholds = selector.percentile_thresholds(scores)

    budgets = [
        selector.select_budget(score, query_features=features, thresholds=thresholds)
        for score, features in zip(scores, features_by_query)
    ]

    documents = _load_documents()
    avg_doc_chars = mean(len(doc) for doc in documents) if documents else 0
    avg_query_chars = mean(len(item["query"]) for item in dataset[:eval_count]) if eval_count else 0
    fixed_budgets = [3, 5, 7, 10, 15]
    method_budgets = fixed_budgets + [mean(budgets) if budgets else 8, 8]
    prompt_overhead_chars = 260
    avg_prompt_tokens = mean(
        (prompt_overhead_chars + avg_query_chars + avg_doc_chars * budget) / 4
        for budget in method_budgets
    )
    calls = estimate_count * len(method_budgets) * latency_runs
    input_tokens = avg_prompt_tokens * calls
    output_tokens = 150 * calls
    cost = (
        input_tokens * GROQ_PROMPT_PRICE_PER_1M
        + output_tokens * GROQ_COMPLETION_PRICE_PER_1M
    ) / 1_000_000

    report = {
        "groq_api_key_present": bool(GROQ_API_KEY),
        "configured_model": LLM_MODEL,
        "requested_sample_size": sample_size,
        "available_questions": len(dataset),
        "planned_questions": eval_count,
        "estimated_questions": estimate_count,
        "publication_ready_sample": len(dataset) >= 500 and sample_size >= 500,
        "latency_runs": latency_runs,
        "threshold_strategy": selector.threshold_strategy,
        "thresholds": (
            round(thresholds[0], 2),
            round(thresholds[1], 2),
        ) if thresholds else (selector.low_threshold, selector.medium_threshold),
        "rds_min": round(min(scores), 2) if scores else 0,
        "rds_max": round(max(scores), 2) if scores else 0,
        "rds_mean": round(mean(scores), 2) if scores else 0,
        "bucket_counts": {k: budgets.count(k) for k in [5, 8, 12]},
        "estimated_calls": calls,
        "estimated_input_tokens": round(input_tokens),
        "estimated_output_tokens": round(output_tokens),
        "estimated_cost_usd": round(cost, 4),
    }

    print("\n--- GROQ PREFLIGHT ---")
    for key, value in report.items():
        print(f"{key}: {value}")
    print("account_credit_check: not available through the OpenAI-compatible Groq API")
    return report
