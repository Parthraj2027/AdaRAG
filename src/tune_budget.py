import argparse
import itertools
import json
from pathlib import Path
from config import DATA_DIR, EVALUATION_DIR
from src.evaluator import Evaluator
from src.pipeline import RAGPipeline
from src.utils import ensure_directory, logger, set_seed


def grid_search(val_size: int = 200, sample_size: int = 1000, output_path: Path = None) -> dict:
    set_seed(42)
    logger.info("Initializing Budget Parameter Tuning...")
    evaluator = Evaluator(sample_size=sample_size)
    if not evaluator.dataset:
        raise ValueError("No dataset available for tuning.")

    evaluator.dataset = evaluator.dataset[:val_size]
    stats_path = Path(EVALUATION_DIR) / "feature_stats.json"
    evaluator.pipeline = RAGPipeline(feature_stats_path=stats_path)
    logger.info(f"Tuning weights and thresholds on validation set of size {len(evaluator.dataset)}...")

    best_f1 = -1.0
    best_params = None
    weight_options = [0.1, 0.15, 0.2, 0.25, 0.3]
    threshold_options = [(2, 6), (3, 7), (4, 8), (3, 8), (4, 9)]

    for combination in itertools.product(weight_options, repeat=5):
        if abs(sum(combination) - 1.0) > 1e-5:
            continue

        weights = {
            "lexical": combination[0],
            "entity": combination[1],
            "reasoning": combination[2],
            "constraint": combination[3],
            "multihop": combination[4],
        }

        for low_th, med_th in threshold_options:
            evaluator.pipeline.budget_selector.weights = weights.copy()
            evaluator.pipeline.budget_selector.low_threshold = low_th
            evaluator.pipeline.budget_selector.medium_threshold = med_th

            results = evaluator.evaluate_run(mode="adaptive")
            f1 = results["avg_f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_params = {
                    "weights": weights.copy(),
                    "thresholds": {"low": low_th, "medium": med_th},
                    "avg_f1": f1,
                    "avg_recall": results["avg_recall"],
                    "avg_hit_rate": results["avg_hit_rate"],
                    "avg_latency": results["avg_latency"],
                    "validation_size": len(evaluator.dataset),
                }

    if best_params is None:
        raise RuntimeError("Grid search did not evaluate any valid weight combination.")

    output_path = output_path or Path(DATA_DIR) / "best_budget_config.json"
    ensure_directory(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print("\n======================================================================")
    print("AdaRAG Parameter Tuning Complete")
    print("======================================================================")
    print(f"Best Validation F1: {best_params['avg_f1']:.4f}")
    print(f"Best Validation Recall: {best_params['avg_recall']:.4f}")
    print(f"Best Validation Hit Rate: {best_params['avg_hit_rate']:.4f}")
    print(f"Best Latency: {best_params['avg_latency']:.4f}s")
    print(f"Wrote best config to: {output_path}")
    print("======================================================================")
    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune AdaRAG weights and thresholds.")
    parser.add_argument("--val_size", type=int, default=200, help="Held-out validation query count.")
    parser.add_argument("--sample_size", type=int, default=1000, help="HotpotQA sample size to prepare/load.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DATA_DIR) / "best_budget_config.json",
        help="Path to write the selected budget config."
    )
    args = parser.parse_args()
    grid_search(val_size=args.val_size, sample_size=args.sample_size, output_path=args.output)
