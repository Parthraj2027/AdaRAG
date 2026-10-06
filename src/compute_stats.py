import argparse
import json
from pathlib import Path
from config import EVALUATION_DIR
from src.analyzer import QueryAnalyzer
from src.normalizer import FeatureNormalizer
from src.utils import ensure_directory, logger, set_seed


def feature_values(features) -> dict:
    entity_score = (
        features.entity_count +
        features.person_count * 0.5 +
        features.organization_count * 0.5 +
        features.location_count * 0.5 +
        features.date_count * 0.5
    )
    multihop = (
        features.multihop +
        features.comparison +
        min(features.entity_count, 3) / 3.0
    )
    return {
        "token_count": features.token_count,
        "average_word_length": features.average_word_length,
        "entity_count": features.entity_count,
        "entity_score": entity_score,
        "reasoning": features.reasoning,
        "constraint_count": features.constraint_count,
        "multihop": multihop,
    }


def compute_stats(dataset_path: Path, output_path: Path, limit: int = 500) -> dict:
    set_seed(42)
    analyzer = QueryAnalyzer()
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    rows = []
    for item in dataset[:limit]:
        query = item.get("query") or item.get("question")
        if query:
            rows.append(feature_values(analyzer.analyze(query)))

    if not rows:
        raise ValueError("No query rows found for feature-stat computation.")

    stats = {}
    for feature_name in FeatureNormalizer.default_stats():
        values = [row[feature_name] for row in rows]
        stats[feature_name] = {"min": min(values), "max": max(values)}

    ensure_directory(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Wrote feature stats for {len(rows)} queries to {output_path}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute min/max query feature stats for AdaRAG.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(EVALUATION_DIR) / "hotpotqa_eval.json",
        help="Evaluation JSON containing query fields."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(EVALUATION_DIR) / "feature_stats.json",
        help="Path to write feature_stats.json."
    )
    parser.add_argument("--limit", type=int, default=500, help="Number of queries to use.")
    args = parser.parse_args()
    compute_stats(args.input, args.output, limit=args.limit)
