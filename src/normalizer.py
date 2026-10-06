import json
from pathlib import Path


class FeatureNormalizer:
    def __init__(self, stats: dict = None, stats_path: Path | str = None):
        self.stats = stats or self.default_stats()
        if stats_path:
            path = Path(stats_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self.stats.update(json.load(f))

    @staticmethod
    def default_stats() -> dict:
        return {
            "token_count": {"min": 3, "max": 45},
            "average_word_length": {"min": 3.0, "max": 10.0},
            "entity_count": {"min": 0, "max": 12},
            "entity_score": {"min": 0.0, "max": 12.0},
            "reasoning": {"min": 0, "max": 5},
            "constraint_count": {"min": 0, "max": 4},
            "multihop": {"min": 0, "max": 3},
        }

    def normalize(self, value: float, feature_name: str) -> float:
        s = self.stats[feature_name]
        if s["max"] == s["min"]:
            return 0.0
        return max(0.0, min(1.0, (value - s["min"]) / (s["max"] - s["min"])))
