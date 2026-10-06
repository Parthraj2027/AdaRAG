import json
from pathlib import Path
import numpy as np
from src.analyzer import QueryFeatures
from src.normalizer import FeatureNormalizer
from src.constants import LOW_THRESHOLD, MEDIUM_THRESHOLD, TOP_K_LOW, TOP_K_MEDIUM, TOP_K_HIGH

class BudgetSelector:
    def __init__(self, config_path: Path | str = None, stats_path: Path | str = None):
        self.weights = {
            "lexical": 0.15,
            "entity": 0.20,
            "reasoning": 0.25,
            "constraint": 0.20,
            "multihop": 0.20
        }
        self.low_threshold = LOW_THRESHOLD
        self.medium_threshold = MEDIUM_THRESHOLD
        self.threshold_strategy = "fixed"
        self.low_percentile = 33
        self.medium_percentile = 67
        self.normalizer = FeatureNormalizer(stats_path=stats_path)
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: Path | str) -> None:
        path = Path(config_path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.weights.update(config.get("weights", {}))
        thresholds = config.get("thresholds", {})
        if isinstance(thresholds, dict):
            self.low_threshold = thresholds.get("low", self.low_threshold)
            self.medium_threshold = thresholds.get("medium", self.medium_threshold)
            self.threshold_strategy = thresholds.get("strategy", self.threshold_strategy)
            self.low_percentile = thresholds.get("low_percentile", self.low_percentile)
            self.medium_percentile = thresholds.get("medium_percentile", self.medium_percentile)
        elif isinstance(thresholds, list) and len(thresholds) >= 2:
            self.low_threshold = thresholds[0]
            self.medium_threshold = thresholds[1]

    def estimate_score(self, features: QueryFeatures, ablated: list[str] = None) -> float:
        w = self.weights.copy()
        if ablated:
            for cat in ablated:
                if cat in w:
                    w[cat] = 0.0
            total = sum(w.values())
            if total > 0:
                w = {k: v / total for k, v in w.items()}

        token_count = features.token_count
        average_word_length = features.average_word_length
        entity_score = (
            features.entity_count +
            features.person_count * 0.5 +
            features.organization_count * 0.5 +
            features.location_count * 0.5 +
            features.date_count * 0.5
        )
        reasoning = features.reasoning
        constraint_count = features.constraint_count
        multihop = (
            features.multihop +
            features.comparison +
            min(features.entity_count, 3) / 3.0
        )

        T_val = 0.5 * self.normalizer.normalize(token_count, "token_count") + 0.5 * self.normalizer.normalize(average_word_length, "average_word_length")
        E_val = self.normalizer.normalize(entity_score, "entity_score")
        R_val = self.normalizer.normalize(reasoning, "reasoning")
        C_val = self.normalizer.normalize(constraint_count, "constraint_count")
        M_val = self.normalizer.normalize(multihop, "multihop")

        rds = 10.0 * (w["lexical"] * T_val + w["entity"] * E_val + w["reasoning"] * R_val + w["constraint"] * C_val + w["multihop"] * M_val)
        return round(rds, 2)

    def percentile_thresholds(self, scores: list[float]) -> tuple[float, float]:
        if not scores:
            return self.low_threshold, self.medium_threshold
        low = float(np.percentile(scores, self.low_percentile))
        medium = float(np.percentile(scores, self.medium_percentile))
        if low >= medium:
            return self.low_threshold, self.medium_threshold
        return low, medium

    def select_budget(
        self,
        score: float,
        query_features: QueryFeatures = None,
        min_budget: int = 5,
        thresholds: tuple[float, float] = None,
    ) -> int:
        low_threshold, medium_threshold = thresholds or (self.low_threshold, self.medium_threshold)

        if score <= low_threshold:
            base_budget = TOP_K_LOW
        elif score <= medium_threshold:
            base_budget = TOP_K_MEDIUM
        else:
            base_budget = TOP_K_HIGH

        if query_features and (query_features.multihop > 0 or query_features.comparison > 0):
            return max(base_budget, min_budget)
        return max(base_budget, min_budget)
