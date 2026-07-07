from src.analyzer import QueryFeatures
from src.constants import LOW_THRESHOLD, MEDIUM_THRESHOLD, TOP_K_LOW, TOP_K_MEDIUM, TOP_K_HIGH

class BudgetSelector:
    def __init__(self):
        self.weights = {
            "lexical": 0.15,
            "entity": 0.20,
            "reasoning": 0.25,
            "constraint": 0.20,
            "multihop": 0.20
        }

    def estimate_score(self, features: QueryFeatures, ablated: list[str] = None) -> float:
        w = self.weights.copy()
        if ablated:
            for cat in ablated:
                if cat in w:
                    w[cat] = 0.0
            total = sum(w.values())
            if total > 0:
                w = {k: v / total for k, v in w.items()}

        T_val = 0.5 * min(features.token_count / 30.0, 1.0) + 0.5 * min(features.average_word_length / 10.0, 1.0)
        E_val = min((features.entity_count + features.person_count * 0.5 + features.organization_count * 0.5 + features.location_count * 0.5 + features.date_count * 0.5) / 5.0, 1.0)
        R_val = min(features.reasoning / 3.0, 1.0)
        C_val = min(features.constraint_count / 3.0, 1.0)
        M_val = min((features.comparison * 1.5 + features.multihop * 2.0 + features.listing * 1.0 + features.boolean * 0.5) / 3.0, 1.0)

        rds = 10.0 * (w["lexical"] * T_val + w["entity"] * E_val + w["reasoning"] * R_val + w["constraint"] * C_val + w["multihop"] * M_val)
        return round(rds, 2)

    def select_budget(self, score: float) -> int:
        if score <= LOW_THRESHOLD:
            return TOP_K_LOW
        elif score <= MEDIUM_THRESHOLD:
            return TOP_K_MEDIUM
        else:
            return TOP_K_HIGH
