import torch
import torch.nn as nn
from pathlib import Path
from src.analyzer import QueryFeatures
from src.budget import BudgetSelector


class RDSNet(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, num_buckets: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_buckets),
        )

    def forward(self, x):
        return self.net(x)


class BudgetSelectorMLP(BudgetSelector):
    def __init__(
        self,
        model_path: Path | str = Path("models") / "rds_mlp.pt",
        config_path: Path | str = None,
        stats_path: Path | str = None,
    ):
        super().__init__(config_path=config_path, stats_path=stats_path)
        self.buckets = [5, 8, 12]
        self.model = RDSNet()
        self.use_mlp = False

        model_file = Path(model_path)
        if model_file.exists():
            self.model.load_state_dict(torch.load(model_file, map_location="cpu"))
            self.model.eval()
            self.use_mlp = True

    def _to_vector(self, features: QueryFeatures) -> list[float]:
        token_count = features.token_count
        average_word_length = features.average_word_length
        entity_score = (
            features.entity_count
            + features.person_count * 0.5
            + features.organization_count * 0.5
            + features.location_count * 0.5
            + features.date_count * 0.5
        )
        reasoning = features.reasoning
        constraint_count = features.constraint_count
        multihop = (
            features.multihop
            + features.comparison
            + min(features.entity_count, 3) / 3.0
        )

        t_val = 0.5 * self.normalizer.normalize(token_count, "token_count") + 0.5 * self.normalizer.normalize(
            average_word_length, "average_word_length"
        )
        e_val = self.normalizer.normalize(entity_score, "entity_score")
        r_val = self.normalizer.normalize(reasoning, "reasoning")
        c_val = self.normalizer.normalize(constraint_count, "constraint_count")
        m_val = self.normalizer.normalize(multihop, "multihop")
        return [t_val, e_val, r_val, c_val, m_val]

    def estimate_score(self, features: QueryFeatures, ablated: list[str] = None) -> float:
        if not self.use_mlp:
            return super().estimate_score(features, ablated=ablated)

        x = torch.tensor([self._to_vector(features)], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            score = (probs[0] * torch.tensor([2.5, 5.5, 9.5])).sum().item()
        return round(score, 2)

    def select_budget(self, score: float, query_features: QueryFeatures = None, min_budget: int = 5) -> int:
        if not self.use_mlp:
            return super().select_budget(score, query_features=query_features, min_budget=min_budget)

        if score < 4.0:
            budget = self.buckets[0]
        elif score < 7.5:
            budget = self.buckets[1]
        else:
            budget = self.buckets[2]

        if query_features and (query_features.multihop > 0 or query_features.comparison > 0):
            return max(budget, min_budget)
        return max(budget, min_budget)
