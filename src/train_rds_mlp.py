"""
Train a compact MLP to map query features to retrieval budget buckets.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import EVALUATION_DIR, MODELS_DIR
from src.analyzer import QueryAnalyzer
from src.budget_mlp import RDSNet
from src.normalizer import FeatureNormalizer
from src.utils import ensure_directory, logger


class BudgetDataset(Dataset):
    def __init__(self, queries: list[dict], normalizer: FeatureNormalizer):
        self.samples = []
        self.analyzer = QueryAnalyzer()
        self.normalizer = normalizer

        for item in queries:
            features = self.analyzer.analyze(item["query"])
            vec = self._feature_vector(features)
            label = self._label_from_query_type(item.get("type", "bridge"))
            self.samples.append((vec, label))

    def _feature_vector(self, features):
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

    @staticmethod
    def _label_from_query_type(q_type: str) -> int:
        mapping = {
            "factoid": 0,
            "bridge": 1,
            "comparison": 2,
        }
        return mapping.get(q_type, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train() -> None:
    ensure_directory(MODELS_DIR)

    eval_path = Path(EVALUATION_DIR) / "hotpotqa_eval.json"
    if not eval_path.exists():
        logger.error("Evaluation set not found. Run download mode first.")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    if not eval_data:
        logger.error("Evaluation set is empty.")
        return

    stats_path = Path(EVALUATION_DIR) / "feature_stats.json"
    normalizer = FeatureNormalizer(stats_path=stats_path if stats_path.exists() else None)

    dataset = BudgetDataset(eval_data[: min(1000, len(eval_data))], normalizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = RDSNet(input_dim=5, hidden_dim=32, num_buckets=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(len(loader), 1)
            acc = correct / max(total, 1)
            logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={acc:.3f}")

    model_path = Path(MODELS_DIR) / "rds_mlp.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
