import time
from pathlib import Path
from src.analyzer import QueryAnalyzer
from src.budget import BudgetSelector
from src.retriever import Retriever
from config import LOCAL_LLM_MODEL, USE_LOCAL_LLM, USE_MOCK_LLM
from src.llm import LLM, MockLLM

class RAGPipeline:
    def __init__(
        self,
        budget_mode: str = "heuristic",
        budget_config_path: Path | str = None,
        feature_stats_path: Path | str = None,
    ):
        self.budget_mode = budget_mode
        self.analyzer = QueryAnalyzer()
        if budget_mode == "mlp":
            from src.budget_mlp import MLPBudgetSelector
            self.budget_selector = MLPBudgetSelector(stats_path=feature_stats_path)
        else:
            self.budget_selector = BudgetSelector(
                config_path=budget_config_path,
                stats_path=feature_stats_path
            )
        self.retriever = Retriever()
        self.retriever.load()
        if USE_MOCK_LLM:
            self.llm = MockLLM()
        elif USE_LOCAL_LLM:
            from src.llm_local import LocalLLM
            self.llm = LocalLLM(model=LOCAL_LLM_MODEL)
        else:
            self.llm = LLM()

    def build_prompt(self, query: str, documents: list[str]) -> str:
        context = "\n\n".join(
            f"[Document {i+1}]\n{doc}"
            for i, doc in enumerate(documents)
        )
        return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    def run(
        self,
        query: str,
        mode: str = "adaptive",
        fixed_k: int = 5,
        ablated: list[str] = None,
        query_type: str = None,
        budget_thresholds: tuple[float, float] = None,
    ) -> dict:
        t_start = time.perf_counter()
        features = self.analyzer.analyze(query)
        rds_score = self.budget_selector.estimate_score(features, ablated=ablated)

        if mode in ("adaptive", "adaptive_mlp"):
            top_k = self.budget_selector.select_budget(
                rds_score,
                query_features=features,
                thresholds=budget_thresholds,
            )
        elif mode == "oracle":
            q_type = query_type.lower() if query_type else ""
            if "comparison" in q_type or features.comparison > 0:
                top_k = 12
            elif "bridge" in q_type or features.multihop > 0:
                top_k = 8
            else:
                top_k = 5
        else:
            top_k = fixed_k

        t_ret_start = time.perf_counter()
        retrieved = self.retriever.search(query, top_k=top_k)
        retrieval_latency = time.perf_counter() - t_ret_start

        documents = [doc for doc, _ in retrieved]
        retrieval_scores = [sim for _, sim in retrieved]

        prompt = self.build_prompt(query, documents)

        t_gen_start = time.perf_counter()
        answer, p_tokens, c_tokens = self.llm.generate(prompt)
        generation_latency = time.perf_counter() - t_gen_start

        total_latency = time.perf_counter() - t_start

        return {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "demand_score": rds_score,
            "documents": documents,
            "retrieval_scores": retrieval_scores,
            "answer": answer,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
            "latency": total_latency
        }
