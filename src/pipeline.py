from src.analyzer import QueryAnalyzer
from src.budget import BudgetSelector
from src.retriever import Retriever
from src.llm import LLM

class RAGPipeline:
    def __init__(self):
        self.analyzer = QueryAnalyzer()
        self.budget_selector = BudgetSelector()
        self.retriever = Retriever()
        self.retriever.load()
        self.llm = LLM()

    def build_prompt(self, query: str, documents: list[str]) -> str:
        context = "\n\n".join(
            f"[Document {i+1}]\n{doc}"
            for i, doc in enumerate(documents)
        )
        return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    def run(self, query: str, mode: str = "adaptive", fixed_k: int = 5, ablated: list[str] = None) -> dict:
        features = self.analyzer.analyze(query)
        score = self.budget_selector.estimate_score(features, ablated=ablated)
        if mode == "adaptive":
            top_k = self.budget_selector.select_budget(score)
        else:
            top_k = fixed_k

        retrieved = self.retriever.search(query, top_k=top_k)
        documents = [doc for doc, _ in retrieved]
        scores = [score for _, score in retrieved]

        prompt = self.build_prompt(query, documents)
        answer, p_tokens, c_tokens = self.llm.generate(prompt)

        return {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "demand_score": score,
            "documents": documents,
            "scores": scores,
            "answer": answer,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens
        }
