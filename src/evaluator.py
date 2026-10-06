"""
Publication-ready evaluator for AdaRAG.
Supports both retrieval-only (free) and full LLM evaluation modes.
"""
import json
import random
import string
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from config import EVALUATION_DIR, RESULTS_DIR, CORPUS_DIR, MODELS_DIR
from src.pipeline import RAGPipeline
from src.utils import logger, ensure_directory


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold: str, a_pred: str) -> float:
    return float(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


class Evaluator:
    def __init__(self):
        self.pipeline: Optional[RAGPipeline] = None
        self.dataset = self.load_dataset()

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def load_dataset(self, require_official: bool = True) -> list[dict]:
        path = Path(EVALUATION_DIR) / "hotpotqa_eval.json"
        if not path.exists():
            if require_official:
                raise FileNotFoundError(
                    f"Official evaluation dataset not found at {path}.\n"
                    f"Run: python run.py --mode download --sample_size 1000\n"
                    f"Do NOT use local benchmark for publication experiments."
                )
            else:
                logger.warning("FALLBACK: Using local benchmark (NOT FOR PUBLICATION)")
                return self._load_local_fallback()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if len(data) < 100:
            raise ValueError(
                f"Dataset has only {len(data)} questions. "
                f"Publication requires >= 500. Re-download with --sample_size 1000."
            )

        for i, item in enumerate(data[:3]):
            if "supporting_facts" not in item:
                raise ValueError(
                    f"Question {i} missing 'supporting_facts'. Invalid dataset format."
                )

        logger.info(f"Loaded official dataset: {len(data)} questions")
        return data

    def _load_local_fallback(self) -> list[dict]:
        """Emergency fallback – never use for paper numbers."""
        return [
            {
                "query": "Who developed FAISS and which group does it belong to?",
                "answer": "FAISS was developed by Meta.",
                "complexity": "easy",
                "supporting_facts": []
            },
        ]

    def download_and_prepare_hotpotqa(self, sample_size: int = 1000) -> None:
        logger.info(f"Downloading official HotpotQA dev dataset ({sample_size} samples)...")
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(
                f"Failed to download HotpotQA: {e}\n"
                f"This is required for publication experiments."
            ) from e

        all_paragraphs = {}
        qa_subset = []

        for item in data[:sample_size]:
            qa_subset.append({
                "query": item["question"],
                "answer": item["answer"],
                "type": item.get("type", "unknown"),
                "level": item.get("level", "unknown"),
                "supporting_facts": item.get("supporting_facts", [])
            })
            for context_item in item.get("context", []):
                title = context_item[0]
                sentences = context_item[1]
                full_text = " ".join(sentences)
                all_paragraphs[title] = f"{title}: {full_text}"

        corpus_chunks = list(all_paragraphs.values())
        ensure_directory(EVALUATION_DIR)
        ensure_directory(CORPUS_DIR)

        with open(Path(EVALUATION_DIR) / "hotpotqa_eval.json", "w", encoding="utf-8") as f:
            json.dump(qa_subset, f, indent=2)

        with open(Path(CORPUS_DIR) / "hotpot_corpus.json", "w", encoding="utf-8") as f:
            json.dump(corpus_chunks, f, indent=2)

        logger.info(f"Saved {len(qa_subset)} questions and {len(corpus_chunks)} unique paragraphs.")
        self.dataset = qa_subset

    # ------------------------------------------------------------------
    # Core evaluation helpers
    # ------------------------------------------------------------------
    def compute_retrieval_metrics(self, retrieved_docs: list[str],
                                  supporting_facts: list) -> tuple[float, float]:
        """Compute recall and hit-rate against gold supporting facts."""
        if not supporting_facts:
            return 0.0, 0.0

        hits = 0
        corpus_path = Path(CORPUS_DIR) / "hotpot_corpus.json"
        corpus_map = {}
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            for para in corpus:
                if ":" in para:
                    title, text = para.split(":", 1)
                    corpus_map[title.strip()] = text.strip()

        for fact in supporting_facts:
            if isinstance(fact, list) and len(fact) == 2:
                title, sent_idx = fact
                gold_text = corpus_map.get(title, "")
            else:
                gold_text = str(fact)

            found = any(gold_text[:80] in doc or doc[:80] in gold_text
                        for doc in retrieved_docs)
            if found:
                hits += 1

        recall = hits / len(supporting_facts) if supporting_facts else 0.0
        hit_rate = 1.0 if hits > 0 else 0.0
        return recall, hit_rate

    def _bootstrap_ci(self, values: list[float], n_bootstrap: int = 1000,
                      ci: float = 0.95) -> tuple[float, float]:
        arr = np.array(values)
        boots = np.random.choice(arr, (n_bootstrap, len(arr)), replace=True)
        means = boots.mean(axis=1)
        lower = np.percentile(means, (1 - ci) / 2 * 100)
        upper = np.percentile(means, (1 + ci) / 2 * 100)
        return float(lower), float(upper)

    def _report_significance(self, baseline: dict, method: dict) -> None:
        base_f1 = [r["f1"] for r in baseline["raw_results"]]
        meth_f1 = [r["f1"] for r in method["raw_results"]]
        base_lat = [r["latency"] for r in baseline["raw_results"]]
        meth_lat = [r["latency"] for r in method["raw_results"]]

        if len(base_f1) < 2 or len(meth_f1) < 2:
            print("Insufficient samples for t-test.")
            return

        t_f1, p_f1 = stats.ttest_rel(base_f1, meth_f1)
        t_lat, p_lat = stats.ttest_rel(base_lat, meth_lat)

        sig_f1 = "***" if p_f1 < 0.001 else "**" if p_f1 < 0.01 else "*" if p_f1 < 0.05 else ""
        sig_lat = "***" if p_lat < 0.001 else "**" if p_lat < 0.01 else "*" if p_lat < 0.05 else ""

        print(f"\n=== Statistical Significance (AdaRAG vs Fixed Top-10) ===")
        print(f"F1:      t={t_f1:.3f}, p={p_f1:.4f} {sig_f1}")
        print(f"Latency: t={t_lat:.3f}, p={p_lat:.4f} {sig_lat}")

    # ------------------------------------------------------------------
    # Full evaluation (retrieval + LLM generation)
    # ------------------------------------------------------------------
    def evaluate_run(self, mode: str, fixed_k: int = 5,
                     ablated: list[str] = None,
                     sample_size: int = None,
                     latency_runs: int = 1) -> dict:
        if self.pipeline is None:
            self.pipeline = RAGPipeline()

        dataset = self.dataset[:sample_size] if sample_size else self.dataset
        if not dataset:
            raise RuntimeError("EVALUATION ABORTED: Dataset is empty.")

        results = []
        total_latency = 0.0
        total_ret_lat = 0.0
        total_gen_lat = 0.0
        total_docs = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for query_index, item in enumerate(dataset, start=1):
            query = item["query"]
            gold = item["answer"]
            gold_facts = item.get("supporting_facts", [])
            logger.info(f"Evaluating query {query_index}/{len(dataset)}")

            # Median latency over multiple runs
            latencies = []
            ret_lats = []
            gen_lats = []
            best_res = None

            for _ in range(latency_runs):
                start = time.perf_counter()
                res = self.pipeline.run(query, mode=mode, fixed_k=fixed_k, ablated=ablated)
                end = time.perf_counter()

                # Approximate retrieval vs generation split
                ret_start = time.perf_counter()
                _ = self.pipeline.retriever.search(query, top_k=res["top_k"])
                ret_lat = time.perf_counter() - ret_start

                latencies.append(end - start)
                ret_lats.append(ret_lat)
                gen_lats.append((end - start) - ret_lat)
                best_res = res

            latency = float(np.median(latencies))
            ret_latency = float(np.median(ret_lats))
            gen_latency = float(np.median(gen_lats))

            em = compute_exact(gold, best_res["answer"])
            f1 = compute_f1(gold, best_res["answer"])
            recall, hit_rate = self.compute_retrieval_metrics(best_res["documents"], gold_facts)

            results.append({
                "query": query,
                "gold": gold,
                "pred": best_res["answer"],
                "em": em,
                "f1": f1,
                "recall": recall,
                "hit_rate": hit_rate,
                "latency": latency,
                "retrieval_latency": ret_latency,
                "generation_latency": gen_latency,
                "top_k": best_res["top_k"],
                "rds_score": best_res.get("rds_score", 0.0),
                "prompt_tokens": best_res["prompt_tokens"],
                "completion_tokens": best_res["completion_tokens"]
            })

            total_latency += latency
            total_ret_lat += ret_latency
            total_gen_lat += gen_latency
            total_docs += best_res["top_k"]
            total_prompt_tokens += best_res["prompt_tokens"]
            total_completion_tokens += best_res["completion_tokens"]

        n = len(results)
        f1_scores = [r["f1"] for r in results]
        lat_scores = [r["latency"] for r in results]
        f1_ci_low, f1_ci_high = self._bootstrap_ci(f1_scores)
        lat_ci_low, lat_ci_high = self._bootstrap_ci(lat_scores)

        return {
            "avg_em": round(sum(r["em"] for r in results) / n, 4),
            "avg_f1": round(sum(r["f1"] for r in results) / n, 4),
            "f1_ci_low": round(f1_ci_low, 4),
            "f1_ci_high": round(f1_ci_high, 4),
            "avg_recall": round(sum(r["recall"] for r in results) / n, 4),
            "avg_hit_rate": round(sum(r["hit_rate"] for r in results) / n, 4),
            "avg_latency": round(total_latency / n, 4),
            "avg_retrieval_latency": round(total_ret_lat / n, 4),
            "avg_generation_latency": round(total_gen_lat / n, 4),
            "latency_ci_low": round(lat_ci_low, 4),
            "latency_ci_high": round(lat_ci_high, 4),
            "avg_docs": round(total_docs / n, 2),
            "avg_tokens": round((total_prompt_tokens + total_completion_tokens) / n, 2),
            "avg_prompt_tokens": round(total_prompt_tokens / n, 2),
            "avg_completion_tokens": round(total_completion_tokens / n, 2),
            "avg_cost": self._compute_cost(total_prompt_tokens / n, total_completion_tokens / n),
            "throughput": round(1.0 / max(total_latency / n, 0.001), 2),
            "raw_results": results
        }

    def _compute_cost(self, prompt_tok: float, compl_tok: float) -> float:
        from config import GROQ_PROMPT_PRICE_PER_1M, GROQ_COMPLETION_PRICE_PER_1M
        return round((prompt_tok * GROQ_PROMPT_PRICE_PER_1M +
                      compl_tok * GROQ_COMPLETION_PRICE_PER_1M) / 1_000_000, 7)

    # ------------------------------------------------------------------
    # Retrieval-only evaluation (FREE – no LLM calls)
    # ------------------------------------------------------------------
    def evaluate_run_retrieval_only(self, mode: str, fixed_k: int = 5,
                                    ablated: list[str] = None,
                                    sample_size: int = None) -> dict:
        if self.pipeline is None:
            self.pipeline = RAGPipeline()

        dataset = self.dataset[:sample_size] if sample_size else self.dataset
        if len(dataset) < 30:
            raise RuntimeError(f"Dataset too small: {len(dataset)} questions.")

        results = []
        total_retrieval_latency = 0.0
        total_feature_latency = 0.0
        total_prompt_tokens = 0

        for item in dataset:
            query = item["query"]
            gold_facts = item.get("supporting_facts", [])

            # Feature extraction + budget selection
            feat_start = time.perf_counter()
            features = self.pipeline.analyzer.analyze(query)
            score = self.pipeline.budget_selector.estimate_score(features, ablated=ablated)
            if mode == "adaptive" or mode == "adaptive_mlp":
                top_k = self.pipeline.budget_selector.select_budget(score)
            else:
                top_k = fixed_k
            feat_time = time.perf_counter() - feat_start

            # Retrieval only
            ret_start = time.perf_counter()
            retrieved = self.pipeline.retriever.search(query, top_k=top_k)
            ret_time = time.perf_counter() - ret_start

            documents = [doc for doc, _ in retrieved]

            # Approximate prompt tokens
            prompt_text = self.pipeline.build_prompt(query, documents)
            prompt_tokens = len(prompt_text) // 4

            recall, hit_rate = self.compute_retrieval_metrics(documents, gold_facts)

            results.append({
                "query": query,
                "top_k": top_k,
                "rds_score": score,
                "recall": recall,
                "hit_rate": hit_rate,
                "retrieval_latency": ret_time,
                "feature_latency": feat_time,
                "prompt_tokens": prompt_tokens,
                "num_docs": len(documents)
            })

            total_retrieval_latency += ret_time
            total_feature_latency += feat_time
            total_prompt_tokens += prompt_tokens

        n = len(results)
        return {
            "avg_em": 0.0,
            "avg_f1": 0.0,
            "avg_recall": round(sum(r["recall"] for r in results) / n, 4),
            "avg_hit_rate": round(sum(r["hit_rate"] for r in results) / n, 4),
            "avg_retrieval_latency": round(total_retrieval_latency / n, 4),
            "avg_feature_latency": round(total_feature_latency / n, 4),
            "avg_latency": round((total_retrieval_latency + total_feature_latency) / n, 4),
            "avg_docs": round(sum(r["num_docs"] for r in results) / n, 2),
            "avg_tokens": round(total_prompt_tokens / n, 2),
            "avg_prompt_tokens": round(total_prompt_tokens / n, 2),
            "avg_cost": 0.0,
            "throughput": round(1.0 / max((total_retrieval_latency + total_feature_latency) / n, 0.001), 2),
            "raw_results": results
        }

    # ------------------------------------------------------------------
    # LLM subset evaluation (cheap – run on small random sample)
    # ------------------------------------------------------------------
    def evaluate_run_llm_subset(self, mode: str, fixed_k: int = None,
                                n: int = 30, latency_runs: int = 1) -> dict:
        if len(self.dataset) < n:
            raise ValueError(f"Dataset has {len(self.dataset)} items, need {n}")
        subset = random.sample(self.dataset, n)
        original_dataset = self.dataset
        self.dataset = subset
        try:
            result = self.evaluate_run(mode=mode, fixed_k=fixed_k,
                                       sample_size=n, latency_runs=latency_runs)
        finally:
            self.dataset = original_dataset
        return result

    # ------------------------------------------------------------------
    # Experiment runners
    # ------------------------------------------------------------------
    def run_experiments(self, sample_size: int = None, latency_runs: int = 1) -> None:
        methods = [
            ("Fixed Top-3", "fixed", 3),
            ("Fixed Top-5", "fixed", 5),
            ("Fixed Top-7", "fixed", 7),
            ("Fixed Top-10", "fixed", 10),
            ("Fixed Top-15", "fixed", 15),
            ("AdaRAG-Heuristic", "adaptive", None),
            ("AdaRAG-MLP", "adaptive_mlp", None),
        ]

        results = {}
        for name, mode, k in methods:
            logger.info(f"Running: {name}")
            try:
                res = self.evaluate_run(mode=mode, fixed_k=k,
                                        sample_size=sample_size,
                                        latency_runs=latency_runs)
                results[name] = res
                logger.info(f"{name} => F1: {res['avg_f1']:.4f}, Latency: {res['avg_latency']:.4f}s")
            except Exception as e:
                logger.error(f"Failed {name}: {e}")
                continue

        if "Fixed Top-10" in results and "AdaRAG-Heuristic" in results:
            self._report_significance(results["Fixed Top-10"], results["AdaRAG-Heuristic"])

        self._save_summary(results, "experiment_results.csv")
        self.plot_pareto(results)
        if "AdaRAG-Heuristic" in results:
            self.plot_budget_distribution(results["AdaRAG-Heuristic"])

    def run_retrieval_experiments(self, sample_size: int = None) -> None:
        """Run full Pareto evaluation without any LLM calls."""
        methods = [
            ("Fixed Top-3", "fixed", 3),
            ("Fixed Top-5", "fixed", 5),
            ("Fixed Top-7", "fixed", 7),
            ("Fixed Top-10", "fixed", 10),
            ("Fixed Top-15", "fixed", 15),
            ("AdaRAG-Heuristic", "adaptive", None),
        ]

        results = {}
        for name, mode, k in methods:
            logger.info(f"Retrieval-only: {name}")
            try:
                res = self.evaluate_run_retrieval_only(mode=mode, fixed_k=k,
                                                       sample_size=sample_size)
                results[name] = res
                print(f"{name}: Recall={res['avg_recall']:.4f}, "
                      f"Docs={res['avg_docs']:.1f}, "
                      f"Tokens={res['avg_tokens']:.0f}, "
                      f"RetLatency={res['avg_retrieval_latency']:.4f}s")
            except Exception as e:
                logger.error(f"Failed {name}: {e}")
                continue

        self._save_summary(results, "retrieval_experiments.csv")
        self.plot_retrieval_pareto(results)
        if "AdaRAG-Heuristic" in results:
            self.plot_budget_distribution(results["AdaRAG-Heuristic"], prefix="retrieval_")

    def run_llm_subset_experiments(self, n: int = 30, latency_runs: int = 1) -> None:
        """Run LLM generation on a small random subset for F1/EM."""
        methods = [
            ("Fixed Top-5", "fixed", 5),
            ("Fixed Top-10", "fixed", 10),
            ("AdaRAG-Heuristic", "adaptive", None),
        ]
        results = {}
        for name, mode, k in methods:
            logger.info(f"LLM subset ({n} queries): {name}")
            try:
                res = self.evaluate_run_llm_subset(mode=mode, fixed_k=k,
                                                   n=n, latency_runs=latency_runs)
                results[name] = res
                print(f"{name}: EM={res['avg_em']:.4f}, F1={res['avg_f1']:.4f}, "
                      f"Latency={res['avg_latency']:.2f}s")
            except Exception as e:
                logger.error(f"Failed {name}: {e}")
                continue

        self._save_summary(results, "llm_subset_experiments.csv")

    def run_ablation_study(self, sample_size: int = None) -> None:
        configs = [
            ("Full AdaRAG", None),
            ("- No Lexical", ["lexical"]),
            ("- No Entity", ["entity"]),
            ("- No Reasoning", ["reasoning"]),
            ("- No Constraint", ["constraint"]),
            ("- No Multihop", ["multihop"]),
        ]
        results = {}
        for name, ablated in configs:
            logger.info(f"Ablation: {name}")
            try:
                res = self.evaluate_run(mode="adaptive", ablated=ablated,
                                        sample_size=sample_size)
                results[name] = res
            except Exception as e:
                logger.error(f"Failed {name}: {e}")
                continue

        rows = []
        for name, res in results.items():
            rows.append({
                "Configuration": name,
                "EM": res["avg_em"],
                "F1": res["avg_f1"],
                "Recall": res.get("avg_recall", 0),
                "Latency (s)": res["avg_latency"],
                "Avg Docs": res["avg_docs"]
            })

        df = pd.DataFrame(rows)
        ensure_directory(RESULTS_DIR)
        df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
        print("\n--- ABLATION STUDY RESULTS ---")
        print(df.to_markdown(index=False))

    # ------------------------------------------------------------------
    # Summary & plotting
    # ------------------------------------------------------------------
    def _save_summary(self, results: dict, filename: str) -> None:
        rows = []
        for name, r in results.items():
            rows.append({
                "Method": name,
                "EM": r.get("avg_em", 0),
                "F1": r.get("avg_f1", 0),
                "F1_CI_Low": r.get("f1_ci_low", 0),
                "F1_CI_High": r.get("f1_ci_high", 0),
                "Recall": r.get("avg_recall", 0),
                "HitRate": r.get("avg_hit_rate", 0),
                "Latency(s)": r.get("avg_latency", 0),
                "Ret_Lat(s)": r.get("avg_retrieval_latency", 0),
                "Gen_Lat(s)": r.get("avg_generation_latency", 0),
                "Latency_CI_Low": r.get("latency_ci_low", 0),
                "Latency_CI_High": r.get("latency_ci_high", 0),
                "AvgDocs": r.get("avg_docs", 0),
                "Tokens": r.get("avg_tokens", 0),
                "PromptTokens": r.get("avg_prompt_tokens", 0),
                "CompletionTokens": r.get("avg_completion_tokens", 0),
                "Cost($)": r.get("avg_cost", 0),
                "Throughput(q/s)": r.get("throughput", 0),
            })

        df = pd.DataFrame(rows)
        ensure_directory(RESULTS_DIR)
        df.to_csv(RESULTS_DIR / filename, index=False)
        print(f"\n--- {filename.upper().replace('.CSV', '')} RESULTS ---")
        print(df.to_markdown(index=False))

    def plot_pareto(self, results: dict) -> None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        fixed = {k: v for k, v in results.items() if k.startswith("Fixed")}
        fixed_names = sorted(fixed.keys(), key=lambda x: fixed[x]["avg_latency"])
        fixed_f1 = [fixed[n]["avg_f1"] for n in fixed_names]
        fixed_lat = [fixed[n]["avg_latency"] for n in fixed_names]

        ax.plot(fixed_lat, fixed_f1, 'o--', color='gray', alpha=0.7,
                label='Fixed Top-k', markersize=8, linewidth=1.5, zorder=1)

        for name, lat, f1 in zip(fixed_names, fixed_lat, fixed_f1):
            ax.annotate(name.replace("Fixed ", ""), (lat, f1),
                       textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        colors = {"AdaRAG-Heuristic": "#d62728", "AdaRAG-MLP": "#2ca02c", "Oracle": "#9467bd"}
        markers = {"AdaRAG-Heuristic": "*", "AdaRAG-MLP": "P", "Oracle": "X"}

        for name in ["AdaRAG-Heuristic", "AdaRAG-MLP", "Oracle"]:
            if name not in results:
                continue
            r = results[name]
            ax.scatter(r["avg_latency"], r["avg_f1"],
                      color=colors[name], marker=markers[name], s=300,
                      edgecolors='black', linewidths=1.5, zorder=5,
                      label=name)

        ax.fill_between(fixed_lat, 0, fixed_f1, alpha=0.1, color='gray', label='Dominated region')
        ax.set_xlabel("Average End-to-End Latency (seconds)", fontsize=12, fontweight='bold')
        ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
        ax.set_title("Efficiency-Quality Pareto Frontier", fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(fixed_f1 + [results.get("AdaRAG-Heuristic", {}).get("avg_f1", 0)]) * 1.15)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "pareto_frontier.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Pareto plot saved to {RESULTS_DIR / 'pareto_frontier.png'}")

    def plot_retrieval_pareto(self, results: dict) -> None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        fixed = {k: v for k, v in results.items() if k.startswith("Fixed")}
        fixed_names = sorted(fixed.keys(), key=lambda x: fixed[x]["avg_tokens"])
        fixed_recall = [fixed[n]["avg_recall"] for n in fixed_names]
        fixed_tokens = [fixed[n]["avg_tokens"] for n in fixed_names]

        ax.plot(fixed_tokens, fixed_recall, 'o--', color='gray', alpha=0.7,
                label='Fixed Top-k', markersize=8, linewidth=1.5)

        for name, tok, rec in zip(fixed_names, fixed_tokens, fixed_recall):
            ax.annotate(name.replace("Fixed ", ""), (tok, rec),
                       textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        if "AdaRAG-Heuristic" in results:
            r = results["AdaRAG-Heuristic"]
            ax.scatter(r["avg_tokens"], r["avg_recall"],
                      color='#d62728', marker='*', s=400,
                      edgecolors='black', linewidths=1.5, zorder=5,
                      label='AdaRAG')

        ax.set_xlabel("Average Prompt Tokens", fontsize=12, fontweight='bold')
        ax.set_ylabel("Retrieval Recall", fontsize=12, fontweight='bold')
        ax.set_title("Retrieval Efficiency-Quality Trade-off", fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "retrieval_pareto.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Retrieval Pareto plot saved to {RESULTS_DIR / 'retrieval_pareto.png'}")

    def plot_budget_distribution(self, result: dict, prefix: str = "") -> None:
        from collections import Counter
        top_ks = [r["top_k"] for r in result["raw_results"]]
        counts = Counter(top_ks)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        buckets = sorted(counts.keys())
        values = [counts[k] for k in buckets]
        colors = ['#2ca02c' if k <= 5 else '#ff7f0e' if k <= 8 else '#d62728' for k in buckets]

        bars = ax.bar([f"Top-{k}" for k in buckets], values, color=colors,
                      edgecolor='black', linewidth=1.2)
        ax.set_ylabel("Number of Queries", fontweight='bold')
        ax.set_xlabel("Assigned Retrieval Budget", fontweight='bold')
        ax.set_title("Adaptive Budget Distribution", fontweight='bold', fontsize=12)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        fname = f"{prefix}budget_distribution.png"
        plt.savefig(RESULTS_DIR / fname, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Budget distribution plot saved to {RESULTS_DIR / fname}")