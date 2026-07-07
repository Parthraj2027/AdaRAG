import json
import string
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from config import EVALUATION_DIR, RESULTS_DIR, CORPUS_DIR
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
        self.pipeline = None
        self.dataset = self.load_dataset()

    def load_dataset(self) -> list[dict]:
        path = Path(EVALUATION_DIR) / "hotpotqa_subset.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return [
            {
                "query": "Who developed FAISS and which company does it belong to?",
                "answer": "FAISS was developed by Meta.",
                "complexity": "easy"
            },
            {
                "query": "Compare TCP and UDP transport protocols and list their key differences.",
                "answer": "TCP is a reliable transport protocol whereas UDP is connectionless.",
                "complexity": "medium"
            },
            {
                "query": "Which protocol does HTTP/3 use, and who originally developed that underlying protocol?",
                "answer": "HTTP/3 uses QUIC, which was originally developed by Google.",
                "complexity": "hard"
            }
        ]

    def download_and_prepare_hotpotqa(self, sample_size: int = 50) -> None:
        logger.info("Downloading official HotpotQA dev dataset...")
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
            logger.info("Successfully downloaded. Processing samples...")
            
            qa_subset = []
            corpus_chunks = []
            
            for item in data[:sample_size]:
                qa_subset.append({
                    "query": item["question"],
                    "answer": item["answer"],
                    "complexity": item.get("type", "comparison")
                })
                for context_item in item["context"]:
                    title = context_item[0]
                    sentences = "".join(context_item[1])
                    corpus_chunks.append(f"{title}: {sentences}")

            ensure_directory(EVALUATION_DIR)
            ensure_directory(CORPUS_DIR)

            with open(Path(EVALUATION_DIR) / "hotpotqa_subset.json", "w", encoding="utf-8") as f:
                json.dump(qa_subset, f, indent=2)

            with open(Path(CORPUS_DIR) / "hotpot_corpus.json", "w", encoding="utf-8") as f:
                json.dump(corpus_chunks, f, indent=2)

            logger.info(f"Prepared {len(qa_subset)} evaluation queries and {len(corpus_chunks)} corpus chunks.")
            self.dataset = qa_subset
        except Exception as e:
            logger.warning(f"Failed to download dataset: {e}. Generating high-quality local benchmark datasets...")
            self.generate_local_benchmark()

    def generate_local_benchmark(self) -> None:
        local_qa = [
            {"query": "Who developed FAISS and which group does it belong to?", "answer": "FAISS was developed by Meta's Fundamental AI Research group.", "complexity": "easy"},
            {"query": "What is the third major version of the Hypertext Transfer Protocol?", "answer": "HTTP/3", "complexity": "easy"},
            {"query": "Which python library is designed for advanced natural language processing and is open source?", "answer": "spaCy", "complexity": "easy"},
            {"query": "What is the general chemical equation for photosynthesis?", "answer": "6CO2 + 6H2O + light energy -> C6H12O6 + 6O2", "complexity": "easy"},
            {"query": "What underlying model is used in the BAAI bge-small-en-v1.5 text embedding model?", "answer": "BAAI bge-small-en-v1.5 model is a highly efficient open-source text embedding model.", "complexity": "easy"},
            {"query": "Explain the difference between TCP and UDP, and state which one is connectionless.", "answer": "TCP is a reliable, connection-oriented protocol that requires a three-way handshake, whereas UDP is a connectionless protocol that does not guarantee delivery.", "complexity": "medium"},
            {"query": "What protocol does HTTP/3 use, and is it a connection-oriented protocol?", "answer": "HTTP/3 runs on top of QUIC, which is not TCP (connectionless-oriented multiplexing over UDP).", "complexity": "medium"},
            {"query": "How does Retrieval-Augmented Generation solve the hallucinations and private domain knowledge problems of LLMs?", "answer": "RAG retrieves relevant chunks from an external knowledge base and appends them to the user prompt, grounding the LLM's answers in facts.", "complexity": "medium"},
            {"query": "Which dataset is designed for multi-hop question answering and provides sentence-level supporting facts?", "answer": "HotpotQA", "complexity": "medium"},
            {"query": "What does a quantum computer use instead of classical bits, and what quantum mechanical principle allows them to run in parallel?", "answer": "Quantum computers use qubits, and superposition allows them to perform parallel computation.", "complexity": "medium"},
            {"query": "Compare the creators of the QUIC protocol and the creators of the FAISS library.", "answer": "QUIC was designed by Google, whereas FAISS was developed by Meta's Fundamental AI Research group.", "complexity": "hard"},
            {"query": "Evaluate how HTTP/3 improves connection stability during network migrations, and explain which organization proposed its underlying protocol.", "answer": "HTTP/3 improves stability by using QUIC, a multiplexed and secure transport protocol proposed by Google in 2012.", "complexity": "hard"},
            {"query": "Compare the dimensions and developers of the BAAI embedding model and the small English spaCy model.", "answer": "The BAAI model has 384 dimensions and was trained by the Beijing Academy of Artificial Intelligence. The small English spaCy pipeline (en_core_web_sm) is developed by the spaCy open-source contributors.", "complexity": "hard"},
            {"query": "Explain how RAG combines information retrieval with LLM generation, and details of the HotpotQA dataset designed to evaluate this multi-hop reasoning capability.", "answer": "RAG embeds user queries, retrieves top-k chunks, and constructs a grounded prompt for the LLM. HotpotQA evaluates this by requiring multi-hop reasoning over Wikipedia articles with gold sentence facts.", "complexity": "hard"},
            {"query": "Identify the chemical process that converts light energy to glucose and compare it with the concept of quantum superpositions.", "answer": "Photosynthesis is the chemical process converting light to glucose. Quantum superposition is a physical concept where qubits exist in multiple states simultaneously.", "complexity": "hard"}
        ]
        
        local_corpus = [
            "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval from an external knowledge base with a generative large language model (LLM). This framework helps address LLM limitations such as factual hallucinations, lack of private domain knowledge, and outdated training data. In a typical RAG system, user queries are first embedded, then compared against a vectorized corpus to extract the top-k most similar text chunks. These chunks are then appended to the user prompt to construct a grounded context for the generator.",
            "FAISS (Facebook AI Similarity Search) is an open-source library developed by Meta's Fundamental AI Research group. It is specifically designed for efficient similarity search and clustering of dense vectors. FAISS contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It supports standard indexing approaches, including flat indexes using L2 distance or inner product, inverted file indexes for sub-linear search times, and quantized indices for reduced memory footprint.",
            "HTTP/3 is the third major version of the Hypertext Transfer Protocol. Unlike HTTP/1.1 and HTTP/2, which rely on the Transmission Control Protocol (TCP), HTTP/3 runs on top of QUIC. QUIC is a multiplexed and secure transport protocol originally designed and proposed by Google in 2012. By utilizing QUIC, HTTP/3 solves the head-of-line blocking problem inherent in TCP, provides faster connection establishment via zero round-trip time (0-RTT) handshakes, and maintains connection stability during network migrations.",
            "The Transmission Control Protocol (TCP) is a core protocol of the Internet Protocol Suite. It provides reliable, ordered, and error-checked delivery of a stream of octets between applications running on hosts communicating over an IP network. TCP is connection-oriented, requiring a three-way handshake before data transfer starts. In contrast, the User Datagram Protocol (UDP) is a simpler, connectionless communication protocol that does not guarantee delivery, order, or error correction, making it ideal for real-time applications like video streaming and gaming.",
            "SentenceTransformers is a Python framework for state-of-the-art sentence, text, and image embeddings. It allows users to compute dense vector representations for sentences or paragraphs using pretrained transformer models. The BAAI bge-small-en-v1.5 model is a highly efficient open-source text embedding model trained by the Beijing Academy of Artificial Intelligence. It maps queries and document chunks into a 384-dimensional vector space, exhibiting strong performance on retrieval benchmarks while maintaining low inference latency.",
            "spaCy is a free, open-source library for advanced Natural Language Processing in Python. It is designed specifically for production use, offering industrial-strength speed and accuracy. spaCy provides pre-trained pipelines for tokenization, parsing, named entity recognition (NER), part-of-speech tagging, and text classification. The 'en_core_web_sm' model is a small English pipeline that includes components for tagger, parser, senter, and ner, trained on written web text.",
            "OpenAI is an artificial intelligence research laboratory that developed the Generative Pre-trained Transformer (GPT) series of models. The gpt-4o-mini model is a highly efficient, lightweight version of OpenAI's GPT-4o, optimized for speed, cost, and high performance on reasoning tasks. It is widely used in RAG systems as the response generation agent because of its stable API and lower cost per token.",
            "HotpotQA is a dataset for diverse, explainable multi-hop question answering. It features 113,000 QA pairs based on Wikipedia, requiring systems to find and reason over multiple supporting documents to answer a question. The dataset is notable because it provides sentence-level supporting facts (gold sentences) that the models must locate to perform reasoning, presenting a rigorous testbed for multi-hop RAG systems.",
            "Quantum computing is a multidisciplinary field comprising aspects of computer science, physics, and mathematics that utilizes quantum mechanics to solve complex problems faster than on classical computers. The field of quantum computing includes hardware research and application development. Quantum computers make use of qubits, which can exist in superposition, allowing them to perform computation in parallel.",
            "Photosynthesis is the chemical process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose. The process occurs in the chloroplasts of plant cells, utilizing carbon dioxide and water to produce oxygen and sugar. The general equation is 6CO2 + 6H2O + light energy yields C6H12O6 + 6O2."
        ]

        ensure_directory(EVALUATION_DIR)
        ensure_directory(CORPUS_DIR)

        with open(Path(EVALUATION_DIR) / "hotpotqa_subset.json", "w", encoding="utf-8") as f:
            json.dump(local_qa, f, indent=2)

        with open(Path(CORPUS_DIR) / "hotpot_corpus.json", "w", encoding="utf-8") as f:
            json.dump(local_corpus, f, indent=2)

        logger.info("Generated high-quality local benchmark dataset and corpus files successfully.")
        self.dataset = local_qa

    def evaluate_run(self, mode: str, fixed_k: int = 5, ablated: list[str] = None) -> dict:
        if self.pipeline is None:
            self.pipeline = RAGPipeline()
            
        results = []
        total_latency = 0.0
        total_docs = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for item in self.dataset:
            start_time = time.perf_counter()
            res = self.pipeline.run(item["query"], mode=mode, fixed_k=fixed_k, ablated=ablated)
            latency = time.perf_counter() - start_time

            em = compute_exact(item["answer"], res["answer"])
            f1 = compute_f1(item["answer"], res["answer"])

            results.append({
                "query": item["query"],
                "gold": item["answer"],
                "pred": res["answer"],
                "em": em,
                "f1": f1,
                "latency": latency,
                "top_k": res["top_k"],
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"]
            })

            total_latency += latency
            total_docs += res["top_k"]
            total_prompt_tokens += res["prompt_tokens"]
            total_completion_tokens += res["completion_tokens"]

        n = len(self.dataset)
        avg_em = sum(r["em"] for r in results) / n
        avg_f1 = sum(r["f1"] for r in results) / n
        avg_latency = total_latency / n
        avg_docs = total_docs / n
        avg_prompt_tok = total_prompt_tokens / n
        avg_compl_tok = total_completion_tokens / n
        avg_cost = (avg_prompt_tok * 0.15 + avg_compl_tok * 0.60) / 1000000

        return {
            "avg_em": round(avg_em, 4),
            "avg_f1": round(avg_f1, 4),
            "avg_latency": round(avg_latency, 4),
            "avg_docs": round(avg_docs, 2),
            "avg_tokens": round(avg_prompt_tok + avg_compl_tok, 2),
            "avg_cost": round(avg_cost, 7),
            "raw_results": results
        }

    def run_experiments(self) -> None:
        logger.info("Running Experiment 1: Fixed Top-5")
        exp1 = self.evaluate_run(mode="fixed", fixed_k=5)
        logger.info("Running Experiment 2: Fixed Top-10")
        exp2 = self.evaluate_run(mode="fixed", fixed_k=10)
        logger.info("Running Experiment 3: AdaRAG (Adaptive)")
        exp3 = self.evaluate_run(mode="adaptive")

        summary = pd.DataFrame([
            {"Method": "Fixed Top-5", "EM": exp1["avg_em"], "F1": exp1["avg_f1"], "Latency (s)": exp1["avg_latency"], "Avg Docs": exp1["avg_docs"], "Tokens": exp1["avg_tokens"], "Cost ($)": exp1["avg_cost"]},
            {"Method": "Fixed Top-10", "EM": exp2["avg_em"], "F1": exp2["avg_f1"], "Latency (s)": exp2["avg_latency"], "Avg Docs": exp2["avg_docs"], "Tokens": exp2["avg_tokens"], "Cost ($)": exp2["avg_cost"]},
            {"Method": "AdaRAG (Ours)", "EM": exp3["avg_em"], "F1": exp3["avg_f1"], "Latency (s)": exp3["avg_latency"], "Avg Docs": exp3["avg_docs"], "Tokens": exp3["avg_tokens"], "Cost ($)": exp3["avg_cost"]}
        ])

        ensure_directory(RESULTS_DIR)
        summary.to_csv(RESULTS_DIR / "experiment_results.csv", index=False)
        print("\n--- EXPERIMENT RESULTS ---")
        print(summary.to_markdown(index=False))

        self.plot_results(exp1, exp2, exp3)

    def run_ablation_study(self) -> None:
        logger.info("Running Baseline AdaRAG")
        base = self.evaluate_run(mode="adaptive")
        logger.info("Running Ablation: No Lexical")
        no_lex = self.evaluate_run(mode="adaptive", ablated=["lexical"])
        logger.info("Running Ablation: No Entities")
        no_ent = self.evaluate_run(mode="adaptive", ablated=["entity"])
        logger.info("Running Ablation: No Reasoning")
        no_reas = self.evaluate_run(mode="adaptive", ablated=["reasoning"])
        logger.info("Running Ablation: No Constraints")
        no_const = self.evaluate_run(mode="adaptive", ablated=["constraint"])

        ablation = pd.DataFrame([
            {"Configuration": "Full AdaRAG", "EM": base["avg_em"], "F1": base["avg_f1"], "Latency (s)": base["avg_latency"], "Avg Docs": base["avg_docs"]},
            {"Configuration": "- No Lexical Features", "EM": no_lex["avg_em"], "F1": no_lex["avg_f1"], "Latency (s)": no_lex["avg_latency"], "Avg Docs": no_lex["avg_docs"]},
            {"Configuration": "- No Entity Features", "EM": no_ent["avg_em"], "F1": no_ent["avg_f1"], "Latency (s)": no_ent["avg_latency"], "Avg Docs": no_ent["avg_docs"]},
            {"Configuration": "- No Reasoning Features", "EM": no_reas["avg_em"], "F1": no_reas["avg_f1"], "Latency (s)": no_reas["avg_latency"], "Avg Docs": no_reas["avg_docs"]},
            {"Configuration": "- No Constraint Features", "EM": no_const["avg_em"], "F1": no_const["avg_f1"], "Latency (s)": no_const["avg_latency"], "Avg Docs": no_const["avg_docs"]}
        ])

        ablation.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
        print("\n--- ABLATION STUDY RESULTS ---")
        print(ablation.to_markdown(index=False))

    def plot_results(self, exp1: dict, exp2: dict, exp3: dict) -> None:
        methods = ["Fixed Top-5", "Fixed Top-10", "AdaRAG"]
        f1_scores = [exp1["avg_f1"], exp2["avg_f1"], exp3["avg_f1"]]
        latencies = [exp1["avg_latency"], exp2["avg_latency"], exp3["avg_latency"]]
        avg_docs = [exp1["avg_docs"], exp2["avg_docs"], exp3["avg_docs"]]

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

        color = "#1f77b4"
        ax1.set_xlabel("Methods", fontweight="bold")
        ax1.set_ylabel("F1 Score", color=color, fontweight="bold")
        ax1.bar(methods, f1_scores, color=color, alpha=0.7, width=0.35, edgecolor="black", linewidth=1.2)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim(0.0, 1.0)

        ax1_twin = ax1.twinx()
        color = "#d62728"
        ax1_twin.set_ylabel("Latency (seconds)", color=color, fontweight="bold")
        ax1_twin.plot(methods, latencies, color=color, marker="o", linewidth=2.5, markersize=8, label="Latency")
        ax1_twin.tick_params(axis="y", labelcolor=color)
        ax1.set_title("Retrieval Quality (F1) vs Latency", fontweight="bold", fontsize=12)

        ax2.bar(methods, avg_docs, color="#2ca02c", alpha=0.7, width=0.35, edgecolor="black", linewidth=1.2)
        ax2.set_xlabel("Methods", fontweight="bold")
        ax2.set_ylabel("Average Retrieved Chunks", fontweight="bold")
        ax2.set_title("Average Retrieval Size Comparison", fontweight="bold", fontsize=12)
        ax2.set_ylim(0.0, max(avg_docs) + 2.0)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "evaluation_comparison.png", dpi=300)
        plt.close()
        logger.info(f"Comparison plot saved to {RESULTS_DIR / 'evaluation_comparison.png'}")
