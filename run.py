import argparse
import sys
from pathlib import Path
from config import DATA_DIR, EVALUATION_DIR
from src.utils import logger, set_seed

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="AdaRAG: Query-Adaptive Retrieval Budgeting")
    parser.add_argument(
        "--mode",
        choices=["index", "evaluate", "evaluate-retrieval", "evaluate-llm-subset", "ablate", "interactive", "download", "compute-stats", "tune", "train-mlp", "preflight"],
        default="interactive",
        help="Pipeline mode to execute"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of HotpotQA questions to evaluate/download"
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=200,
        help="Number of validation questions to use for budget tuning"
    )
    parser.add_argument(
        "--budget_config",
        type=Path,
        default=Path(DATA_DIR) / "best_budget_config.json",
        help="Path to tuned budget config JSON"
    )
    parser.add_argument(
        "--feature_stats",
        type=Path,
        default=Path(EVALUATION_DIR) / "feature_stats.json",
        help="Path to feature min/max stats JSON"
    )
    parser.add_argument(
        "--allow_local_eval",
        action="store_true",
        help="Allow evaluation/ablation/training on local or non-publication datasets"
    )
    parser.add_argument(
        "--latency_runs",
        type=int,
        default=1,
        help="Number of repeated generations per query; median latency is reported"
    )
    parser.add_argument(
        "--llm_subset_size",
        type=int,
        default=30,
        help="Number of queries for the LLM subset evaluation"
    )
    args = parser.parse_args()

    if args.mode == "download":
        from src.evaluator import Evaluator
        evaluator = Evaluator()
        evaluator.download_and_prepare_hotpotqa(sample_size=args.sample_size)
    elif args.mode == "preflight":
        from src.preflight import run_preflight
        run_preflight(
            sample_size=args.sample_size,
            latency_runs=args.latency_runs,
            budget_config_path=args.budget_config,
            feature_stats_path=args.feature_stats,
        )
    elif args.mode == "compute-stats":
        from src.compute_stats import compute_stats
        compute_stats(
            Path(EVALUATION_DIR) / "hotpotqa_eval.json",
            args.feature_stats,
            limit=args.val_size
        )
    elif args.mode == "tune":
        from src.tune_budget import grid_search
        grid_search(
            val_size=args.val_size,
            sample_size=args.sample_size,
            output_path=args.budget_config
        )
    elif args.mode == "index":
        from src.indexer import Indexer
        indexer = Indexer()
        indexer.build_index()
    elif args.mode == "evaluate":
        from src.evaluator import Evaluator
        evaluator = Evaluator(
            sample_size=args.sample_size,
            budget_config_path=args.budget_config,
            feature_stats_path=args.feature_stats,
            require_official_dataset=not args.allow_local_eval,
        )
        if not args.allow_local_eval and args.sample_size >= 500:
            evaluator.assert_publication_ready()
        elif not args.allow_local_eval:
            logger.warning("Running a pilot evaluation below 500 queries; do not use this as publication-grade evidence.")
        evaluator.run_experiments(sample_size=args.sample_size, latency_runs=args.latency_runs)
    elif args.mode == "evaluate-retrieval":
        from src.evaluator import Evaluator
        evaluator = Evaluator()
        evaluator.run_retrieval_experiments(sample_size=args.sample_size)
    elif args.mode == "evaluate-llm-subset":
        from src.evaluator import Evaluator
        evaluator = Evaluator()
        evaluator.run_llm_subset_experiments(
            n=args.llm_subset_size,
            latency_runs=args.latency_runs,
        )
    elif args.mode == "ablate":
        from src.evaluator import Evaluator
        evaluator = Evaluator(
            sample_size=args.sample_size,
            budget_config_path=args.budget_config,
            feature_stats_path=args.feature_stats,
            require_official_dataset=not args.allow_local_eval,
        )
        if not args.allow_local_eval:
            evaluator.assert_publication_ready()
        evaluator.run_ablation_study()
    elif args.mode == "train-mlp":
        if not args.allow_local_eval:
            from src.evaluator import Evaluator
            evaluator = Evaluator(sample_size=args.sample_size, require_official_dataset=True)
            evaluator.assert_publication_ready()
        from src.train_rds_mlp import train
        train()
    elif args.mode == "interactive":
        from src.indexer import Indexer
        from src.pipeline import RAGPipeline
        try:
            pipeline = RAGPipeline(
                budget_config_path=args.budget_config,
                feature_stats_path=args.feature_stats
            )
        except FileNotFoundError:
            logger.warning("FAISS Index files not found. Automatically building index from data/corpus first.")
            indexer = Indexer()
            indexer.build_index()
            pipeline = RAGPipeline(
                budget_config_path=args.budget_config,
                feature_stats_path=args.feature_stats
            )

        print("\n======================================================================")
        print("AdaRAG Interactive Mode (Query-Adaptive Retrieval Budgeting)")
        print("======================================================================")
        while True:
            try:
                query = input("\nEnter your query (or type 'exit' to quit): ").strip()
                if not query:
                    continue
                if query.lower() == "exit":
                    break
                res = pipeline.run(query, mode="adaptive")
                print("\n--- AdaRAG Execution Summary ---")
                print(f"Query             : {res['query']}")
                print(f"Linguistic Score  : {res['demand_score']}")
                print(f"Retrieval Budget  : Top-{res['top_k']}")
                print(f"Context Pieces    : {len(res['documents'])}")
                print(f"Tokens Used       : {res['total_tokens']}")
                print("\n--- Retrieved Documents (Top Match) ---")
                if res["documents"]:
                    print(f"Content: {res['documents'][0]} (Score: {res['retrieval_scores'][0]:.4f})")
                else:
                    print("None")
                print("\n--- Final Generated Answer ---")
                print(res["answer"])
                print("======================================================================")
            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Failed to process query: {e}")

if __name__ == "__main__":
    main()
