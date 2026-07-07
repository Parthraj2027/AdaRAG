import argparse
import sys
from src.indexer import Indexer
from src.evaluator import Evaluator
from src.pipeline import RAGPipeline
from src.utils import logger, set_seed

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="AdaRAG: Query-Adaptive Retrieval Budgeting")
    parser.add_argument(
        "--mode",
        choices=["index", "evaluate", "ablate", "interactive", "download"],
        default="interactive",
        help="Pipeline mode to execute"
    )
    args = parser.parse_args()

    if args.mode == "download":
        evaluator = Evaluator()
        evaluator.download_and_prepare_hotpotqa(sample_size=50)
    elif args.mode == "index":
        indexer = Indexer()
        indexer.build_index()
    elif args.mode == "evaluate":
        evaluator = Evaluator()
        evaluator.run_experiments()
    elif args.mode == "ablate":
        evaluator = Evaluator()
        evaluator.run_ablation_study()
    elif args.mode == "interactive":
        try:
            pipeline = RAGPipeline()
        except FileNotFoundError:
            logger.warning("FAISS Index files not found. Automatically building index from data/corpus first.")
            indexer = Indexer()
            indexer.build_index()
            pipeline = RAGPipeline()

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
                    print(f"Content: {res['documents'][0]} (Score: {res['scores'][0]:.4f})")
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