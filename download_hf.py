import json
from pathlib import Path

from datasets import load_dataset

print("Downloading HotpotQA from Hugging Face...", flush=True)
dataset = load_dataset("hotpot_qa", "distractor", split="validation")
subset = dataset.select(range(min(500, len(dataset))))

qa_subset = []
all_paragraphs = {}

for item in subset:
    context = item.get("context", {})
    titles = context.get("title", []) if isinstance(context, dict) else []
    sentences = context.get("sentences", []) if isinstance(context, dict) else []
    context_items = list(zip(titles, sentences))

    supporting_facts = item.get("supporting_facts", {})
    fact_titles = supporting_facts.get("title", []) if isinstance(supporting_facts, dict) else []
    fact_indices = supporting_facts.get("sent_id", []) if isinstance(supporting_facts, dict) else []
    normalized_facts = [[title, index] for title, index in zip(fact_titles, fact_indices)]

    qa_subset.append({
        "query": item["question"],
        "answer": item["answer"],
        "type": item.get("type", "unknown"),
        "level": item.get("level", "unknown"),
        "supporting_facts": normalized_facts,
    })

    for title, paragraph_sentences in context_items:
        all_paragraphs[title] = f"{title}: {' '.join(paragraph_sentences)}"

Path("data/evaluation").mkdir(parents=True, exist_ok=True)
Path("data/corpus").mkdir(parents=True, exist_ok=True)

with open("data/evaluation/hotpotqa_eval.json", "w", encoding="utf-8") as file:
    json.dump(qa_subset, file, indent=2)

with open("data/corpus/hotpot_corpus.json", "w", encoding="utf-8") as file:
    json.dump(list(all_paragraphs.values()), file, indent=2)

print(f"Total validation questions: {len(dataset)}")
print(f"Saved {len(qa_subset)} questions and {len(all_paragraphs)} unique paragraphs.")
