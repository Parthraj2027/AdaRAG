import json
import urllib.request
from pathlib import Path

url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
print("Downloading... this may take 2-3 minutes", flush=True)
with urllib.request.urlopen(req, timeout=120) as response:
    data = json.loads(response.read().decode("utf-8"))

print(f"Total questions in dev set: {len(data)}")
subset = data[:500]
Path("data/evaluation").mkdir(parents=True, exist_ok=True)
Path("data/corpus").mkdir(parents=True, exist_ok=True)

with open("data/evaluation/hotpotqa_eval.json", "w", encoding="utf-8") as f:
    json.dump([{
        "query": item["question"],
        "answer": item["answer"],
        "type": item.get("type", "unknown"),
        "level": item.get("level", "unknown"),
        "supporting_facts": item.get("supporting_facts", [])
    } for item in subset], f, indent=2)

all_paragraphs = {}
for item in subset:
    for ctx in item.get("context", []):
        title, sentences = ctx[0], ctx[1]
        all_paragraphs[title] = f"{title}: {' '.join(sentences)}"

with open("data/corpus/hotpot_corpus.json", "w", encoding="utf-8") as f:
    json.dump(list(all_paragraphs.values()), f, indent=2)

print(f"Saved {len(subset)} questions and {len(all_paragraphs)} unique paragraphs.")
