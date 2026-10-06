import json
from pathlib import Path
from config import CORPUS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.retriever import Retriever
from src.utils import logger

class Indexer:
    def __init__(self):
        self.retriever = Retriever()

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
            boundary = -1
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = text.rfind(sep, start + overlap, end)
                if idx != -1:
                    boundary = idx + len(sep)
                    break
            if boundary == -1:
                boundary = end
            chunks.append(text[start:boundary].strip())
            start = boundary - overlap
            if start < 0:
                start = 0
        return [c for c in chunks if c]

    def build_index(self) -> None:
        logger.info("Starting index construction...")
        documents = []
        corpus_path = Path(CORPUS_DIR)
        # Prefer a single explicit corpus source to avoid mixing unrelated datasets.
        preferred_sources = [
            corpus_path / "hotpot_corpus.json",
            corpus_path / "corpus_data.json",
        ]

        chosen_source = next((p for p in preferred_sources if p.exists()), None)
        if chosen_source is not None:
            logger.info(f"Using corpus source: {chosen_source.name}")
            with open(chosen_source, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        chunks = self.chunk_text(item)
                        documents.extend(chunks)
                    elif isinstance(item, dict) and "text" in item:
                        chunks = self.chunk_text(item["text"])
                        documents.extend(chunks)
        else:
            for path in corpus_path.glob("**/*"):
                if path.suffix == ".txt":
                    text = path.read_text(encoding="utf-8")
                    chunks = self.chunk_text(text)
                    documents.extend(chunks)
                elif path.suffix == ".json":
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, str):
                                    chunks = self.chunk_text(item)
                                    documents.extend(chunks)
                                elif isinstance(item, dict) and "text" in item:
                                    chunks = self.chunk_text(item["text"])
                                    documents.extend(chunks)
        if not documents:
            raise FileNotFoundError("No source documents found in data/corpus.")
        logger.info(f"Indexing {len(documents)} document chunks...")
        self.retriever.add_documents(documents)
        self.retriever.save()
        logger.info("FAISS Index build successfully completed.")
