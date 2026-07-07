import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, DEVICE
from src.utils import logger, validate_text

class Embedder:
    _model = None

    def __init__(self):
        if Embedder._model is None:
            logger.info("Loading embedding model...")
            Embedder._model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=DEVICE
            )
            logger.info(f"Loaded: {EMBEDDING_MODEL}")
        self.model = Embedder._model

    def encode(self, text: str) -> np.ndarray:
        text = validate_text(text)
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        validated_texts = [validate_text(t) for t in texts]
        embeddings = self.model.encode(
            validated_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings

    def embedding_dimension(self) -> int:
        try:
            return self.model.get_embedding_dimension()
        except AttributeError:
            return self.model.get_sentence_embedding_dimension()

    def similarity(self, text1: str, text2: str) -> float:
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        score = float(np.dot(emb1, emb2))
        return round(score, 4)