"""
Serwis embeddingów - konwersja tekstu na wektory
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicjalizuje serwis embeddingów

        Args:
            model_name: Nazwa modelu z sentence-transformers
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Tworzy embeddingi dla tekstu lub listy tekstów

        Args:
            texts: Tekst lub lista tekstów

        Returns:
            Numpy array z embeddingami
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def get_dimension(self) -> int:
        """Zwraca wymiar embeddingów"""
        return self.embedding_dim

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Oblicza podobieństwo cosinusowe między dwoma embeddingami

        Args:
            embedding1: Pierwszy embedding
            embedding2: Drugi embedding

        Returns:
            Wartość podobieństwa (0-1)
        """
        # Normalizacja wektorów
        emb1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        emb2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

        return np.dot(emb1, emb2)
