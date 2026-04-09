"""
Magazyn wektorowy - przechowywanie i wyszukiwanie dokumentów
"""

import faiss
import numpy as np
from typing import List, Dict, Tuple
import os


class VectorStore:
    def __init__(self, embedding_dim: int, store_path: str = "models/vector_store.index"):
        """
        Inicjalizuje magazyn wektorowy

        Args:
            embedding_dim: Wymiar embeddingów
            store_path: Ścieżka do zapisu indeksu
        """
        self.embedding_dim = embedding_dim
        self.store_path = store_path
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        self.metadata = []

    def add_documents(
        self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict] = None
    ) -> None:
        """
        Dodaje dokumenty do magazynu

        Args:
            embeddings: Numpy array z embeddingami
            documents: Lista tekstów dokumentów
            metadata: Lista metadanych dla każdego dokumentu
        """
        if len(embeddings) != len(documents):
            raise ValueError("Liczba embeddingów musi się zgadzać z liczbą dokumentów")

        # Konwersja do float32 (wymóg FAISS)
        embeddings = np.array(embeddings).astype("float32")

        # Dodanie do indeksu
        self.index.add(embeddings)

        # Przechowywanie dokumentów i metadanych
        self.documents.extend(documents)
        if metadata is None:
            metadata = [{"id": i} for i in range(len(documents))]
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Wyszukuje k najbardziej podobnych dokumentów

        Args:
            query_embedding: Embedding zapytania
            k: Liczba dokumentów do zwrócenia

        Returns:
            Lista tuple'i (dokument, odległość euklidesowa, metadane)
        """
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append(
                    (
                        self.documents[idx],
                        float(distance),
                        self.metadata[idx] if idx < len(self.metadata) else {},
                    )
                )

        return results

    def save(self) -> None:
        """Zapisuje indeks na dysk"""
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        faiss.write_index(self.index, self.store_path)

    def load(self) -> None:
        """Ładuje indeks z dysku"""
        if os.path.exists(self.store_path):
            self.index = faiss.read_index(self.store_path)

    def get_size(self) -> int:
        """Zwraca liczbę dokumentów w magazynie"""
        return self.index.ntotal
