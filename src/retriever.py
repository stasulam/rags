"""
Retriever - wyszukiwanie dokumentów na podstawie zapytania
"""

from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from typing import List, Dict, Tuple


class Retriever:
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        """
        Inicjalizuje retriever

        Args:
            embedding_service: Serwis do tworzenia embeddingów
            vector_store: Magazyn wektorowy
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Wyszukuje dokumenty podobne do zapytania

        Args:
            query: Tekst zapytania
            top_k: Liczba dokumentów do zwrócenia

        Returns:
            Lista słowników z dokumentami, score podobieństwa (0-1) i metadanymi
        """
        # Tworzenie embeddingu dla zapytania
        query_embedding = self.embedding_service.embed(query)[0]

        # Wyszukiwanie w magazynie
        results = self.vector_store.search(query_embedding, k=top_k)

        # Formatowanie wyników
        retrieved_docs = []
        for doc, distance, metadata in results:
            similarity_score = 1 / (1 + distance)  # Konwersja odległości na podobieństwo (0-1)
            retrieved_docs.append({"content": doc, "score": similarity_score, "metadata": metadata})

        return retrieved_docs

    def add_documents(self, documents: List[str], metadata: List[Dict] = None) -> None:
        """
        Dodaje dokumenty do magazynu

        Args:
            documents: Lista dokumentów
            metadata: Metadane dla każdego dokumentu
        """
        # Tworzenie embeddingów
        embeddings = self.embedding_service.embed(documents)

        # Dodanie do magazynu
        self.vector_store.add_documents(embeddings, documents, metadata)
