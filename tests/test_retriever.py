"""
Testy dla retriever'a
"""

import unittest
from src.retriever import Retriever
from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore


class TestRetriever(unittest.TestCase):
    def setUp(self):
        """Przygotowanie dla każdego testu"""
        embedding_service = EmbeddingService()
        vector_store = VectorStore(embedding_service.get_dimension())
        self.retriever = Retriever(embedding_service, vector_store)

        # Dodanie testowych dokumentów
        documents = [
            "Sztuczna inteligencja to dziedzina informatyki",
            "Machine learning pozwala maszynom się uczyć",
            "Python jest popularnym językiem programowania",
            "Kotów są domowymi zwierzętami",
        ]
        self.retriever.add_documents(documents)

    def test_retrieve(self):
        """Test wyszukiwania"""
        query = "Co to jest AI?"
        results = self.retriever.retrieve(query, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIn("content", results[0])
        self.assertIn("score", results[0])

    def test_retrieve_with_metadata(self):
        """Test wyszukiwania z metadanymi"""
        documents = ["Dokument o informatyce"]
        metadata = [{"author": "John Doe", "date": "2024"}]

        self.retriever.add_documents(documents, metadata)
        results = self.retriever.retrieve("informatyka", top_k=1)

        self.assertIsNotNone(results[0]["metadata"])


if __name__ == "__main__":
    unittest.main()
