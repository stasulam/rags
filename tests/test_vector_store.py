"""
Testy dla magazynu wektorowego
"""

import unittest
import numpy as np
import tempfile
import os
from src.vector_store import VectorStore
from src.embedding_service import EmbeddingService


class TestVectorStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Inicjalizacja dla wszystkich testów"""
        cls.embedding_dim = 384
        cls.temp_dir = tempfile.mkdtemp()
        cls.store_path = os.path.join(cls.temp_dir, "test_store.index")

    def setUp(self):
        """Przygotowanie dla każdego testu"""
        self.store = VectorStore(self.embedding_dim, self.store_path)

    def test_add_documents(self):
        """Test dodawania dokumentów"""
        embeddings = np.random.rand(3, self.embedding_dim).astype("float32")
        documents = ["Doc 1", "Doc 2", "Doc 3"]

        self.store.add_documents(embeddings, documents)
        self.assertEqual(self.store.get_size(), 3)

    def test_search(self):
        """Test wyszukiwania"""
        embeddings = np.random.rand(5, self.embedding_dim).astype("float32")
        documents = [f"Document {i}" for i in range(5)]

        self.store.add_documents(embeddings, documents)

        query_embedding = np.random.rand(self.embedding_dim).astype("float32")
        results = self.store.search(query_embedding, k=3)

        self.assertEqual(len(results), 3)

    def test_save_and_load(self):
        """Test zapisywania i ładowania"""
        embeddings = np.random.rand(2, self.embedding_dim).astype("float32")
        documents = ["Doc 1", "Doc 2"]

        self.store.add_documents(embeddings, documents)
        initial_size = self.store.get_size()

        self.store.save()

        new_store = VectorStore(self.embedding_dim, self.store_path)
        new_store.load()

        self.assertEqual(new_store.get_size(), initial_size)


if __name__ == "__main__":
    unittest.main()
