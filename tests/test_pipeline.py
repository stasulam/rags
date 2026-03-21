"""
Testy dla pipeline'u RAG
"""
import unittest
from src.pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        """Przygotowanie dla każdego testu"""
        self.pipeline = RAGPipeline()
        self.pipeline.initialize_with_sample_data()
    
    def test_initialize_pipeline(self):
        """Test inicjalizacji pipeline"""
        size = self.pipeline.get_vector_store_size()
        self.assertGreater(size, 0)
    
    def test_query(self):
        """Test zapytania do pipeline"""
        result = self.pipeline.query("Co to jest sztuczna inteligencja?", top_k=3)
        
        self.assertIn("question", result)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertEqual(len(result["sources"]), 3)
    
    def test_add_documents(self):
        """Test dodawania dokumentów"""
        initial_size = self.pipeline.get_vector_store_size()
        
        new_docs = ["Nowy dokument 1", "Nowy dokument 2"]
        self.pipeline.add_documents(new_docs)
        
        new_size = self.pipeline.get_vector_store_size()
        self.assertEqual(new_size, initial_size + len(new_docs))


if __name__ == "__main__":
    unittest.main()
