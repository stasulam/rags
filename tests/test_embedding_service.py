"""
Testy dla serwisu embeddingów
"""
import unittest
import numpy as np
from src.embedding_service import EmbeddingService


class TestEmbeddingService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Inicjalizacja dla wszystkich testów"""
        cls.service = EmbeddingService()
    
    def test_embed_single_text(self):
        """Test embeddingu dla pojedynczego tekstu"""
        text = "To jest test"
        embedding = self.service.embed(text)
        
        self.assertEqual(embedding.shape, (1, self.service.get_dimension()))
    
    def test_embed_multiple_texts(self):
        """Test embeddingu dla wielu tekstów"""
        texts = ["Pierwszy tekst", "Drugi tekst", "Trzeci tekst"]
        embeddings = self.service.embed(texts)
        
        self.assertEqual(embeddings.shape, (len(texts), self.service.get_dimension()))
    
    def test_embedding_dimension(self):
        """Test wymiaru embeddingu"""
        dim = self.service.get_dimension()
        self.assertGreater(dim, 0)
    
    def test_similarity_score(self):
        """Test obliczeń podobieństwa"""
        text1 = "Koty są domowymi zwierzętami"
        text2 = "Koty są fajne"
        text3 = "Dynamika płynów jest trudna"
        
        emb1 = self.service.embed(text1)[0]
        emb2 = self.service.embed(text2)[0]
        emb3 = self.service.embed(text3)[0]
        
        similarity_12 = self.service.similarity(emb1, emb2)
        similarity_13 = self.service.similarity(emb1, emb3)
        
        # Podobne teksty powinny mieć wyższe podobieństwo
        self.assertGreater(similarity_12, similarity_13)


if __name__ == "__main__":
    unittest.main()
