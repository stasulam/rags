"""
Główny pipeline RAG
"""
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator
from .data_loader import DataLoader
from typing import List, Dict


class RAGPipeline:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 generator_model: str = "Qwen/Qwen3-4B"):
        """
        Inicjalizuje RAG pipeline
        
        Args:
            embedding_model: Model do embeddingów
            generator_model: Model do generowania odpowiedzi (domyślnie microsoft/Phi-3-mini-4k-instruct)
        """
        self.embedding_service = EmbeddingService(embedding_model)
        self.vector_store = VectorStore(self.embedding_service.get_dimension())
        self.retriever = Retriever(self.embedding_service, self.vector_store)
        self.generator = Generator(generator_model)
        self.data_loader = DataLoader()
    
    def initialize_with_sample_data(self) -> None:
        """Inicjalizuje pipeline przy użyciu przykładowych danych"""
        # Pobranie przykładowych danych
        sample_data = self.data_loader.load_sample_data()
        documents, metadata = self.data_loader.format_documents(sample_data)
        
        # Dodanie dokumnetów
        self.retriever.add_documents(documents, metadata)
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Procesuje pytanie i zwraca odpowiedź wraz z źródłami
        
        Args:
            question: Pytanie użytkownika
            top_k: Liczba dokumentów do wyszukania
            
        Returns:
            Słownik z odpowiedzią i źródłami
        """
        # Wyszukanie dokumentów
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        
        # Generowanie odpowiedzi
        response = self.generator.generate(question, retrieved_docs)
        
        return {
            "question": question,
            "answer": response,
            "sources": retrieved_docs,
            "num_sources": len(retrieved_docs)
        }
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None) -> None:
        """
        Dodaje dokumenty do pipeline
        
        Args:
            documents: Lista dokumentów
            metadata: Metadane dla dokumentów
        """
        self.retriever.add_documents(documents, metadata)
    
    def get_vector_store_size(self) -> int:
        """Zwraca rozmiar magazynu wektorowego"""
        return self.vector_store.get_size()
