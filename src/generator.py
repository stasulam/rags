"""
Generator - generowanie odpowiedzi na podstawie kontekstu
"""
from transformers import pipeline
from typing import List, Dict


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        """
        Inicjalizuje generator

        Args:
            model_name: Nazwa modelu do generowania tekstu (domyślnie Qwen/Qwen3-4B)
        """
        try:
            self.generator = pipeline("text-generation", model=model_name, max_length=200)
        except Exception as e:
            # Fallback jeśli model nie jest dostępny
            print(f"⚠️ Nie można załadować modelu {model_name}: {str(e)}")
            print("Fallback na distilgpt2...")
            self.generator = pipeline("text-generation", model="distilgpt2", max_length=200)
    
    def generate(self, query: str, context_docs: List[Dict], max_length: int = 150) -> str:
        """
        Generuje odpowiedź na podstawie zapytania i dokumentów kontekstowych
        
        Args:
            query: Tekst zapytania
            context_docs: Lista dokumentów kontekstowych
            max_length: Maksymalna długość odpowiedzi
            
        Returns:
            Wygenerowana odpowiedź
        """
        # Przygotowanie kontekstu z wyszukanych dokumentów
        context = "\n".join([doc["content"] for doc in context_docs[:3]])
        
        # Przygotowanie promptu
        prompt = f"""Na podstawie poniższego kontekstu odpowiedz na pytanie.

Kontekst:
{context}

Pytanie: {query}

Odpowiedź:"""
        
        try:
            # Generowanie odpowiedzi
            result = self.generator(prompt, max_new_tokens=max_length, num_return_sequences=1)
            response = result[0]['generated_text']
            
            # Ekstrakcja tylko odpowiedzi (bez promptu)
            if "Odpowiedź:" in response:
                response = response.split("Odpowiedź:")[-1].strip()
            
            return response
        except Exception as e:
            # Fallback - zwrócenie najlepszego dokumentu jako odpowiedź
            if context_docs:
                return f"Na podstawie dostępnych źródeł: {context_docs[0]['content']}"
            return f"Nie mogę odpowiedzieć na pytanie: {query}"
    
    def generate_with_context(self, query: str, context: str, max_length: int = 150) -> str:
        """
        Generuje odpowiedź bezpośrednio z kontekstu tekstowego
        
        Args:
            query: Pytanie
            context: Tekst kontekstu
            max_length: Maksymalna długość odpowiedzi
            
        Returns:
            Odpowiedź
        """
        prompt = f"""Pytanie: {query}
Kontekst: {context}
Odpowiedź: """
        
        try:
            result = self.generator(prompt, max_new_tokens=max_length, num_return_sequences=1)
            return result[0]['generated_text']
        except Exception as e:
            return f"Błąd przy generowaniu odpowiedzi: {str(e)}"
