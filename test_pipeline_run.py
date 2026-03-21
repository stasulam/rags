"""
Plik do testowania pipeline'u RAG
"""
from src.pipeline import RAGPipeline
import json


def test_pipeline():
    """Testuje RAG pipeline na przykładowych pytaniach"""
    
    print("🚀 Inicjalizacja RAG Pipeline...")
    pipeline = RAGPipeline()
    pipeline.initialize_with_sample_data()
    
    print(f"✅ Pipeline gotowy. Dokumentów w magazynie: {pipeline.get_vector_store_size()}\n")
    
    # Lista testowych pytań
    test_questions = [
        "Co to jest sztuczna inteligencja?",
        "Jakie są zastosowania machine learning?",
        "Opowiedz mi o Renesansie",
        "Czym zajmuje się biologia?",
        "Jakie są właściwości fizyki?"
    ]
    
    # Przetestowanie każdego pytania
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Pytanie #{i}: {question}")
        print('='*60)
        
        result = pipeline.query(question, top_k=3)
        results.append(result)
        
        print(f"\n📝 Odpowiedź:\n{result['answer']}\n")
        
        print(f"📚 Znalezione źródła ({result['num_sources']}):")
        for j, source in enumerate(result['sources'], 1):
            print(f"\n  Źródło {j}:")
            print(f"  Temat: {source['metadata'].get('topic', 'N/A')}")
            print(f"  Podobieństwo: {(1 - source['score'] / 2) * 100:.1f}%")
            print(f"  Tekst: {source['content'][:100]}...")
    
    # Zapis wyników
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n✅ Testy zakończone. Wyniki zapisane do 'test_results.json'")


if __name__ == "__main__":
    test_pipeline()
