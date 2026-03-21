# 🤖 RAG Pipeline - Zaawansowane Wyszukiwanie i Generowanie

Projekt implementujący **Retrieval-Augmented Generation (RAG)** - system łączący wyszukiwanie semantyczne dokumentów z generowaniem odpowiedzi na naturalne pytania.

## 📋 Przegląd

RAG Pipeline łączy dwie główne techniki:
1. **Retrieval** - Wyszukiwanie relevantnych dokumentów na podstawie zapytania użytkownika
2. **Generation** - Generowanie odpowiedzi na podstawie znalezionych dokumentów i pytania

## 🎯 Funkcjonalności

- ✅ **Embedding Service** - Konwersja tekstu na wektory za pomocą Sentence Transformers
- ✅ **Vector Store** - Przechowywanie i wyszukiwanie dokumentów z FAISS
- ✅ **Retriever** - Semantyczne wyszukiwanie dokumentów
- ✅ **Generator** - Generowanie odpowiedzi na pythonie zmieniane transformery
- ✅ **Streamlit GUI** - Interaktywny interfejs graficzny
- ✅ **Sample Data** - Dane przykładowe z różnych tematów (technologia, science, historia, etc.)

## 📁 Struktura projektu

```
rags/
├── src/
│   ├── __init__.py                 # Inicjalizacja pakietu
│   ├── embedding_service.py        # Serwis embeddingów
│   ├── vector_store.py             # Magazyn wektorowy (FAISS)
│   ├── retriever.py                # Wyszukiwanie dokumentów
│   ├── generator.py                # Generowanie odpowiedzi
│   ├── data_loader.py              # Ładowanie danych
│   └── pipeline.py                 # Główny pipeline RAG
├── data/
│   └── sample_data.json            # Dane przykładowe
├── models/
│   └── vector_store.index          # Indeks FAISS
├── tests/
│   ├── test_embedding_service.py   # Testy embeddingów
│   ├── test_vector_store.py        # Testy magazynu
│   ├── test_retriever.py           # Testy retriever'a
│   └── test_pipeline.py            # Testy pipeline'u
├── app.py                          # Główny interfejs Streamlit
├── requirements.txt                # Zależności projektu
└── README.md                       # Ten plik
```

## 🚀 Instalacja

### Wymagania
- Python 3.8+
- pip lub conda

### Kroki instalacji

1. **Klonowanie repozytorium**
```bash
cd ~/Documents/projects/rags
```

2. **Tworzenie wirtualnego środowiska (opcjonalne)**
```bash
python -m venv venv
source venv/bin/activate  # Na macOS/Linux
# lub
venv\Scripts\activate  # Na Windows
```

3. **Instalacja zależności**
```bash
pip install -r requirements.txt
```

## 💻 Uruchomienie

### Uruchomienie interfejsu GUI (Streamlit)

```bash
streamlit run app.py
```

Aplikacja otworzy się w przeglądarce pod adresem `http://localhost:8501`

### Testowanie jednostkowe

```bash
python -m pytest tests/
# lub
python -m unittest discover tests/
```

## 📚 Użytkowanie

### 1. Interfejs Streamlit

Aplikacja oferuje:
- ❓ Panel do zadawania pytań
- 🔍 Wyszukiwanie dokumentów
- 📝 Generowanie odpowiedzi
- 📊 Wyświetlanie źródeł
- 📜 Historia pytań
- ⚙️ Ustawienia parametrów

### 2. Programistyczne użytkowanie

```python
from src.pipeline import RAGPipeline

# Inicjalizacja pipeline'u
pipeline = RAGPipeline()
pipeline.initialize_with_sample_data()

# Zapytanie
result = pipeline.query("Co to jest sztuczna inteligencja?", top_k=5)

print(f"Pytanie: {result['question']}")
print(f"Odpowiedź: {result['answer']}")
print(f"Liczba źródeł: {result['num_sources']}")

for source in result['sources']:
    print(f"- {source['content'][:100]}... (Score: {source['score']:.4f})")
```

### 3. Dodawanie nowych dokumentów

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.initialize_with_sample_data()

# Dodanie nowych dokumentów
new_docs = [
    "Python to język programowania stworzony przez Guido van Rossum",
    "Django jest framework'iem webowym dla Pythona"
]

metadata = [
    {"topic": "Programowanie", "language": "pl"},
    {"topic": "Web Development", "language": "pl"}
]

pipeline.add_documents(new_docs, metadata)

# Zapytanie
result = pipeline.query("Opowiedz mi o Pythonie")
```

## 🧬 Architektura

### Przepływ danych

```
Pytanie użytkownika
        ↓
[Embedding Service] - Konwersja pytania na wektor
        ↓
[Retriever] - Wyszukiwanie w [Vector Store]
        ↓
Znalezione dokumenty
        ↓
[Generator] - Generowanie odpowiedzi
        ↓
Odpowiedź użytkownika
```

### Komponenty

#### EmbeddingService
- Model: `all-MiniLM-L6-v2` (domyślnie)
- Wymiar embeddingów: 384
- Obsługuje pojedyncze teksty i listy tekstów

#### VectorStore
- Backend: FAISS (Facebook AI Similarity Search)
- Algorytm: L2 (Euclidean distance)
- Obsługuje save/load indeksu

#### Retriever
- Wyszukuje K najbardziej podobnych dokumentów
- Zwraca dokumenty z metadata i score'ami

#### Generator
- Model: GPT-2 (domyślnie)
- Generuje tekstowe odpowiedzi
- Obsługuje kontekst złożony z wyszukanych dokumentów

## 🧪 Testowanie

### Uruchomienie testów

```bash
# Wszystkie testy
python -m pytest tests/ -v

# Specyficzny test
python -m pytest tests/test_pipeline.py -v

# Z pokryciem kodu
python -m pytest tests/ --cov=src
```

### Testy unit

- `test_embedding_service.py` - Testy konwersji tekstu na wektory
- `test_vector_store.py` - Testy magazynu wektorowego
- `test_retriever.py` - Testy wyszukiwania dokumentów
- `test_pipeline.py` - Testy całego pipeline'u

## 📊 Dane przykładowe

Projekt zawiera dane przykładowe na następujące tematy:
- **Technologia** - AI, Machine Learning
- **Nauka** - Fizyka, Biologia
- **Historia** - Renesans, Rewolucja Przemysłowa
- **Geografia** - Afryka, Azja
- **Literatura** - Dante, Shakespeare
- **Sztuka** - Impresjonizm, Surrealizm

Wszystkie dane są przechowywane w pliku `data/sample_data.json`

## 🔧 Konfiguracja

### Zmiana modelu embeddingów

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline(
    embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2",
    generator_model="gpt2"
)
```

### Dostępne modele embeddingów
- `all-MiniLM-L6-v2` (szybki, 384-wymiarowy)
- `all-mpnet-base-v2` (dokładniejszy, 768-wymiarowy)
- `paraphrase-MiniLM-L6-v2` (parafrazy)

## 🐛 Troubleshooting

### Problem: Brak dostępu do modeli transformers

**Rozwiązanie:**
```bash
pip install --upgrade transformers sentence-transformers
```

### Problem: Błąd FAISS

**Rozwiązanie:**
```bash
pip install faiss-cpu  # lub faiss-gpu dla GPU
```

### Problem: Streamlit nie uruchamia się

**Rozwiązanie:**
```bash
streamlit run app.py --logger.level=debug
```

## 📝 Przykładowe pytania

- "Co to jest sztuczna inteligencja?"
- "Jakie są zastosowania machine learning?"
- "Opowiedz mi o Renesansie"
- "Czym zajmuje się biologia?"
- "Jakie są właściwości fizyki?"

## 🤝 Wkład

Aby przyczynić się do projektu:
1. Fork
2. Utwórz branch'a feature (`git checkout -b feature/AmazingFeature`)
3. Commit zmian (`git commit -m 'Add some AmazingFeature'`)
4. Push do branch'a (`git push origin feature/AmazingFeature`)
5. Otwórz Pull Request

## 📄 Licencja

Ten projekt jest dostępny pod licencją MIT.

## 🙏 Podziękowania

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)

## 📞 Kontakt

Pytania lub sugestie? Stwórz issue w repozytorium.

---

**Ostatnia aktualizacja:** 21 marca 2026
