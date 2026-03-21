"""
Główny interfejs aplikacji Streamlit
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Dodanie źródła do ścieżki
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RAGPipeline
from src.data_loader import DataLoader


def initialize_session():
    """Inicjalizuje zmienne sessji"""
    if "pipeline" not in st.session_state:
        with st.spinner("Inicjalizowanie RAG pipeline..."):
            st.session_state.pipeline = RAGPipeline()
            st.session_state.pipeline.initialize_with_sample_data()
            st.session_state.query_history = []


def display_header():
    """Wyświetla nagłówek aplikacji"""
    st.set_page_config(
        page_title="RAG Pipeline",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 RAG Pipeline - Zaawansowane Wyszukiwanie i Generowanie")
    st.markdown("""
    Aplikacja implementująca **Retrieval-Augmented Generation** - system łączący wyszukiwanie dokumentów
    z generowaniem odpowiedzi na naturalne pytania.
    """)


def display_sidebar():
    """Wyświetla panel boczny"""
    with st.sidebar:
        st.header("⚙️ Ustawienia")
        
        # Liczba dokumentów do wyszukania
        top_k = st.slider(
            "Liczba dokumentów do wyszukania",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
        
        # Rozmiar magazynu
        st.divider()
        st.subheader("📊 Statystyki")
        pipeline = st.session_state.pipeline
        store_size = pipeline.get_vector_store_size()
        st.metric("Liczba dokumentów", store_size)
        
        # Historia pytań
        st.divider()
        st.subheader("📜 Historia")
        if st.session_state.query_history:
            for i, query in enumerate(st.session_state.query_history[-10:], 1):
                with st.expander(f"Pytanie {i}: {query[:50]}..."):
                    st.write(query)
        else:
            st.info("Brak historii pytań")
        
        return top_k


def display_main_interface(top_k: int):
    """Wyświetla główny interfejs"""
    # Sekcja wejścia
    st.header("❓ Zadaj pytanie")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Wpisz swoje pytanie:",
            placeholder="Np. Co to jest sztuczna inteligencja?",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 Szukaj", use_container_width=True)
    
    # Przetwarzanie pytania
    if search_button and question:
        st.session_state.query_history.append(question)
        
        with st.spinner("Wyszukuję dokumenty i generuję odpowiedź..."):
            result = st.session_state.pipeline.query(question, top_k=top_k)
        
        display_results(result)
    
    # Przykładowe pytania
    st.divider()
    st.subheader("💡 Przykładowe pytania:")
    
    example_questions = [
        "Co to jest sztuczna inteligencja?",
        "Jakie są zastosowania machine learning?",
        "Opowiedz mi o Renesansie",
        "Czym zajmuje się biologia?"
    ]
    
    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]
    
    for idx, example in enumerate(example_questions):
        with columns[idx % 4]:
            if st.button(example, key=f"example_{idx}", use_container_width=True):
                st.session_state.query_history.append(example)
                with st.spinner("Wyszukuję dokumenty i generuję odpowiedź..."):
                    result = st.session_state.pipeline.query(example, top_k=top_k)
                st.rerun()


def display_results(result: dict):
    """Wyświetla wyniki wyszukiwania i generowania"""
    # Wyświetl odpowiedź
    st.success("✅ Odpowiedź wygenerowana!")
    
    with st.expander("📝 Pełna odpowiedź", expanded=True):
        st.write(result["answer"])
    
    # Wyświetl źródła
    st.subheader(f"📚 Znalezione źródła ({result['num_sources']})")
    
    for i, source in enumerate(result["sources"], 1):
        with st.expander(
            f"Źródło {i} - {source['metadata'].get('topic', 'Nieznany temat')} "
            f"(Score: {source['score']:.4f})"
        ):
            st.write(source["content"])
            
            # Metadane
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"**Temat:** {source['metadata'].get('topic', 'N/A')}")
            with col2:
                st.caption(f"**Podobieństwo:** {(1 - source['score'] / 2) * 100:.1f}%")


def display_about_section():
    """Wyświetla sekcję 'O projekcie'"""
    st.divider()
    st.header("ℹ️ O projekcie")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏗️ Komponenty")
        st.write("""
        - **Embedding Service**: Konwersja tekstu na wektory
        - **Vector Store**: Magazyn dokumentów z FAISS
        - **Retriever**: Wyszukiwanie podobnych dokumentów
        - **Generator**: Generowanie odpowiedzi
        """)
    
    with col2:
        st.subheader("📊 Technologie")
        st.write("""
        - Sentence Transformers
        - FAISS (Facebook AI Similarity Search)
        - Hugging Face Transformers
        - Streamlit
        """)


def main():
    """Główna funkcja aplikacji"""
    display_header()
    initialize_session()
    top_k = display_sidebar()
    display_main_interface(top_k)
    display_about_section()


if __name__ == "__main__":
    main()
