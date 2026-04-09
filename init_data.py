"""
Plik do inicjalizacji danych przykładowych
"""

from src.data_loader import DataLoader
import os


def initialize_sample_data():
    """Pobiera i zapisuje przykładowe dane"""
    data_loader = DataLoader()

    # Pobranie danych
    sample_data = data_loader.load_sample_data()

    # Zapis do pliku
    data_path = "data/sample_data.json"
    data_loader.save_data(sample_data, data_path)

    print(f"✅ Dane zostały zapisane do {data_path}")
    print(f"📊 Liczba dokumentów: {len(sample_data)}")

    # Wyswietlenie statystyk
    topics = {}
    for item in sample_data:
        topic = item.get("topic", "Unknown")
        topics[topic] = topics.get(topic, 0) + 1

    print("\n📚 Rozkład tematów:")
    for topic, count in sorted(topics.items()):
        print(f"   - {topic}: {count} dokumentów")


if __name__ == "__main__":
    initialize_sample_data()
