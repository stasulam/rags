"""
Data Loader - ładowanie i przygotowywanie danych
"""

import json
import os
from typing import List, Dict


class DataLoader:
    @staticmethod
    def load_sample_data() -> List[Dict]:
        """
        Zwraca przykładowe dane o różnych tematach

        Returns:
            Lista słowników z danymi
        """
        sample_data = [
            {
                "topic": "Technologia",
                "content": "Sztuczna inteligencja to dziedzina nauki zajmująca się tworzeniem maszyn i systemów, które mogą wykonywać zadania wymagające ludzkiej inteligencji. Ki jest używana w wielu aplikacjach takich jak systemy rekomendacyjne, chatboty i analiza danych.",
            },
            {
                "topic": "Technologia",
                "content": "Machine Learning to poddziedzina SI, która pozwala komputerom uczyć się na podstawie danych bez jawnego programowania. Algorytmy ML znajdują zastosowanie w klasyfikacji, regresji i grupowaniu danych.",
            },
            {
                "topic": "Nauka",
                "content": "Fizyka jest nauką zajmującą się badaniem właściwości materii i energii. Dzieli się na wiele gałęzi takich jak mechanika, termonika czy elektromagnetyka. Prawa fizyki opisują sposób funcionowania wszechświata.",
            },
            {
                "topic": "Nauka",
                "content": "Biologia to nauka o życiu i organizmach żywych. Obejmuje wiele dziedzin takich jak genetyka, mikrobiologia czy ekologia. Zrozumienie biologii jest kluczowe dla medycyny i ochrony środowiska.",
            },
            {
                "topic": "Historia",
                "content": "Renesans był okresem kulturalnym i intelektualnym w Europie w 14-17 wieku. Charakteryzował się odrodzeniem zainteresowania nauką, sztuką i humanizmem. Ważne postacie to Leonardo da Vinci, Mikołaj Kopernik i William Shakespeare.",
            },
            {
                "topic": "Historia",
                "content": "Rewolucja Przemysłowa zmieniła społeczeństwo w 18-19 wieku. Wprowadzenie maszyn i automatyzacji umożliwiło masową produkcję. To doprowadziło do zaobserwalizacji społecznych, urbanizacji i zmian w strukturze pracy.",
            },
            {
                "topic": "Geografia",
                "content": "Afryka jest drugim co do wielkości kontynentem. Posiada dużą różnorodność geograficzną - od pustyń, przez sawanny, do lasów tropikalnych. Kontynent jest domem dla ponad miliarda ludzi i bogatymi zasobami naturalnymi.",
            },
            {
                "topic": "Geografia",
                "content": "Azja jest największym kontynentem pod względem powierzchni i populacji. Obejmuje wiele krajów o różnych kulturach, tradycjach i systemach politycznych. Od gór Himalajów po tropikalne wyspy Azji Południowo-Wschodniej.",
            },
            {
                "topic": "Literatura",
                "content": "Dante Alighieri był włoskim poetą, autorem słynnego dzieła 'Boska Komedia'. Utwór opisuje podróż przez Piekło, Czyściec i Raj. Jest uważany za jedno z najważniejszych dzieł literatury światowej.",
            },
            {
                "topic": "Literatura",
                "content": "William Shakespeare był angielskim dramaturem i poetą z przełomu 16-17 wieku. Napisał około 37 sztuk teatralnych i 154 sonety. Jego dzieła takie jak 'Hamlet' czy 'Romeo i Julia' są popularne do dzisiaj.",
            },
            {
                "topic": "Sztuka",
                "content": "Impressjonizm był ruchem artystycznym w malarstwie, który powstał w Francji w 19 wieku. Artyści tacy jak Claude Monet i Pierre-Auguste Renoir fokusowali się na przedstawieniu światła i koloru. Charakteryzował się szybkimi pociągnięciami pędzla.",
            },
            {
                "topic": "Sztuka",
                "content": "Surrealізм to ruch sztuki i literatury, który powstał w 20 wieku. Artyści tacy jak Salvador Dalí badali podświadomość i sny. Dzieła charakteryzują się fantazyjnymi, irracjonalnymi obrazami.",
            },
        ]

        return sample_data

    @staticmethod
    def save_data(data: List[Dict], filepath: str) -> None:
        """
        Zapisuje dane do pliku JSON

        Args:
            data: Dane do zapisania
            filepath: Ścieżka do pliku
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_data(filepath: str) -> List[Dict]:
        """
        Ładuje dane z pliku JSON

        Args:
            filepath: Ścieżka do pliku

        Returns:
            Załadowane dane
        """
        if not os.path.exists(filepath):
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def format_documents(data: List[Dict]) -> tuple:
        """
        Formatuje dane do użytku w RAG pipeline

        Args:
            data: Lista słowników z danymi

        Returns:
            Tuple (dokumenty, metadane)
        """
        documents = []
        metadata = []

        for i, item in enumerate(data):
            documents.append(item.get("content", ""))
            metadata.append(
                {
                    "id": i,
                    "topic": item.get("topic", "Unknown"),
                    "source": "sample_data",
                }
            )

        return documents, metadata
