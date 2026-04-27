import os
from typing import List
import chromadb
from chromadb.utils import embedding_functions

# Импортируем наш класс Chunk из модуля нарезки
from ingest.chunking import Chunk

# Путь, где будет физически храниться наша база данных на диске
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector_db")

class VectorStore:
    def __init__(self, collection_name: str = "rag_collection"):
        """Инициализация базы данных и модели эмбеддингов."""
        os.makedirs(DB_DIR, exist_ok=True)
        
        # PersistentClient сохраняет базу в папку DB_DIR
        self.client = chromadb.PersistentClient(path=DB_DIR)
        
        # Загружаем локальную модель для векторизации текста
        # При первом запуске она скачается (около 80 МБ)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Получаем или создаем коллекцию (как таблица в SQL)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_chunks(self, chunks: List[Chunk]):
        """Переводит текст чанков в векторы и сохраняет их в базу."""
        if not chunks:
            print("Нет чанков для добавления.")
            return

        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            documents.append(chunk.text)
            metadatas.append(chunk.metadata)
            # Генерируем уникальный ID для каждого куска
            # Формат: имяфайла_стратегия_индекс
            safe_source = chunk.metadata.get('source', 'doc').replace(" ", "_")
            strategy = chunk.metadata.get('chunking_strategy', 'unk')
            chunk_id = f"{safe_source}_{strategy}_{chunk.chunk_index}"
            ids.append(chunk_id)

        print(f"Векторизация и добавление {len(documents)} чанков в ChromaDB...")
        
        # ChromaDB сама вызовет embedding_fn для массива documents
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print("Успешно добавлено!")

    def search(self, query: str, top_k: int = 5):
        """Ищет наиболее подходящие чанки по смыслу (семантический поиск)."""
        print(f"\nИщем по запросу: '{query}'")
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results

# Блок для проверки
if __name__ == "__main__":
    from ingest.load_data import load_directory
    from ingest.chunking import chunk_recursive
    
    # 1. Загружаем и режем документы
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    docs = load_directory(DATA_DIR)
    chunks = chunk_recursive(docs)
    
    # 2. Кладем в базу
    vector_db = VectorStore()
    vector_db.add_chunks(chunks)
    
    # 3. Делаем тестовый семантический поиск!
    # Замените этот вопрос на любой, который связан с вашими документами
    test_query = "What is attention mechanism?" 
    
    search_results = vector_db.search(query=test_query, top_k=3)
    
    print("\n--- Результаты поиска (Топ 3) ---")
    for i in range(len(search_results['documents'][0])):
        text = search_results['documents'][0][i]
        meta = search_results['metadatas'][0][i]
        distance = search_results['distances'][0][i] # Насколько близок вектор (чем меньше, тем лучше)
        
        print(f"\n[Результат {i+1}] Из файла: {meta['source']} (Дистанция: {distance:.4f})")
        print(f"Текст: {text[:150]}...")
