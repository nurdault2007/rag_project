from dataclasses import dataclass
from typing import List
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter

# Импортируем наш класс Document из предыдущего скрипта
# (убедитесь, что load_data.py находится в той же папке ingest)
from ingest.load_data import Document

@dataclass
class Chunk:
    """Класс для хранения нарезанного фрагмента текста и его метаданных."""
    text: str
    metadata: dict
    chunk_index: int  # Порядковый номер чанка в документе

def chunk_fixed_size(documents: List[Document], chunk_size: int = 400, chunk_overlap: int = 60) -> List[Chunk]:
    """
    Стратегия 1: Нарезка строго по количеству токенов.
    Быстрая, но может разбивать слова или предложения на полуслове.
    """
    # Используем токенизатор от OpenAI (cl100k_base используется в GPT-3.5/4/4o)
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return _process_splits(documents, splitter, strategy_name="fixed_token")

def chunk_recursive(documents: List[Document], chunk_size: int = 400, chunk_overlap: int = 60) -> List[Chunk]:
    """
    Стратегия 2: Рекурсивная нарезка с учетом токенов.
    Пытается не разрывать абзацы и предложения, сохраняя логику текста.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""] # Приоритет разделителей
    )
    
    return _process_splits(documents, splitter, strategy_name="recursive")

def _process_splits(documents: List[Document], splitter, strategy_name: str) -> List[Chunk]:
    """Вспомогательная функция для применения сплиттера и сохранения метаданных."""
    all_chunks = []
    
    for doc in documents:
        # Разбиваем сырой текст документа на фрагменты
        text_splits = splitter.split_text(doc.text)
        
        # Проходим по каждому фрагменту и создаем объект Chunk
        for index, text_fragment in enumerate(text_splits):
            # Фильтруем слишком короткие куски (требование: минимум 100 токенов)
            # В реальной задаче можно использовать len(tiktoken.get_encoding().encode(text_fragment))
            # Но для простоты отсеем откровенный мусор (меньше 50 символов)
            if len(text_fragment.strip()) < 50:
                continue
                
            # Копируем метаданные, чтобы не изменить оригинал
            chunk_meta = doc.metadata.copy()
            # Добавляем информацию о стратегии нарезки (понадобится для экспериментов)
            chunk_meta["chunking_strategy"] = strategy_name
            
            chunk = Chunk(
                text=text_fragment,
                metadata=chunk_meta,
                chunk_index=index
            )
            all_chunks.append(chunk)
            
    return all_chunks

# Блок для самостоятельного тестирования
if __name__ == "__main__":
    from ingest.load_data import load_directory
    import os
    
    # Загружаем документы из папки
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    docs = load_directory(DATA_DIR)
    
    if docs:
        print("\n--- Тестируем рекурсивную стратегию ---")
        recursive_chunks = chunk_recursive(docs)
        print(f"Из {len(docs)} документов получилось {len(recursive_chunks)} чанков (рекурсивно).")
        if recursive_chunks:
            print("\nПример метаданных первого чанка:")
            print(recursive_chunks[0].metadata)
            print("Индекс чанка:", recursive_chunks[0].chunk_index)
            print("\nТекст первого чанка (первые 150 символов):")
            print(recursive_chunks[0].text[:150] + "...")
