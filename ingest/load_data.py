import os
import datetime
import PyPDF2
from dataclasses import dataclass
from typing import List

# Создаем структуру данных для хранения текста и метаданных вместе
@dataclass
class Document:
    text: str
    metadata: dict

def get_file_creation_date(filepath: str) -> str:
    """Вспомогательная функция для получения даты создания файла из ОС."""
    timestamp = os.path.getctime(filepath)
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

def load_txt(filepath: str) -> Document:
    """Загружает обычный текстовый файл и формирует метаданные."""
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    filename = os.path.basename(filepath)
    # Для TXT файлов заголовком делаем имя файла (без расширения)
    title = os.path.splitext(filename)[0] 
    date = get_file_creation_date(filepath)
    
    metadata = {
        "source": filename,
        "title": title,
        "date": date,
        "file_type": "txt"
    }
    
    return Document(text=text, metadata=metadata)

def load_pdf(filepath: str) -> Document:
    """Загружает PDF файл, извлекает текст и доступные метаданные (заголовок, дату)."""
    text = ""
    filename = os.path.basename(filepath)
    title = os.path.splitext(filename)[0]
    date = get_file_creation_date(filepath)
    
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        # Пытаемся достать встроенные метаданные PDF
        pdf_meta = reader.metadata
        if pdf_meta:
            if pdf_meta.title:
                title = pdf_meta.title
            # Даты в PDF часто имеют сложный формат, поэтому берем ОС-дату как фоллбэк,
            # но если хотите заморочиться, можно парсить pdf_meta.creation_date
        
        # Считываем текст со всех страниц
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
                
    metadata = {
        "source": filename,
        "title": title,
        "date": date,
        "file_type": "pdf"
    }
    
    return Document(text=text, metadata=metadata)

def load_directory(directory_path: str) -> List[Document]:
    """
    Проходит по указанной директории, определяет тип файлов 
    и использует соответствующий загрузчик.
    """
    documents = []
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Директория {directory_path} не найдена.")
        
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        
        # Пропускаем папки, работаем только с файлами
        if os.path.isdir(filepath):
            continue
            
        ext = os.path.splitext(filename)[1].lower()
        
        try:
            if ext == '.txt':
                doc = load_txt(filepath)
                documents.append(doc)
                print(f"Успешно загружен TXT: {filename}")
            elif ext == '.pdf':
                doc = load_pdf(filepath)
                documents.append(doc)
                print(f"Успешно загружен PDF: {filename}")
            else:
                print(f"Пропущен файл неподдерживаемого формата: {filename}")
        except Exception as e:
            print(f"Ошибка при загрузке {filename}: {e}")
            
    return documents

# Блок для самостоятельного тестирования скрипта
if __name__ == "__main__":
    # Создайте папку data/raw в корне проекта и положите туда пару файлов для теста
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    
    # Создаем папку, если ее нет
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"Ищем файлы в: {DATA_DIR}")
    docs = load_directory(DATA_DIR)
    
    print(f"\nВсего загружено документов: {len(docs)}")
    if docs:
        print("\nПример метаданных первого документа:")
        print(docs[0].metadata)
        print("\nПервые 100 символов текста:")
        print(docs[0].text[:100] + "...")
