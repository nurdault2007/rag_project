import os
import google.generativeai as genai
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class GeminiGenerator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY не найден в файле .env")
        
        genai.configure(api_key=api_key)
        # Используем новейшую модель gemini-1.5-flash (она быстрая и точная)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def generate_answer(self, query: str, search_results: dict) -> str:
        """Формирует промпт и генерирует ответ на основе контекста."""
        
        # 1. Собираем контекст из результатов поиска
        context_blocks = []
        for i in range(len(search_results['documents'][0])):
            text = search_results['documents'][0][i]
            meta = search_results['metadatas'][0][i]
            source = meta.get('source', 'Unknown')
            context_blocks.append(f"SOURCE: {source}\nCONTENT: {text}")
        
        full_context = "\n\n---\n\n".join(context_blocks)

        # 2. Строгий системный промпт (Требование проекта!)
        system_instructions = (
            "You are a helpful assistant. Use ONLY the provided context to answer the user's question. "
            "If the answer is not in the context, say: 'I cannot find this in the provided documents.' "
            "For every fact you state, you MUST mention the source filename (e.g., ). "
            "Do not use your own knowledge outside of the context."
        )

        prompt = f"""
{system_instructions}

CONTEXT:
{full_context}

USER QUESTION: 
{query}

ANSWER:
"""
        
        # 3. Генерация
        response = self.model.generate_content(prompt)
        return response.text

# Блок для интеграционного теста
if __name__ == "__main__":
    from retrieval.vector_store import VectorStore
    
    # Инициализируем базу и генератор
    vector_db = VectorStore()
    generator = GeminiGenerator()
    
    # 1. Поиск (Retrieval)
    query = "What is the main idea of attention mechanism?"
    results = vector_db.search(query, top_k=3)
    
    # 2. Генерация (Generation)
    print("\n--- Генерируем ответ через Gemini ---")
    answer = generator.generate_answer(query, results)
    print(answer)
