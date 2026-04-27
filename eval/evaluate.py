import json
import os
import sys
import time
import google.generativeai as genai
from tqdm import tqdm

# Добавляем пути
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval.vector_store import VectorStore
from generation.generator import GeminiGenerator

def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        if filepath.endswith('.json'):
            try:
                return json.load(f)
            except:
                f.seek(0)
                return [json.loads(line) for line in f]

class RAGEvaluator:
    def __init__(self):
        self.vector_db = VectorStore()
        self.generator = GeminiGenerator()
        # Для оценки (судейства) используем классическую gemini-pro (она стабильнее)
        self.judge_model = genai.GenerativeModel('gemini-pro')

    def evaluate_answer(self, question, expected_answer, generated_answer):
        """Использует LLM для оценки схожести сгенерированного ответа с эталонным."""
        prompt = f"""
        You are an impartial judge evaluating a RAG system.
        Question: {question}
        Expected Answer: {expected_answer}
        Generated Answer: {generated_answer}
        
        Does the Generated Answer contain the same core information as the Expected Answer? 
        Answer strictly with '1' if Yes, or '0' if No. Do not write anything else.
        """
        try:
            response = self.judge_model.generate_content(prompt).text.strip()
            return 1.0 if '1' in response else 0.0
        except Exception:
            return 0.0

    def run_evaluation(self, dataset, top_k=3):
        total_score = 0
        results = []
        
        # Берем только первые 10 вопросов для экспериментов, чтобы не биться в лимиты API
        test_subset = dataset[:10] 
        
        print(f"Начинаем оценку {len(test_subset)} вопросов (Top-k: {top_k})...")
        
        for item in tqdm(test_subset):
            question = item['question']
            expected = item['answer']
            
            while True:
                try:
                    # 1. Поиск
                    retrieved = self.vector_db.search(question, top_k=top_k)
                    
                    # 2. Генерация
                    generated = self.generator.generate_answer(question, retrieved)
                    
                    # 3. Оценка
                    score = self.evaluate_answer(question, expected, generated)
                    
                    total_score += score
                    results.append({"question": question, "score": score})
                    
                    # Пауза в 5 секунд, чтобы API не ругался
                    time.sleep(5)
                    break 
                    
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "Quota" in error_msg or "exhausted" in error_msg.lower():
                        print(f"\n[!] Сработал лимит Google API. Ждем 60 секунд...")
                        time.sleep(60)
                    else:
                        print(f"\n[!] Ошибка (пропускаем): {error_msg}")
                        break
            
        accuracy = total_score / len(test_subset)
        print(f"\n✅ Итоговая точность (Answer Relevance): {accuracy * 100:.1f}%")
        return accuracy

if __name__ == "__main__":
    DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "qadata.json")
    
    evaluator = RAGEvaluator()
    dataset = load_dataset(DATASET_PATH)
    
    # Запуск базового эксперимента
    evaluator.run_evaluation(dataset, top_k=3)
