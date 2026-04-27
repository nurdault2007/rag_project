import os
import google.generativeai as genai
from dotenv import load_dotenv

# Загружаем ключ
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Доступные модели для генерации текста:")
# Запрашиваем у Google список всех моделей
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
