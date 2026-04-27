import streamlit as st
import sys
import os

# Добавляем корневую папку в пути Python, чтобы Streamlit видел наши модули
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.vector_store import VectorStore
from generation.generator import GeminiGenerator

# Настройка страницы
st.set_page_config(page_title="Мой RAG Чат-бот", page_icon="🤖")
st.title("📚 Умный чат-бот по документам")
st.markdown("Задайте вопрос по загруженным файлам (PDF, TXT). Бот ответит с указанием источника.")

# Инициализируем базу и генератор только один раз (чтобы не тормозило при каждом сообщении)
@st.cache_resource
def load_rag_components():
    db = VectorStore()
    gen = GeminiGenerator()
    return db, gen

vector_db, generator = load_rag_components()

# Создаем хранилище истории сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отрисовываем предыдущие сообщения на экране
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Строка ввода для пользователя
if prompt := st.chat_input("Напишите ваш вопрос здесь..."):
    # Добавляем вопрос пользователя в историю и на экран
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генерируем ответ бота
    with st.chat_message("assistant"):
        with st.spinner("Ищу информацию в документах..."):
            # 1. Ищем чанки в базе
            search_results = vector_db.search(prompt, top_k=3)
            
            # 2. Генерируем ответ
            answer = generator.generate_answer(prompt, search_results)
            
            # 3. Показываем на экране
            st.markdown(answer)
            
    # Сохраняем ответ бота в историю
    st.session_state.messages.append({"role": "assistant", "content": answer})
