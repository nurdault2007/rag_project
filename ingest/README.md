# Custom RAG Chatbot with Source Citations

A production-ready Retrieval-Augmented Generation (RAG) system that allows users to chat with a custom knowledge base. This system extracts information from PDF and TXT documents, indexes them in a vector database, and generates grounded responses using a Large Language Model (LLM) with mandatory source citations.

## 🚀 Features
- **Multi-format Support:** Ingests data from PDF and TXT files.
- **Metadata Tracking:** Preserves source names, titles, and dates for precise citations.
- **Advanced Chunking:** Implements multiple text-splitting strategies for optimal retrieval.
- **Grounded Generation:** System prompts ensure the LLM only answers based on provided context.
- **Citations:** Every answer includes references to the source documents.

## 📁 Project Structure
```text
├── data/
│   ├── raw/               # Drop your PDF and TXT files here
│   └── processed/         # Chunks and vector index storage
├── ingest/
│   └── load_data.py       # Document ingestion and metadata extraction
├── retrieval/             # Vector DB setup and search logic
├── generation/            # LLM prompt templates and API calls
├── ui/                    # Chat interface (Streamlit/Gradio)
├── eval/                  # Evaluation scripts and datasets
├── .env                   # Environment variables (API keys)
├── requirements.txt       # Project dependencies
└── README.md
