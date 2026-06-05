# Local Engineering RAG Chatbot

This is a Django + React/Vite RAG chatbot for engineering documents.

## Start Backend

```powershell
cd "D:\rag chatbot"
.\.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8012 --noreload
```

## Start Frontend

```powershell
cd "D:\rag chatbot\frontend"
npm run dev
```

Open `http://127.0.0.1:5173`.

The ingestion pipeline now performs structure-aware extraction, noise cleaning, heading hierarchy detection, semantic chunking, table preservation, metadata enrichment, embeddings, hybrid retrieval, reranking, and local LLM answering through `llama-cpp-python`.
