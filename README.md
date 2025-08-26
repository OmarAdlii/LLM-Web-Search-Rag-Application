# 🔍 LLM Web Search with Streamlit

This project integrates **LLMs (Ollama’s LLaMA 3.2)** with **web search, crawling, and vector databases** to provide detailed, context-based answers.  
The system fetches web results, filters and cleans content, stores it in a vector database (**ChromaDB**), and queries it to give precise, context-aware responses through a **Streamlit interface**.

---

## ✨ Features
- ✅ Web search using [DuckDuckGo Search API (`ddgs`)](https://pypi.org/project/ddgs/)  
- ✅ Website crawling with [`crawl4ai`](https://github.com/unclecode/crawl4ai)  
- ✅ Content filtering (BM25 relevance-based)  
- ✅ Context storage & retrieval with [ChromaDB](https://www.trychroma.com/)  
- ✅ LLM-powered answers using [Ollama](https://ollama.com/) (`llama3.2:3b` model)  
- ✅ Streamlit interface for interactive queries  
- ✅ Robots.txt respect (avoids crawling disallowed pages)  

