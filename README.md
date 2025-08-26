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

---

## 📂 Project Structure
app.py # Main Streamlit app
web-search-llm-db/ # Persistent ChromaDB storage (auto-created)

yaml
Copy
Edit

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/llm-web-search.git
cd llm-web-search
2. Create Virtual Environment (Optional but Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
📦 Requirements
Here are the main libraries used:

streamlit

ollama

ddgs

crawl4ai

chromadb

langchain

sentence-transformers

Note: You also need Ollama installed locally and the llama3.2:3b model pulled:

bash
Copy
Edit
ollama pull llama3.2:3b
▶️ Usage
Run the app with:

bash
Copy
Edit
streamlit run app.py
Workflow:
Enter a query in the text area.

Toggle Enable web search if you want fresh results from the internet.

Press ⚡ Go to get the LLM’s detailed, context-based response.

📸 Example
Input: What are the latest advancements in quantum computing?

If web search is enabled → The system fetches fresh data from the web, crawls pages, stores relevant chunks in ChromaDB, and provides a context-based response.

If disabled → The LLM responds without external context.

⚖️ License
This project is licensed under the MIT License.

🤝 Contributing
Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to change.
