import os
import ollama
import streamlit as st
from ddgs import DDGS
import asyncio
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult
import chromadb
import tempfile
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context.
Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

Context will be passed as "Context:"
User question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. When the context supports an answer, ensure your response is clear, concise, and directly addresses the question.
5. When there is no context, just say you have no context and stop immediately.
6. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.
7. Avoid explaining why you cannot answer or speculating about missing details. Simply state that you lack sufficient context when necessary.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.
6. Do not mention what you received in context, just focus on answering based on the context.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def call_llm(prompt: str, with_context: bool = True, context: str | None = None):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Context: {context}, Question: {prompt}",
        },
    ]

    if not with_context:
        messages.pop(0)
        messages[0]["content"] = prompt

    response = ollama.chat(model="llama3.2:3b", stream=True, messages=messages)

    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

def get_vector_collection() -> tuple[chromadb.Collection, chromadb.Client]:
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chroma_client = chromadb.PersistentClient(
        path="./web-search-llm-db", 
        settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_or_create_collection(
        name="web_llm",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    return (
        chroma_client.get_or_create_collection(
            name="web_llm",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        ),
        chroma_client,)


def normalize_url(url):
    normalized_url = (
        url.replace("https://", "")
        .replace("www.", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )
    print("Normalized URL", normalized_url)
    return normalized_url
  
def add_to_vector_database(results: list[CrawlResult]):
    collection, _ = get_vector_collection()

    for result in results:
        documents, metadatas, ids = [], [], []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        markdown_result = None
        if hasattr(result, "markdown_v2") and getattr(result.markdown_v2, "fit_markdown", None):
            markdown_result = result.markdown_v2.fit_markdown
        elif hasattr(result, "markdown") and result.markdown:
            markdown_result = result.markdown

        if not markdown_result:
            continue

        temp_file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        temp_file.write(markdown_result)
        temp_file.flush()

        loader = UnstructuredMarkdownLoader(temp_file.name, mode="single")
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        os.unlink(temp_file.name)  # Delete the temporary file

        normalized_url = normalize_url(result.url)

        if all_splits:
            for idx, split in enumerate(all_splits):
                documents.append(split.page_content)
                metadatas.append({"source": result.url})
                ids.append(f"{normalized_url}_{idx}")

            print("Upsert collection: ", id(collection))
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )


async def crawl_webpages(urls:list[str],prompt:str)->CrawlResult:
  bm25_filter=BM25ContentFilter(user_query=prompt,bm25_threshold=1.2)
  md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)
  crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "form", "img", "a"],
        only_text=True,
        exclude_social_media_links=True,
        keep_data_attributes=False,
        cache_mode=CacheMode.BYPASS,
        remove_overlay_elements=True,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        page_timeout=20000,  # in ms: 20 seconds
    )
  browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True)
  async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls, config=crawler_config)
        return results

def check_robots_txt(urls:list[str])->list[str]:
  allowed_urls=[]
  for url in urls:
    try:
      robots_url=f"{urlparse(url=url).scheme}://{urlparse(url).netloc}/robots.txt"
      rp = RobotFileParser(robots_url)
      rp.read()

      if rp.can_fetch("*", url):
        allowed_urls.append(url)
    except:
      allowed_urls.append(url)

  return allowed_urls



def get_web_urls(search_term:str,num_result:int=10)->list[str]:
  try:
    discard_urls=["youtube.com", "britannica.com", "vimeo.com"]
    for url in discard_urls:
      search_term += f' -site:{url}'
    
    results = list(DDGS().text(search_term, max_results=num_result))
    
    urls = [result['href'] for result in results if 'href' in result]
    #st.write("Extracted URLs:", urls)
    
    return check_robots_txt(urls)
  except Exception as e:
    print(f"Failed to fetch results from the web", str(e))
    st.write(f"Failed to fetch results from the web", str(e))
    st.stop()
  

async def run():
    st.set_page_config(page_title="LLM with Web Search")

    st.header("🔍 LLM Web Search")
    prompt = st.text_area(
        label="Put your query here",
        placeholder="Add your query...",
        label_visibility="hidden",
    )
    is_web_search = st.toggle("Enable web search", value=False, key="enable_web_search")
    go = st.button(
        "⚡️ Go",
    )

    collection, chroma_client = get_vector_collection()

    if prompt and go:
        if is_web_search:
            web_urls = get_web_urls(search_term=prompt)
            if not web_urls:
                st.write("No results found.")
                st.stop()

            results = await crawl_webpages(urls=web_urls, prompt=prompt)
            add_to_vector_database(results)

            qresults = collection.query(query_texts=[prompt], n_results=10)
            context = qresults.get("documents")[0]

            chroma_client.delete_collection(
                name="web_llm"
            )  # Delete collection after use

            llm_response = call_llm(
                context=context, prompt=prompt, with_context=is_web_search
            )
            st.write_stream(llm_response)
        else:
            llm_response = call_llm(prompt=prompt, with_context=is_web_search)
            st.write_stream(llm_response)


if __name__ == "__main__":
    asyncio.run(run())
