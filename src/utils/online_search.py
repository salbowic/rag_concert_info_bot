from serpapi import GoogleSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging
from dotenv import load_dotenv
import os

# Load and validate environment variables from .env file
load_dotenv()
if not os.getenv('SERP_API_KEY'):
    raise ValueError("SERP_API_KEY is not set in the environment variables.")

def perform_web_search(artist_name: str):
    """
    Perform a web search to retrieve details about an artist's upcoming concerts using SerpAPI.
    
    Args:
        artist_name (str): The name of the artist to search for.

    Returns:
        list: A list of relevant documents retrieved from the search results.
    """
    
    try:
        search_params = {
            "q": f"{artist_name} upcoming concerts",
            "hl": "en",
            "gl": "us",
            "api_key": os.getenv('SERP_API_KEY')
        }

        search = GoogleSearch(search_params)
        results = search.get_dict()

        if "organic_results" not in results or not results["organic_results"]:
            logging.warning("No search results found.")
            return []

        docs = [result["snippet"] for result in results["organic_results"] if "snippet" in result]

        if not docs:
            logging.warning("No relevant snippets were extracted from the search results.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(" ".join(docs))

        if not splits:
            logging.warning("No text chunks were created from the search results.")
            return []

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        web_vectorstore = Chroma.from_texts(
            texts=splits,
            embedding=embeddings,
            persist_directory="./../web_chroma_persist"
        )

        retriever = web_vectorstore.as_retriever(search_kwargs={"k": 8})
        relevant_docs = retriever.get_relevant_documents(artist_name)

        if not relevant_docs:
            logging.warning("No relevant documents were retrieved.")
            return []

        return relevant_docs

    except Exception as e:
        logging.error(f"An error occurred during the web search: {e}")
        return []
