from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Validate environment variables
if not os.getenv('LANGCHAIN_API_KEY'):
    raise ValueError("LANGCHAIN_API_KEY is not set in the environment variables.")
if not os.getenv('GROQ_API_KEY'):
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

# Initialize global components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./../chroma_persist",
    embedding_function=embeddings,
    collection_name="concert_tour_collection"
)
