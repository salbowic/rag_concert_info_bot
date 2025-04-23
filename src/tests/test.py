import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.functions import ingest_document, retrieve_documents, generate_response, classify_input
from utils.vectorstore_setup import vectorstore

# Load documents and queries from a text file
def load_data_from_file(filepath: str):
    documents = []
    queries = []
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("### Example Documents:"):
                current_section = "documents"
            elif line.startswith("### Example Questions:"):
                current_section = "queries"
            elif current_section == "documents" and line:
                documents.append(line)
            elif current_section == "queries" and line:
                queries.append(line)
    return documents, queries

def test_system(documents, queries):
    print("Ingesting documents...")
    for doc in documents:
        result = ingest_document(doc, vectorstore)
        print(result)
    
    print("\nTesting queries...")
    for query in queries:
        print(f"\nQuery: {query}")
        # Retrieve relevant documents
        retrieved_docs = retrieve_documents(query, vectorstore)
        if retrieved_docs:
            # Generate a response
            response = generate_response(retrieved_docs, query)
            print(f"Response: {response}")
        else:
            print("No relevant documents found.")

if __name__ == "__main__":
    documents, queries = load_data_from_file("./tests/test_data.txt")
    test_system(documents, queries)