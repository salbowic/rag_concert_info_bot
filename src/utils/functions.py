from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

def _is_relevant_document(document: str) -> bool:
    """
    Check if a document is relevant to the concert tour domain.

    Args:
        document (str): The document content.

    Returns:
        bool: True if the document is relevant, False otherwise.
    """
    keywords = ["concert", "tour", "venue", "performer", "schedule", "logistics", "artist", "band", "event", "performance", "dates", "location"]
    return any(keyword in document.lower() for keyword in keywords)

def _generate_summary(document: str) -> str:
    """
    Generate a structured summary of the document.

    Args:
        document (str): The document content.

    Returns:
        str: A structured summary of the document.
    """
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant that summarizes documents related to concert tours. 
    Your task is to extract key information from the from the document passed by the user and present it in a structured format with the following fields:
    - Date(s): Extract any specific dates or years mentioned in the document. If only a year is provided, include it.
    - Performer(s): Identify the name(s) of the artist(s) or band(s) mentioned in the document.
    - Location(s): List all cities or countries mentioned in the document where events will take place.
    - Venue(s): Extract the names of venues mentioned in the document.
    - Logistical  5262 notes: Summarize any logistical details, such as collaborations, merchandise, or special arrangements.
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {document}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=500,
        timeout=None,
        max_retries=1,
    )
    chain = prompt | llm
    ai_message = chain.invoke({"document": document})
    return ai_message.content if hasattr(ai_message, "content") else str(ai_message)

def ingest_document(document: str, vectorstore):
    """
    Ingest a document into the RAG system.

    Args:
        document (str): The document content.
        vectorstore: The vectorstore instance for storing documents.

    Returns:
        str: A message indicating the result of the ingestion process.
    """
    if not _is_relevant_document(document):
        return "Sorry, I cannot ingest documents with other themes."

    summary = _generate_summary(document)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(summary)
    documents = [
        Document(page_content=chunk, metadata={"source": "user_input"})
        for chunk in splits
    ]
    vectorstore.add_documents(documents)
    return f"Thank you for sharing! Your document has been successfully added to the database. Here is a brief summary of the data from the document:\n{summary}"

def retrieve_documents(query: str, vectorstore):
    """
    Retrieve documents from the vectorstore based on a query.

    Args:
        query (str): The search query.
        vectorstore: The vectorstore instance for retrieving documents.

    Returns:
        list: A list of retrieved documents.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever.invoke(query)

def generate_response(docs, question: str):
    """
    Generate a response based on retrieved documents and a user question.

    Args:
        docs (list): The retrieved documents.
        question (str): The user's question.

    Returns:
        str: The generated response.
    """
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant that answers the questions passed by the user using the given context.
    You will generate a precise and informative answer and nothing more.
    Do not include any other sentences or explanations then the plain answer.
    Here is the context:
    {context}
    
    Use the context and do not tell the user that you have some context provided.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=400,
        timeout=None,
        max_retries=2,
    )
    chain = prompt | llm
    response = chain.invoke({"context": docs, "question": question})
    return response.content if hasattr(response, "content") else str(response)

def classify_input(user_input: str) -> str:
    """
    Classify the user input as either a 'document' or a 'question'.

    Args:
        user_input (str): The input provided by the user.

    Returns:
        str: 'document' if the input is a document ingestion request, 'question' if it's a query.
    """
    
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant that classifies the user input as either a 'document' or a 'question'.
    You will return only the classification result and nothing more.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    "Please, add this document to your database: The 2025–2026 concert tour will feature performances by Taylor Swift."
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    document
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    "Where is Lady Gaga planning to give concerts during autumn 2025?"
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    question
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_input}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=10,
        timeout=None,
        max_retries=2,
    )
    
    chain = prompt | llm
    response = chain.invoke({"user_input": user_input})
    
    if "document" in str(response).lower():
        return "document"
    elif "question" in str(response).lower():
        return "question"
    return "unknown"