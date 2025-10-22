from utils.functions import ingest_document, retrieve_documents, generate_response, classify_input
from utils.vectorstore_setup import vectorstore
from utils.online_search import perform_web_search


def run_cli():
    """
    Run the CLI-based application for the Concert Tour Information Bot.
    """
    print("Welcome to the Concert Tour Information Bot (CLI Mode)!")
    print("Type 'web()' to switch to Web Search mode.")
    print("Type 'exit()' to quit the application.\n")

    mode = "Core RAG Functionality"

    while True:
        # Display the current mode
        print(f"\n[Current Mode: {mode}]")
        user_input = input("Enter your query: ").strip()

        # Exit condition
        if user_input.lower() == "exit()":
            if mode == "Web Search":
                mode = "Core RAG Functionality"
                print("Switched back to Core RAG Functionality mode.")
            else:
                print("Goodbye!")
                break

        # Switch to Web Search mode
        elif user_input.lower() == "web()":
            mode = "Web Search"
            print("Switched to Web Search mode.")

        elif user_input:
            if mode == "Core RAG Functionality":
                classification = classify_input(user_input)

                if classification == "document":
                    # Handle document ingestion
                    result = ingest_document(user_input, vectorstore)
                    print(result)
                elif classification == "question":
                    # Handle question answering
                    docs = retrieve_documents(user_input, vectorstore)
                    response = generate_response(docs, user_input)
                    print("Response:\n", response)
                else:
                    print("Error: Unable to classify the input. Please try again.")
            elif mode == "Web Search":
                # Handle web search
                relevant_docs = perform_web_search(user_input)
                if relevant_docs:
                    response = generate_response(
                        docs=relevant_docs,
                        question=f"Show me details about the {user_input} upcoming concerts."
                    )
                    print("Response:\n", response)
                else:
                    print("Error: No relevant information found for the given artist.")
        else:
            print("Error: Please enter a valid query.")
