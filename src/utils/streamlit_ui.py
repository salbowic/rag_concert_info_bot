import streamlit as st
from utils.functions import ingest_document, retrieve_documents, generate_response, classify_input
from utils.vectorstore_setup import vectorstore
from utils.online_search import perform_web_search

def run_streamlit_ui():
    """
    Run the Streamlit UI for the Concert Tour Information Bot.
    This interface allows users to interact with the bot in two modes:
    1. Core RAG Functionality: Add documents or ask questions.
    2. Web Search: Search for information about musicians or bands.
    """

    st.set_page_config(
        page_title="Concert Tours Information Bot",
        page_icon="🎤",
        layout="centered",
    )

    st.title("🎤 Concert Tours Information Bot")

    with st.expander("ℹ️ Show Info", expanded=False):
        st.markdown(
            """
            ### About This Tool
            **Core RAG Functionality**:
            - **Add a Document**: Paste a document to include it in the knowledge base.
            - **Ask a Question**: Input a question to retrieve relevant insights.

            **Web Search**:
            - **Find Information**: Search for details about musicians or bands, including upcoming concerts.

            **Key Features**:
            - AI-driven document processing and question answering.
            - Integrated web search for concert-related data.
            - Supports queries on VIP packages, logistics, and more.

            **Need Help?**
            Reach out to us at [support@example.com](mailto:support@example.com) for assistance or feedback.
            """
        )

    # Add mode selection buttons to the main page
    st.subheader("Choose a Mode:")
    mode = st.radio("", ["Core RAG Functionality", "Web Search"], horizontal=True)

    # Input field for user requests
    placeholder_text = (
        "Enter your query (e.g., add a document or ask a question):"
        if mode == "Core RAG Functionality"
        else "Enter the name of a musician or band to search for their concert:"
    )
    user_input = st.text_area(placeholder_text, height=250)

    # Submit button
    if st.button("🚀 Submit"):
        if user_input.strip():
            if mode == "Core RAG Functionality":
                classification = classify_input(user_input)

                if classification == "document":
                    document_content = user_input[len("Please, add this document to your database:"):].strip()
                    if document_content:
                        result = ingest_document(document_content, vectorstore)
                        st.success(result)
                    else:
                        st.error("Please provide the document content to add.")
                elif classification == "question":
                    docs = retrieve_documents(user_input, vectorstore)
                    response = generate_response(docs, user_input)
                    st.write("### Response:")
                    st.write(response)
                else:
                    st.error("Unable to classify the input. Please try again.")
            elif mode == "Web Search":
                relevant_docs = perform_web_search(user_input)
                if relevant_docs:
                    response = generate_response(
                        docs=relevant_docs,
                        question=f"Show me details about the {user_input} upcoming concerts."
                    )
                    st.write("### Response:")
                    st.write(response)
                else:
                    st.error("No relevant information found for the given artist.")
        else:
            st.error("Please enter a valid query.")

    # Footer
    st.divider()
    st.markdown(
        """
        **Note**: This tool is powered by advanced AI and may occasionally provide incomplete or inaccurate results.  
        For feedback or support, contact us at [support@example.com](mailto:support@example.com).
        """
    )