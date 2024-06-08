import streamlit as st
import os
from dotenv import load_dotenv

from modules.process_documents import get_pdf_text, get_text_chunks
from modules.vectorstore import get_vectorstore
from modules.rag import get_conversation_chain
from langchain_community.vectorstores.pgvector import PGVector

import traceback
from htmlTemplates import css, bot_template, user_template


# TODO: Add a function to handle user input
def handle_userinput(user_question):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    try:
        # Ensure user_question is a string
        if not isinstance(user_question, str):
            user_question = str(user_question)
        
        response = st.session_state.conversation({'question': user_question})
        
    except ValueError:
        st.write("Sorry, I didn't understand that. Could you rephrase your question?")
        print(traceback.format_exc())
        return

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

# TODO: Create a streamlit app
def main():
    st.set_page_config(
        page_title="Streamlit Question Answering App",
        layout="wide",
        page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            """
            ### Instructions:
            1. Browse and upload PDF files
            2. Enter collection name (optional)
            3. Click Process
            4. Type your question in the search bar to get more insights
            """
        )
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", type="pdf", accept_multiple_files=True)

        collection_name = st.text_input("Enter your collection name (Optional)")
        
        # If the user clicks the "Process" button, the following code is executed:
        # i. raw_text = get_pdf_text(pdf_docs): retrieves the text content from the uploaded PDF documents.
        # ii. text_chunks = get_text_chunks(raw_text): splits the text content into smaller chunks for efficient processing.
        # iii. vectorstore = get_vectorstore(text_chunks): creates a vector store that stores the vector representations of the text chunks.
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(
                    text_chunks, 
                    CONNECTION_STRING, 
                    collection_name
                )

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success('PDF uploaded successfully!', icon="✅")

    # Check if the conversation and chat history are not present in the session state and initialize them to None.
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(get_vectorstore(None, CONNECTION_STRING))
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    # A header with the text appears at the top of the Streamlit application.
    st.header("GenAI Q&A with pgvector, AWS Aurora Postgres, and Langchain :books::parrot:")    
    st.markdown("Ask a question about your documents:")
    # Create a text input box where you can ask questions about your documents.
    user_question = st.chat_input(placeholder="Ask me something")
    
    
    # If the user enters a question, it calls the handle_userinput() function to process the user's input.
    if user_question:
        with st.spinner("Processing..."):
            handle_userinput(user_question)
    
    # Clear chat history and user input when the Clear Chat button is clicked
    if st.button("Clear Chat"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()  # Refresh the page to reflect the changes
    
    with st.sidebar:
        st.divider()

    st.sidebar.markdown(
    """
    ### Sample questions to get started:
    1. Question 1?
    2. Question 2?
    """
)
    
if __name__ == '__main__':
    load_dotenv()

    CONNECTION_STRING = PGVector.connection_string_from_db_params(                                                  
        driver = os.environ.get("PGVECTOR_DRIVER"),
        user = os.environ.get("PGVECTOR_USER"),                                      
        password = os.environ.get("PGVECTOR_PASSWORD"),                                  
        host = os.environ.get("PGVECTOR_HOST"),                                            
        port = os.environ.get("PGVECTOR_PORT"),                                          
        database = os.environ.get("PGVECTOR_DATABASE")
    )

    main()                                       