import streamlit as st
import os
from dotenv import load_dotenv
import traceback
from langchain_community.vectorstores.pgvector import PGVector

from applications.qa_bot.modules.process_documents import get_pdf_text, get_text_chunks
from applications.qa_bot.modules.vectorstore import get_vectorstore
from applications.qa_bot.modules.rag import get_conversation_chain
from applications.qa_bot.htmlTemplates import css, bot_template, user_template

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

# Define a function to initialize the session state
def initialize_session_state(vectorstore, model_type, model):
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(vectorstore, model_type, model)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

# TODO: Create a streamlit app
def main():
    st.set_page_config(
        page_title="Streamlit Question Answering App",
        layout="wide",
        page_icon=":books:"
    )
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
        
        # Check if connection string is set
        if not CONNECTION_STRING:
            st.sidebar.error("PostgreSQL connection string is not set. Please set the PG_CONNECTION_STRING environment variable.")

        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", type="pdf", accept_multiple_files=True)

        collection_name = st.text_input("Enter your collection name (Optional)")
        
        # Updated model selection
        model_type = st.selectbox('Select model type', ['OpenAI', 'Ollama', 'Anthropic'])

        # Set up API keys
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if model_type == 'OpenAI':
            model = 'gpt-3.5-turbo'
            if not openai_api_key:
                st.sidebar.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        elif model_type == 'Ollama':
            model = st.selectbox('Select Ollama model', ['phi3', 'llama3', 'mistral'])
        elif model_type == 'Anthropic':
            model = 'claude-3-sonnet-20240229'
            if not anthropic_api_key:
                st.sidebar.error("Anthropic API key is not set. Please set the ANTHROPIC_API_KEY environment variable.")
        
        # If the user clicks the "Process" button, the following code is executed:
        # i. raw_text = get_pdf_text(pdf_docs): retrieves the text content from the uploaded PDF documents.
        # ii. text_chunks = get_text_chunks(raw_text): splits the text content into smaller chunks for efficient processing.
        # iii. vectorstore = get_vectorstore(text_chunks): creates a vector store that stores the vector representations of the text chunks.
        if st.button("Process"):
            with st.spinner("Processing"):
                try:
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
                    st.session_state.conversation = get_conversation_chain(vectorstore, model_type, model)
                    st.success('PDF uploaded successfully!', icon="✅")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")

    # A header with the text appears at the top of the Streamlit application.
    st.header("GenAI Q&A with pgvector, AWS Aurora Postgres, and Langchain :books::parrot:")    
    st.markdown(f"Ask a question about your documents with __{model_type} - {model}__:")    
    # Create a text input box where you can ask questions about your documents.
    user_question = st.chat_input(placeholder="Ask me something")
    # Select chat model
    vectorstore = get_vectorstore(None, CONNECTION_STRING)
    try:
        initialize_session_state(vectorstore, model_type, model)
    except Exception as e:
        if model_type == 'OpenAI':
            st.warning('Missing OpenAI key')
        elif model_type == 'Ollama':
            st.warning('Make sure you initialized Ollama')
        elif model_type == 'Anthropic':
            st.warning('Missing Anthropic API key')
        st.error(f"Error: {str(e)}")


    # If the user enters a question, it calls the handle_userinput() function to process the user's input.
    if user_question:
        with st.spinner("Processing..."):
            handle_userinput(user_question)
        with st.expander("See retrieved documents"):
            docs = vectorstore.similarity_search(user_question, k=2)
            result = "\n\n".join([doc.page_content for doc in docs])
            st.write(result)
    
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