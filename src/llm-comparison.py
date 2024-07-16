"""
This Streamlit application compares responses from different Language Models (LLMs) using Retrieval-Augmented Generation (RAG).
It retrieves relevant context from a PostgreSQL vector store and uses it to generate responses from Ollama, Anthropic, and OpenAI models.
The app allows users to select different Ollama models and visualizes the generation process in real-time.
"""

import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.vectorstores import PGVector
from langchain_community.vectorstores.pgvector import PGVector
import anthropic

# Import custom vectorstore module
from applications.qa_bot.modules.vectorstore import get_vectorstore

# Load environment variables
load_dotenv()

# Set up API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# # Set up PostgreSQL connection string
# CONNECTION_STRING = PGVector.connection_string_from_db_params(                                                  
#         driver = os.environ.get("PGVECTOR_DRIVER"),
#         user = os.environ.get("PGVECTOR_USER"),                                      
#         password = os.environ.get("PGVECTOR_PASSWORD"),                                  
#         host = os.environ.get("PGVECTOR_HOST"),                                            
#         port = os.environ.get("PGVECTOR_PORT"),                                          
#         database = os.environ.get("PGVECTOR_DATABASE")
#     )

# # Initialize vectorstore
# vectorstore = get_vectorstore(None, CONNECTION_STRING)

# def get_relevant_context(query: str, top_k: int = 3) -> str:
#     """
#     Retrieve relevant context from the vector store based on the input query.

#     Args:
#         query (str): The input question or prompt.
#         top_k (int): The number of most relevant documents to retrieve.

#     Returns:
#         str: A string containing the concatenated content of the most relevant documents.
#     """
#     docs = vectorstore.similarity_search(query, k=top_k)
#     return "\n\n".join([doc.page_content for doc in docs])

from applications.qa_bot.modules.vectorstore import (
    get_llama_index_retriever, 
    process_documents
)
from llama_index.core import Document
import PyPDF2

def extract_text_from_pdf(uploaded_files):
    text_data = []
    for uploaded_file in uploaded_files:
        reader = PyPDF2.PdfReader(uploaded_file)
        num_pages = len(reader.pages)
        # Extract text from each page of the PDF
        for page in range(num_pages):
            page_obj = reader.pages[page]
            text_data.append(page_obj.extract_text())
    return text_data

def get_relevant_llama_index(query: str):
    # vector_store = get_llama_index_vector_store()
    llama_index_retriever = get_llama_index_retriever()

    docs = llama_index_retriever.retrieve(query)

    return "\n\n".join([doc.text for doc in docs]), docs
    # return docs

# Set temperature 
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

# Set up Langchain components
prompt_template = ChatPromptTemplate.from_template("Context: {context}\n\nQuestion: {question}\n\nAnswer:")

# Initialize OpenAI model
openai_llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=temperature)
openai_chain = RunnableSequence(prompt_template | openai_llm)

# Function to get Claude 3.5 Sonnet response
def get_claude_response(context: str, question: str) -> str:
    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=temperature,
        system="You are a helpful AI assistant. Use the provided context to answer the user's question.",
        messages=[
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    return message.content[0].text

# Streamlit UI setup
from applications.qa_bot.modules.process_documents import get_pdf_text
with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Process"):
        with st.spinner("Processing"):
            # get pdf text
            # text_docs = get_pdf_text(pdf_docs)
            text_docs = extract_text_from_pdf(pdf_docs)
            documents = [Document(text=doc) for doc in text_docs]  # Convert to Document objects
            vector_store = process_documents(documents)
            st.success('PDF uploaded successfully!', icon="✅")

st.title("LLM Response Comparison")
st.write("This application compare the responses between different LLM including phi3, llama3, mistral, openai, anthropic")

# Ollama model selection in the sidebar
ollama_models = ["phi3", "llama3", "mistral"]
selected_ollama_model = st.sidebar.selectbox("Select Ollama Model", ollama_models)

# Initialize Ollama LLM with the selected model
ollama_llm = ChatOllama(model=selected_ollama_model, temperature=temperature)
ollama_chain = RunnableSequence(prompt_template | ollama_llm)

# Main input area for user prompt
user_prompt = st.text_area("Enter your prompt:")

if st.button("Generate Responses", key="response"):
    if user_prompt:
    # Retrieve relevant context from the vector store
    # context = get_relevant_context(user_prompt)
    # st.subheader("Retrieved Context:")
    # st.write(context)
        context, docs = get_relevant_llama_index(user_prompt)
        
        tab1, tab2, tab3 = st.tabs(["Retrieved Context", "Generated Questions", "Responses"])
        with tab1:
            st.write(context)
        with tab2:
            for doc in docs:
                st.write(doc.metadata['document_title'])
                st.write(doc.metadata['questions_this_excerpt_can_answer'])
                st.write(doc.score)
        
        with tab3:
            # Create three columns for displaying LLM responses
            col1, col2, col3 = st.columns(3)
            
            # Generate and display Ollama response
            with col1:
                st.subheader(f"Ollama ({selected_ollama_model})")
                with st.spinner("Generating..."):
                    ollama_response = ollama_chain.invoke({"context": context, "question": user_prompt}, config={"callbacks": [StreamlitCallbackHandler(st.container())]})
                st.write(ollama_response.content)
            
            # Generate and display Anthropic (Claude 3.5 Sonnet) response
            with col2:
                st.subheader("Anthropic (Claude 3.5 Sonnet)")
                with st.spinner("Generating..."):
                    claude_response = get_claude_response(context, user_prompt)
                st.write(claude_response)
            
            # Generate and display OpenAI response
            with col3:
                st.subheader("OpenAI (GPT-3.5)")
                with st.spinner("Generating..."):
                    openai_response = openai_chain.invoke({"context": context, "question": user_prompt}, config={"callbacks": [StreamlitCallbackHandler(st.container())]})
                st.write(openai_response.content)
    else:
        st.warning("Please enter a question.")

# Sidebar information and warnings
st.sidebar.title("Settings")
st.sidebar.info("Make sure to set your API keys and connection string as environment variables:\n\n"
                "OPENAI_API_KEY\n"
                "ANTHROPIC_API_KEY\n"
                "PG_CONNECTION_STRING")
st.sidebar.warning("Ensure Ollama is running locally on the default port.")

# Check for missing API keys and connection string
if not openai_api_key:
    st.sidebar.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
if not anthropic_api_key:
    st.sidebar.error("Anthropic API key is not set. Please set the ANTHROPIC_API_KEY environment variable.")
# if not CONNECTION_STRING:
#     st.sidebar.error("PostgreSQL connection string is not set. Please set the PG_CONNECTION_STRING environment variable.")