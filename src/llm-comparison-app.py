"""
This Streamlit application compares responses from different Language Models (LLMs) using Retrieval-Augmented Generation (RAG).
It retrieves relevant context from a PostgreSQL vector store and uses it to generate responses from Ollama, Anthropic, and OpenAI models.
The app allows users to select different Ollama models and visualizes the generation process in real-time.
"""

import streamlit as st
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.vectorstores import PGVector
import anthropic
import os
from dotenv import load_dotenv

# Import custom vectorstore module
from qa_bot.modules.vectorstore import get_vectorstore
from langchain_community.vectorstores.pgvector import PGVector

# Load environment variables
load_dotenv()

# Set up API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Set up PostgreSQL connection string
CONNECTION_STRING = PGVector.connection_string_from_db_params(                                                  
        driver = os.environ.get("PGVECTOR_DRIVER"),
        user = os.environ.get("PGVECTOR_USER"),                                      
        password = os.environ.get("PGVECTOR_PASSWORD"),                                  
        host = os.environ.get("PGVECTOR_HOST"),                                            
        port = os.environ.get("PGVECTOR_PORT"),                                          
        database = os.environ.get("PGVECTOR_DATABASE")
    )

# Initialize vectorstore
vectorstore = get_vectorstore(None, CONNECTION_STRING)

def get_relevant_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve relevant context from the vector store based on the input query.

    Args:
        query (str): The input question or prompt.
        top_k (int): The number of most relevant documents to retrieve.

    Returns:
        str: A string containing the concatenated content of the most relevant documents.
    """
    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in docs])

# Set up Langchain components
prompt_template = ChatPromptTemplate.from_template("Context: {context}\n\nQuestion: {question}\n\nAnswer:")

# Initialize LLMs
openai_llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

# Create RunnableSequence for OpenAI
openai_chain = RunnableSequence(prompt_template | openai_llm)

# Function to get Claude 3.5 Sonnet response
def get_claude_response(context: str, question: str) -> str:
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are a helpful AI assistant. Use the provided context to answer the user's question.",
        messages=[
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    return message.content[0].text

# Streamlit UI setup
st.set_page_config(layout='wide')
st.title("LLM Response Comparison")

# Ollama model selection in the sidebar
ollama_models = ["phi3", "llama3", "mistral"]
selected_ollama_model = st.sidebar.selectbox("Select Ollama Model", ollama_models)

# Initialize Ollama LLM with the selected model
ollama_llm = ChatOllama(model=selected_ollama_model)
ollama_chain = RunnableSequence(prompt_template | ollama_llm)

# Main input area for user prompt
user_prompt = st.text_area("Enter your prompt:", height=100)

if st.button("Generate Responses"):
    if user_prompt:
        # Retrieve relevant context from the vector store
        context = get_relevant_context(user_prompt)
        
        st.subheader("Retrieved Context:")
        st.write(context)

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
if not CONNECTION_STRING:
    st.sidebar.error("PostgreSQL connection string is not set. Please set the PG_CONNECTION_STRING environment variable.")