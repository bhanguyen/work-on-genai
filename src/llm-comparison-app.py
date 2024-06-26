"""
This Streamlit application compares responses from different Language Models (LLMs) using Retrieval-Augmented Generation (RAG).
It retrieves relevant context from a vector store and uses it to generate responses from Ollama, Anthropic, and OpenAI models.
"""

import streamlit as st
from openai import OpenAI
import anthropic
import requests
import os
import json
from dotenv import load_dotenv

# Import custom vectorstore module
from qa_bot.modules.vectorstore import get_vectorstore
from langchain_community.vectorstores.pgvector import PGVector

# Load environment variables
load_dotenv()

# Set up API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
ollama_url = "http://localhost:11434/api/generate"  # Adjust if necessary

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
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def get_ollama_response(prompt: str, context: str, model: str = "phi3") -> str:
    """
    Get a response from the Ollama API using the provided prompt and context.

    Args:
        prompt (str): The user's question or prompt.
        context (str): Relevant context retrieved from the vector store.
        model (str): The Ollama model to use (default is "phi3").

    Returns:
        str: The generated response from Ollama, or an error message if the request fails.
    """
    try:
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        response = requests.post(ollama_url, json={"model": model, "prompt": full_prompt}, stream=True)
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if 'response' in json_response:
                    full_response += json_response['response']
                if json_response.get('done', False):
                    break

        return full_response
    except requests.RequestException as req_err:
        st.error(f"Error communicating with Ollama API: {req_err}")
        return "Error: Unable to communicate with Ollama API. Please check if Ollama is running and accessible."
    except json.JSONDecodeError as json_err:
        st.error(f"Error decoding JSON from Ollama API: {json_err}")
        return "Error: Unable to decode Ollama response. Please check the Ollama server logs."

def get_anthropic_response(prompt: str, context: str, model: str = "claude-3-sonnet-20240229") -> str:
    """
    Get a response from the Anthropic API using the provided prompt and context.

    Args:
        prompt (str): The user's question or prompt.
        context (str): Relevant context retrieved from the vector store.
        model (str): The Anthropic model to use (default is "claude-3-sonnet-20240229").

    Returns:
        str: The generated response from Anthropic, or an error message if the request fails.
    """
    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        full_prompt = f"Context: {context}\n\nHuman: {prompt}\n\nAssistant:"
        message = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        st.error(f"Error with Anthropic API: {e}")
        return "Error: Unable to get response from Anthropic. Please check your API key and try again."

def get_openai_response(prompt: str, context: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Get a response from the OpenAI API using the provided prompt and context.

    Args:
        prompt (str): The user's question or prompt.
        context (str): Relevant context retrieved from the vector store.
        model (str): The OpenAI model to use (default is "gpt-3.5-turbo").

    Returns:
        str: The generated response from OpenAI, or an error message if the request fails.
    """
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. Use the following context to answer the user's question: {context}"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return "Error: Unable to get response from OpenAI. Please check your API key and try again."

# Set up Streamlit page configuration
st.set_page_config(layout='wide')
st.title("LLM Response Comparison")

# User input
user_prompt = st.text_area("Enter your prompt:", height=100)

# Generate responses when the button is clicked
if st.button("Generate Responses"):
    if user_prompt:
        # Retrieve relevant context
        context = get_relevant_context(user_prompt)
        
        # Display retrieved context
        st.subheader("Retrieved Context:")
        st.write(context)

        # Create three columns for LLM responses
        col1, col2, col3 = st.columns(3)
        
        # Generate and display Ollama response
        with col1:
            st.subheader("Ollama (phi3)")
            with st.spinner("Generating..."):
                ollama_response = get_ollama_response(user_prompt, context)
            st.write(ollama_response)
        
        # Generate and display Anthropic response
        with col2:
            st.subheader("Anthropic (Claude 3.5 Sonnet)")
            with st.spinner("Generating..."):
                anthropic_response = get_anthropic_response(user_prompt, context)
            st.write(anthropic_response)
        
        # Generate and display OpenAI response
        with col3:
            st.subheader("OpenAI (GPT-3.5)")
            with st.spinner("Generating..."):
                openai_response = get_openai_response(user_prompt, context)
            st.write(openai_response)
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