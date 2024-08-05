"""
This Streamlit application compares responses from different Language Models (LLMs) using Retrieval-Augmented Generation (RAG).
It retrieves relevant context from a PostgreSQL vector store and uses it to generate responses from Ollama, Anthropic, and OpenAI models.
The app allows users to select different Ollama models and visualizes the generation process in real-time.
"""

import streamlit as st
import os
import PyPDF2
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Import langchain libraries for prompting and chat models
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.callbacks import StreamlitCallbackHandler

# Import LlamaIndex libraries for document processing and retrieval
from llama_index.core import Document

# Import custom module
from applications.qa_bot.modules.vectorstore import (
    get_llama_index_retriever, 
    process_documents
)
# from applications.qa_bot.modules.process_documents import extract_text_from_pdf
from vector_visual import main

# Load environment variables
load_dotenv()

# Set up API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

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

@st.cache_data
def get_relevant_context(query: str):
    # vector_store = get_llama_index_vector_store()
    llama_index_retriever = get_llama_index_retriever()

    docs = llama_index_retriever.retrieve(query)

    return "\n\n".join([doc.text for doc in docs]), docs

def get_prompt_template(context: str, question: str):
    return ChatPromptTemplate.from_template(
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )

def get_llm_response(context: str, question: str, model: str, temperature:float):
    # Define prompt template
    prompt_template = get_prompt_template(context, question)

    # Innitialize the appropriate chat model based on the selected model name
    if model in model_categories["Ollama"]:
        llm = ChatOllama(model=model, temperature=temperature)
    elif model in model_categories["Anthropic"]:
        llm = ChatAnthropic(model=model, anthropic_api_key=anthropic_api_key, temperature=temperature)
    elif model in model_categories["OpenAI"]:
        llm = ChatOpenAI(model=model, openai_api_key=openai_api_key, temperature=temperature)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Set up the chain
    chain = RunnableSequence(prompt_template | llm)
    response = chain.invoke(
        {"context": context, "question": user_prompt},
        config={"callbacks": [StreamlitCallbackHandler(st.container())]}
    )
    return response.content

# Display response
def display_response(context, question, model, temperature):
    try:
        with st.spinner("Generating..."):
            response = get_llm_response(context=context, question=question, model=model, temperature=temperature)
            st.write(response)
    except Exception as e:
        st.error(f"Error generating response from {model}: {str(e)}")
    
# Streamlit UI Main App
st.title("LLM Response Comparison with LlamaIndex :llama:")
st.write("This application compare the responses between different LLMs hosted with OpenAI, Anthropic, and Ollama")

# Uploading documents process
with st.sidebar:
    # Upload documents
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    # Process documents
    if st.button("Process"):
        with st.spinner("Processing"):
            # get pdf text
            text_docs = extract_text_from_pdf(pdf_docs)
            documents = [Document(text=doc) for doc in text_docs]  # Convert to Document objects
            vector_store = process_documents(documents)
            st.success('PDF uploaded successfully!', icon="✅")

# Define model categories and models
model_categories = {
"Ollama": ["phi3", "llama3.1", "mistral", "gemma2"],
"Anthropic": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-5-sonnet-20240620"],
"OpenAI": ["gpt-4o-mini", "gpt-3.5-turbo"]
}

with st.sidebar:
    # Initialize session state for selections
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = {category: [] for category in model_categories}

    # Category selection using checkboxes and model selection
    for category, models in model_categories.items():
        category_selected = st.checkbox(f"Include {category} models")
        if category_selected:
            st.session_state.selected_models[category] = st.multiselect(
                f"Select {category} models:",
                models,
                default=st.session_state.selected_models[category]
            )
        else:
            st.session_state.selected_models[category] = []

# Flatten the selected models list
selected_models = [model for models in st.session_state.selected_models.values() for model in models]

# Sidebar information and warnings
st.sidebar.title("Settings")
# Set temperature 
# temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
temperature = 0
st.sidebar.info("Make sure to set your API keys and connection string as environment variables:\n\n"
                "OPENAI_API_KEY\n"
                "ANTHROPIC_API_KEY")
st.sidebar.warning("Ensure Ollama is running locally on the default port.")

# Constants
HOST = os.environ.get("DB_HOST")
USER = os.environ.get("DB_USER")
PASSWORD = os.environ.get("DB_PASSWORD")
PORT = os.environ.get("DB_PORT")
DATABASE = os.environ.get("PGVECTOR_DATABASE")

# Main input area for user prompt
user_prompt = st.text_input("Enter your prompt:", placeholder="Your query")

with st.expander("Check Query Embedding"):

    # Database connection details
    CONNECTION_STRING = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"

    # Create an engine
    engine = create_engine(CONNECTION_STRING)

    # Define the query
    query = "SELECT node_id FROM data_llamaindex;"

    # Connect to the database and execute the query
    with engine.connect() as connection:
        result = connection.execute(text(query))
        context_node_ids = [row[0] for row in result.fetchall()]

    # Print the populated list
    print(context_node_ids)
    main(context_node_ids, user_prompt)

if st.button("Generate Responses", key="response"):
    if user_prompt:
        # Retrieve relevant context from the vector store
        context, docs = get_relevant_context(user_prompt)
        
        tab1, tab2, tab3 = st.tabs(["Retrieved Context", "Generated Questions", "Responses"])
        
        with tab1:
            st.write(context)

        with tab2:
            for doc in docs:
                st.write(doc.metadata['document_title'])
                st.write(doc.metadata['questions_this_excerpt_can_answer'])
                st.write(doc.score)
        
        with tab3:
            tabs = st.tabs(selected_models)
            for i, tab in enumerate(tabs):
                with tab:
                    display_response(context, user_prompt, selected_models[i], temperature)
    else:
        st.warning("Please enter a question.")

# Check for missing API keys and connection string
if not openai_api_key:
    st.sidebar.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
if not anthropic_api_key:
    st.sidebar.error("Anthropic API key is not set. Please set the ANTHROPIC_API_KEY environment variable.")
if not CONNECTION_STRING:
    st.sidebar.error("PostgreSQL connection string is not set. Please set the PG_CONNECTION_STRING environment variable.")