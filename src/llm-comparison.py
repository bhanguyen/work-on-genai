"""
This Streamlit application compares responses from different Language Models (LLMs) using Retrieval-Augmented Generation (RAG).
It retrieves relevant context from a PostgreSQL vector store and uses it to generate responses from Ollama, Anthropic, and OpenAI models.
The app allows users to select different Ollama models and visualizes the generation process in real-time.
"""

import streamlit as st
import os
import PyPDF2
from dotenv import load_dotenv

# Import langchain libraries for prompting and chat models
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.callbacks import StreamlitCallbackHandler
import anthropic

# Import LlamaIndex libraries for document processing and retrieval
from llama_index.core import Document

# Import custom vectorstore module
from applications.qa_bot.modules.vectorstore import (
    get_llama_index_retriever, 
    process_documents
)
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

def get_openai_response(context: str, question: str, model: str, temperature):
    # Initialize OpenAI model
    openai_llm = ChatOpenAI(model=model, openai_api_key=openai_api_key, temperature=temperature)
    prompt_template = get_prompt_template(context, question)
    openai_chain = RunnableSequence(prompt_template | openai_llm)
    openai_response = openai_chain.invoke(
        {"context": context, "question": user_prompt}, 
        config={"callbacks": [StreamlitCallbackHandler(st.container())]}
    )
    return openai_response.content

# Function to get Claude 3.5 Sonnet response
def get_claude_response(context: str, question: str, model: str, temperature):
    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
    message = anthropic_client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=temperature,
        system="You are a helpful AI assistant. Use the provided context to answer the user's question.",
        messages=[
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    return message.content[0].text

# Function to get Ollama's response
def get_ollama_response(context: str, question: str, model: str, temperature):
    # Initialize Ollama LLM with the selected model
    ollama_llm = ChatOllama(model=model, temperature=temperature)
    prompt_template = get_prompt_template(context, question)
    ollama_chain = RunnableSequence(prompt_template | ollama_llm)
    ollama_response = ollama_chain.invoke(
        {"context": context, "question": user_prompt}, 
        config={"callbacks": [StreamlitCallbackHandler(st.container())]}
    )
    return ollama_response.content

# Display response
def display_response(model, prompt):
    model_categories = {
        "Ollama": ["phi3", "llama3", "mistral", "gemma2"],
        "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-2.1"],
        "OpenAI": ["gpt-4", "gpt-3.5-turbo"]
    }
    try:
        if model in model_categories["Ollama"]:
            with st.spinner("Generating..."):
                    ollama_response = get_ollama_response(
                        context=context, 
                        question=user_prompt,
                        model=model, 
                        temperature=temperature
                    )
            st.write(ollama_response)
            
        elif model in model_categories["Anthropic"]:
            with st.spinner("Generating..."):
                    claude_response = get_claude_response(context, user_prompt)
            st.write(claude_response)

        elif model in model_categories["OpenAI"]:
            with st.spinner("Generating..."):
                    openai_response = get_openai_response(
                        context=context,
                        question=user_prompt,
                        model=model,
                        temperature=temperature
                    )
            st.write(openai_response)
        else:
            raise ValueError(f"Unknown model: {model}")
    except Exception as e:
        st.error(f"Error generating response from {model}: {str(e)}")
    
# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("LLM Response Comparison with LlamaIndex :llama:")
st.write("This application compare the responses between different LLM including phi3, llama3, mistral, openai, anthropic")

# Setup sidebar
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

# Ollama model selection in the sidebar
# ollama_models = ["phi3", "llama3", "mistral"]
# selected_ollama_model = st.sidebar.selectbox("Select Ollama Model", ollama_models)
model_categories = {
"Ollama": ["phi3", "llama3", "mistral", "gemma2"],
"Anthropic": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-5-sonnet-20240620"],
"OpenAI": ["gpt-40-mini", "gpt-3.5-turbo"]
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
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
st.sidebar.info("Make sure to set your API keys and connection string as environment variables:\n\n"
                "OPENAI_API_KEY\n"
                "ANTHROPIC_API_KEY")
st.sidebar.warning("Ensure Ollama is running locally on the default port.")

# Main input area for user prompt
user_prompt = st.text_input("Enter your prompt:", placeholder="Your query")
from vector_visual import main
with st.expander("Check Query Embedding"):
    context_node_ids = [
                'e6cc36a4-4ac1-4840-b3e7-3360fb7b68b3',
                '18d79297-540a-41b5-90f5-79ae1dcb196e',
                '093d9c81-4db6-4f41-a7bb-439130ecb013',
                '9d493f75-7b05-4ce0-9b2c-fe123414fc20',
                'd756cc58-a2fb-45b1-9f64-a85fbda72599',
                '0187ecc2-238e-49d6-8695-f943253c7c8f',
                '7e0f7f7c-86b0-43c6-a839-2764528c2e8a',
                '18a1329d-6f35-468e-bb8f-cbc69d89187f',
                'd5b18386-dc02-4885-84f3-5c94ea52b086',
                '59c41c65-77e0-46a1-99e3-b8cfad25efe7',
                'e8c1d19f-0200-4129-b875-207af0e4ac9e',
                '1f2a47e5-7a42-4a3d-9c33-ab50a3201456'
            ]
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
    #         # Create three columns for displaying LLM responses
    #         col1, col2, col3 = st.columns(3)
            
    #         # Generate and display Ollama response
    #         with col1:
    #             st.subheader(f"Ollama ({selected_ollama_model})")
    #             with st.spinner("Generating..."):
    #                 ollama_response = get_ollama_response(
    #                     context=context, 
    #                     question=user_prompt,
    #                     model=selected_ollama_model, 
    #                     temperature=temperature
    #                 )
    #             st.write(ollama_response)
            
    #         # Generate and display Anthropic (Claude 3.5 Sonnet) response
    #         with col2:
    #             st.subheader("Anthropic (Claude 3.5 Sonnet)")
    #             with st.spinner("Generating..."):
    #                 claude_response = get_claude_response(context, user_prompt)
    #             st.write(claude_response)
            
    #         # Generate and display OpenAI response
    #         with col3:
    #             st.subheader("OpenAI (GPT-3.5)")
    #             with st.spinner("Generating..."):
    #                 openai_response = get_openai_response(
    #                     context=context,
    #                     question=user_prompt,
    #                     model="gpt-3.5-turbo",
    #                     temperature=temperature
    #                 )
    #             st.write(openai_response)
    # else:
    #     st.warning("Please enter a question.")
            

            tabs = st.tabs(selected_models)
            for i, tab in enumerate(tabs):
                with tab:
                    display_response(selected_models[i], user_prompt)

# Check for missing API keys and connection string
if not openai_api_key:
    st.sidebar.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
if not anthropic_api_key:
    st.sidebar.error("Anthropic API key is not set. Please set the ANTHROPIC_API_KEY environment variable.")
# if not CONNECTION_STRING:
#     st.sidebar.error("PostgreSQL connection string is not set. Please set the PG_CONNECTION_STRING environment variable.")