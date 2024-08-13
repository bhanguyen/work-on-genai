import os
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

import streamlit as st


def init_retriever(vectorstore):
    """
    Initialize and return a retriever from a given vector store.

    Args:
        vectorstore (VectorStore): The vector store to use for retrieval.

    Returns:
        Retriever: A configured retriever object.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "include_metadata": True}
    )

def init_prompt():
    """
    Initialize and return a prompt template for question answering.

    Returns:
        PromptTemplate: A configured prompt template object.
    """
    prompt_template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    Always use English as the language in your responses.
    In your answers, always use a professional tone.
    Simply answer the question clearly and with lots of detail using only the relevant details from the information below. 
    If the context does not contain the answer, say "Sorry, I didn't understand that. Could you rephrase your question?"
    Use three sentences maximum and keep the answer concise.

    Now read this context below and answer the question at the bottom.
    
    Context: {context}

    Question: {question}
    
    Answer:
    """
    return PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

def init_chat_model(model_type: str, model: str):
    """
    Initialize and return a chat model based on the specified type and model name.

    Args:
        model_type (str): The type of model to initialize ('OpenAI', 'Ollama', or 'Anthropic').
        model (str): The specific model name to use.

    Returns:
        object: An initialized chat model object, or None if initialization fails.

    Raises:
        Exception: If there's an error during model initialization.
    """
    try:
        if model_type == 'OpenAI':
            # Initialize OpenAI chat model
            llm = ChatOpenAI(
                model_name=model,
                verbose=True,
                temperature=0,  # Set to 0 for more deterministic outputs
            )
        elif model_type == 'Ollama':
            # Initialize Ollama chat model
            llm = ChatOllama(
                model=model,
                verbose=True,
                temperature=0,  # Set to 0 for more deterministic outputs
            )
        elif model_type == 'Anthropic':
            # Initialize Anthropic (Claude) chat model
            llm = ChatAnthropic(
                model=model,
                verbose=True,
                temperature=0,  # Set to 0 for more deterministic outputs
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return llm
    except Exception as e:
        # Log the error and return None if initialization fails
        st.warning(f"Error initializing {model_type} model {model}: {str(e)}")
        return None
    
def get_conversation_chain(vectorstore, model_type: str, model: str):
    """
    Generate and return a conversational chain for question answering.

    This function sets up a retrieval-based question answering system using the specified
    vector store and language model.

    Args:
        vectorstore (VectorStore): The vector store containing the document embeddings.
        model_type (str): The type of language model to use ('OpenAI', 'Ollama', or 'Anthropic').
        model (str): The specific model name to use.

    Returns:
        Callable: An invocable function that runs the conversation chain.

    Raises:
        ValueError: If the language model initialization fails.
    """
    # Initialize components
    retriever = init_retriever(vectorstore)
    PROMPT = init_prompt()
    llm = init_chat_model(model_type, model)
    
    # Set up conversation memory
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key='chat_history',
        return_messages=True,
        ai_prefix='Assistant',
        output_key='answer'
    )
    
    # Create the conversational retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',  # 'stuff' method: stuff all retrieved documents into the prompt
        combine_docs_chain_kwargs={'prompt': PROMPT},
        retriever=retriever,
        get_chat_history=lambda h: h,  # Use full chat history
        memory=memory,
        return_source_documents=True,
        verbose=True  # Set to True for detailed logging
    )
    
    return conversation_chain.invoke

def get_rag_chain(vectorstore, model_type: str, model: str):
    """
    Generate and return a Retrieval-Augmented Generation (RAG) chain.

    This function sets up a RAG system using the specified vector store and language model.

    Args:
        vectorstore (VectorStore): The vector store containing the document embeddings.
        model_type (str): The type of language model to use ('OpenAI', 'Ollama', or 'Anthropic').
        model (str): The specific model name to use.

    Returns:
        Runnable: A runnable RAG chain.

    Raises:
        ValueError: If the language model initialization fails.
    """
    # Initialize components
    retriever = init_retriever(vectorstore)
    PROMPT = init_prompt()
    llm = init_chat_model(model_type, model)

    def format_docs(docs):
        """Combine the content of multiple documents into a single string."""
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Create RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain