import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOllama

import streamlit as st

# Define a function to initialize the chat model
def initialize_chat_model(model):
    try:
        if model == 'OpenAI':
            llm = ChatOpenAI(
                model_name='gpt-3.5-turbo',
                temperature=0,
                # api_key=os.environ.get('OPENAI_API_KEY')
            )
            print(os.environ.get('OPENAI_API_KEY'))
        elif model == 'Ollama':
            llm = ChatOllama(model='phi3')
    except Exception as e:
        st.warning(f"Error initializing model {model}: {str(e)}")
        llm = None
    return llm
    
# TODO: Add a function to generate conversational chain with OpenAI API
def get_conversation_chain(vectorstore, model):

    # Step 1: Define retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1, "include_metadata": True}
    )

    # Step 2: Augment
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
    # Use bullet-points and provide as much detail as possible in your answer. 
    PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
    )

    # Step 3: Generate
    # Create a ChatOpenAI object with the OpenAI API key
    llm = initialize_chat_model(model)
    
    if llm is None:
        raise ValueError(f"Failed to initialize the model: {model}")
    
    # Using memory
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key='chat_history',
        return_messages=True,
        ai_prefix='Assistant',
        output_key='answer')
    
    # Get conversational chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        combine_docs_chain_kwargs={'prompt': PROMPT},
        retriever=retriever,
        get_chat_history=lambda h : h,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain.invoke
