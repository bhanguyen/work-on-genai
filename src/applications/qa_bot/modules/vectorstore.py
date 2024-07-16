import os
from langchain_community.vectorstores.pgvector import PGVector

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

# TODO: Add a function to create a vector store, and create embeddings
def get_vectorstore(text_chunks, CONNECTION_STRING, collection_name=None):
    """
    Creates a PGVector object with embeddings generated using the OpenAI API.

    Args:
        text_chunks (list): A list of text chunks to be embedded.
        collection_name (str, optional): Name of the collection/table. Defaults to "langchain".
        CONNECTION_STRING (str): Connection string for the database.

    Returns:
        PGVector: A PGVector object with the embedded text chunks.
    """
    
    # Raise error if CONNECTION_STRING is not provided
    if not CONNECTION_STRING:
        raise ValueError("CONNECTION_STRING must be provided.")
    
    # Define embeddings model
    # embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = HuggingFaceEmbeddings()  #model_name="all-MiniLM-L6-v2")
    
    # Assign "langchain" for collection name if not provided
    if not collection_name:
        collection_name = "langchain"
    
    if text_chunks is None:
        # If no text chunks are provided, create a PGVector object with the default embedding function
        return PGVector(
            connection_string=CONNECTION_STRING,
            embedding_function=embeddings,
            collection_name=collection_name,
            use_jsonb=True
        )
    
    # Ensure text_chunks is a list of strings
    if not isinstance(text_chunks, list):
        raise TypeError("text_chunks must be a list of strings.")
    
    # Convert all elements to strings
    text_chunks = [str(text) for text in text_chunks]
    
    # Create a PGVector object with the embedded text chunks
    return PGVector.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=CONNECTION_STRING,
        use_jsonb=True
    )

########## LLAMA-INDEX ##############
import logging
from typing import List

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

def get_llama_index_reader(pdf_docs) -> List[Document]:
    reader = SimpleDirectoryReader(pdf_docs)
    return reader.load_data(show_progress=True)

@st.cache_resource
def get_llama_index_embeddings() -> HuggingFaceEmbedding:
    """
    Get or create a cached HuggingFace embedding model.
    
    Returns:
        HuggingFaceEmbedding: The embedding model.
    """
    return HuggingFaceEmbedding("sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def get_db_engine():
    """
    Get or create a cached SQLAlchemy engine with connection pooling.
    
    Returns:
        Engine: The SQLAlchemy engine.
    """
    connection_string = "postgresql://postgres:Password@localhost:5432/rag"
    return create_engine(connection_string, poolclass=QueuePool)

def get_llama_index_vector_store() -> PGVectorStore:
    """
    Create a PostgreSQL vector store.
    
    Returns:
        PGVectorStore: The vector store.
    """
    return PGVectorStore.from_engine(get_db_engine(), embed_dim=768)

def process_documents(documents: List[Document]) -> PGVectorStore:
    """
    Process a list of documents and store them in the vector store.
    
    This function takes a list of Document objects, processes them using
    various transformations, and stores the processed data in a PostgreSQL
    vector store. It also displays a progress bar during processing.
    
    Args:
        documents (List[Document]): The list of documents to process.
    
    Returns:
        PGVectorStore: The vector store containing the processed documents.
    """
    # Initialize components for document processing
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    title_extractor = TitleExtractor(nodes=5)
    qa_extractor = QuestionsAnsweredExtractor(questions=5)
    embed_model = get_llama_index_embeddings()
    vector_store = get_llama_index_vector_store()
    docstore = SimpleDocumentStore()
    
    # Set up the ingestion pipeline with all transformations
    transformations = [text_splitter, title_extractor, qa_extractor, embed_model]
    pipeline = IngestionPipeline(
        transformations=transformations,
        vector_store=vector_store,
        docstore=docstore
    )

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting document processing for {len(documents)} documents.")

    # Process documents with a progress bar
    with st.progress(0) as progress_bar:
        for i, doc in enumerate(documents):
            # Process each document individually
            pipeline.run(documents=[doc])
            # Update progress bar
            progress_bar.progress((i + 1) / len(documents))

    logger.info("Document processing completed.")
    return vector_store

def get_llama_index_vectorstore_index() -> VectorStoreIndex:
    """
    Create and return a VectorStoreIndex.

    This function sets up a VectorStoreIndex using a PostgreSQL vector store
    and a SimpleDocumentStore. It uses the default storage context and
    a pre-configured embedding model.

    Returns:
        VectorStoreIndex: An index built from the vector store and document store.
    """
    # Get the vector store
    vector_store = get_llama_index_vector_store()

    # Create a new document store
    docstore = SimpleDocumentStore()

    # Set up the storage context with the vector store and document store
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore
    )

    # Create and return the VectorStoreIndex
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=get_llama_index_embeddings(),
    )
    return index

def get_llama_index_retriever() -> VectorIndexRetriever:
    """
    Create and return a VectorIndexRetriever.

    This function sets up a VectorIndexRetriever using the VectorStoreIndex
    and a SentenceTransformerRerank postprocessor. It's configured to retrieve
    the top 5 most similar results and then rerank the top 3.

    Returns:
        VectorIndexRetriever: A configured retriever for querying the vector index.
    """
    # Get the vector store index
    index = get_llama_index_vectorstore_index()

    # Create a reranker for post-processing
    reranker = SentenceTransformerRerank(top_n=3)
    
    # Create and return the VectorIndexRetriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,  # Retrieve top 5 similar results
        node_postprocessors=[reranker]  # Apply reranking to top 3
    )
    return retriever