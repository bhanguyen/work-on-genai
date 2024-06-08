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
    embeddings = HuggingFaceEmbeddings()
    
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