import streamlit as st
import logging
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
from tqdm import tqdm

from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.llms import Ollama

from llama_index.core import Document
from src.applications.qa_bot.modules.rag import init_chat_model
from src.applications.qa_bot.modules.vectorstore import (
    get_llama_index_retriever, 
    process_documents
)

load_dotenv()

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages the conversation history and memory for the chatbot.

    Attributes:
        memory (ConversationSummaryBufferMemory): The memory object for storing conversation history.
    """

    def __init__(self, llm: Any, max_token_limit: int = 2000):
        """
        Initialize the ConversationManager.

        Args:
            llm (Any): The language model to use for summarization.
            max_token_limit (int): The maximum number of tokens to store in memory.
        """
        # Initialize the conversation memory with the specified parameters
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=max_token_limit,
            memory_key="chat_history",
            return_messages=True,
            ai_prefix='Assistant',
            human_prefix='Human',
            output_key='answer'
        )

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role (str): The role of the message sender ('user', 'assistant', or 'system').
            content (str): The content of the message.
        """
        # Add the message to the chat memory based on the role
        if role == "user":
            self.memory.chat_memory.add_message(HumanMessage(content=content))
        elif role == "assistant":
            self.memory.chat_memory.add_message(AIMessage(content=content))
        elif role == "system":
            self.memory.chat_memory.add_message(SystemMessage(content=content))
        # Note: If an invalid role is provided, the message is silently ignored

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Retrieve the chat history.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the chat history.
        """
        # Convert the chat memory messages to a list of dictionaries
        return [
            {"role": message.type, "content": message.content} for message in self.memory.chat_memory.messages
        ]

    def clear_history(self) -> None:
        """Clear the conversation history."""
        # Reset the memory, effectively clearing the conversation history
        self.memory.clear()

def get_dynamic_greeting(model: str, model_type: str) -> str:
    """
    Generate a dynamic greeting using the Ollama model.

    Args:
        model (str): The name of the model being used.
        model_type (str): The type of the model being used.
        llm (Any): The language model object.

    Returns:
        str: A dynamically generated greeting.
    """
    logger.info(f"Generating dynamic greeting for {model_type} - {model}")
    
    # Initialize the Ollama model for generating the greeting
    if model_type == 'Ollama':
        ollama_model = Ollama(model=model)
    else:
        ollama_model = Ollama(model="gemma2")
    
    # Construct the prompt for generating the greeting
    prompt = f"""You are a friendly and helpful AI assistant. 
    Create a warm, engaging greeting for a user who has just started a conversation with you.
    Introduce yourself as if you were the AI model {model}, and provide some details about your true creator.
    Encourage the user to ask questions or share their thoughts.
    Keep the greeting concise but inviting, around 1-2 sentences."""

    # Generate the greeting using the Ollama model
    greeting = ollama_model(prompt)
    logger.info("Dynamic greeting generated successfully")
    
    # Return the greeting, stripping any leading/trailing whitespace
    return greeting.strip()

def initialize_conversation(model_type: str, model: str) -> ConversationManager:
    """
    Initialize a new conversation with a given model type and model.

    Args:
        model_type (str): The type of the model (e.g., 'OpenAI', 'Anthropic', 'Ollama').
        model (str): The specific model to use.

    Returns:
        ConversationManager: An initialized ConversationManager object.
    """
    logger.info(f"Initializing conversation with {model_type} - {model}")
    
    # Initialize the chat model
    llm = init_chat_model(model_type, model)
    
    # Create a new ConversationManager instance
    conversation_manager = ConversationManager(llm)
    
    # Generate a dynamic greeting
    greeting = get_dynamic_greeting(model, model_type)
    
    # Add the greeting to the conversation history
    conversation_manager.add_message("assistant", greeting)
    
    logger.info("Conversation initialized successfully")
    return conversation_manager

def extract_text_from_pdf(uploaded_files: List[BytesIO]) -> List[str]:
    """
    Extract text from uploaded PDF files.

    Args:
        uploaded_files (List[BytesIO]): A list of uploaded PDF files as BytesIO objects.

    Returns:
        List[str]: A list of extracted text from each PDF file.
    """
    logger.info(f"Extracting text from {len(uploaded_files)} PDF files")
    text_data = []
    
    for uploaded_file in uploaded_files:
        try:
            # Create a PDF reader object
            reader = PyPDF2.PdfReader(uploaded_file)
            num_pages = len(reader.pages)
            file_text = []
            
            # Extract text from each page of the PDF
            for page in range(num_pages):
                page_obj = reader.pages[page]
                file_text.append(page_obj.extract_text())
            
            # Join all pages' text and add to the text_data list
            text_data.append("\n".join(file_text))
            logger.info(f"Successfully extracted text from PDF with {num_pages} pages")
        except Exception as e:
            # Log any errors that occur during text extraction
            logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
    
    return text_data

def process_documents_progressively(documents: List[Document], batch_size: int = 10) -> None:
    """
    Process documents in batches, showing progress.

    Args:
        documents (List[Document]): List of documents to process.
        batch_size (int): Number of documents to process in each batch.
    """
    # Calculate the total number of batches
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    # Create a Streamlit progress bar
    progress_bar = st.progress(0)
    
    # Process documents in batches
    for i in tqdm(range(0, len(documents), batch_size), total=total_batches, desc="Processing documents"):
        # Get the current batch of documents
        batch = documents[i:i+batch_size]
        
        # Process the batch (assuming this function is defined elsewhere)
        process_documents(batch)
        
        # Update the progress bar
        progress = (i + batch_size) / len(documents)
        progress_bar.progress(min(progress, 1.0))
    
    # Clear the progress bar after processing is complete
    progress_bar.empty()
    st.success(f"Processed documents successfully!", icon="✅")

@st.cache_data(show_spinner=False)
def get_relevant_context(query: str) -> Tuple[str, List[Any]]:
    """
    Retrieve relevant context for a given query.

    Args:
        query (str): The user's query.

    Returns:
        Tuple[str, List[Any]]: A tuple containing the context string and a list of relevant documents.
    """
    logger.info(f"Retrieving context for query: {query}")
    
    # Get the LlamaIndex retriever
    llama_index_retriever = get_llama_index_retriever()
    
    try:
        # Retrieve relevant documents using the LlamaIndex retriever
        docs = llama_index_retriever.retrieve(query)
        
        # Concatenate the text from all retrieved documents
        context = "\n\n".join([doc.text for doc in docs])
        
        logger.info(f"Retrieved {len(docs)} relevant documents")
        return context, docs
    except Exception as e:
        # Log any errors that occur during retrieval
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        # Return empty context and document list if retrieval fails
        return "", []

def get_llm_response(
    context: str, 
    question: str, 
    model_type: str, 
    model: str, 
    conversation_manager: ConversationManager
) -> str:
    """
    Generate a response using the language model based on the given context, question, and chat history.

    Args:
        context (str): The relevant context for the question.
        question (str): The current question from the user.
        model_type (str): The type of the language model to use.
        model (str): The specific model to use.
        conversation_manager (ConversationManager): The conversation manager containing chat history.

    Returns:
        str: The generated response from the language model.
    """
    logger.info(f"Generating response for question: {question}")

    # Retrieve chat history from the conversation manager
    chat_history: List[Dict[str, Any]] = conversation_manager.get_chat_history()
    
    # Format chat history into a string
    formatted_chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    
    # Create the system message (prompt template)
    prompt_template = """
    You are a knowledgeable assistant created for question-answering tasks helping users find information.
    Use the provided context and chat history to give helpful and informative responses.
    Always use English and maintain a professional yet friendly tone.
    Provide clear, detailed responses using only relevant information.
    Use bullet points or numbered lists if necessary for better readability.
    If you lack information to fully answer, say so and ask for more details.
    Keep responses concise, around 3-4 sentences.

    Context: {context}

    Chat History:
    {formatted_chat_history}

    Current Question: {question}

    Please provide a response based on the above information."""

    # Initialize the language model
    llm = init_chat_model(model_type, model)

    # Create a ChatPromptTemplate from the template string
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Set up the chain: combine the prompt template with the language model
    chain = RunnableSequence(prompt | llm)

    try:
        # Invoke the chain to get the response
        response = chain.invoke({
            "context": context, 
            "formatted_chat_history": formatted_chat_history, 
            "question": question
        })
        
        logger.info("Response generated successfully")
        return response.content
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return "I apologize, but I encountered an error while generating a response. Could you please try asking your question again?"

def main():
    # st.set_page_config(page_title="Knowledge Assistant Chatbot")
    st.title("Knowledge Assistant Chatbot")

    # Sidebar for document upload and processing
    with st.sidebar:
        st.markdown("## Instructions:")
        st.markdown("1. Upload PDF files")
        st.markdown("2. Click 'Process' to analyze documents")
        st.markdown("3. Select model type and specific model")
        st.markdown("4. Start chatting with the bot")

        # File uploader for PDF documents
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        # Process button for uploaded documents
        if st.button("Process"):
            with st.spinner("Processing"):
                try:
                    # Extract text from uploaded PDFs
                    text_docs = extract_text_from_pdf(pdf_docs)
                    # Convert extracted text to Document objects
                    documents = [Document(text=doc) for doc in text_docs]
                    # Process documents progressively
                    process_documents_progressively(documents)
                    # st.success('PDF uploaded successfully!', icon="✅")
                except Exception as e:
                    logger.error(f"Error processing documents: {e}", exc_info=True)
                    st.error("An error occurred while processing the documents. Please try again.")

    # Define model categories and models
    MODEL_CATEGORIES : Dict[str, List[str]] = {
        "Ollama": ["llama3.1", "phi3", "gemma2"],
        "Anthropic": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"],
        "OpenAI": ["gpt-4o-mini", "gpt-3.5-turbo"]
    }

    # Use session state to keep track of the previous selections
    if "previous_model_type" not in st.session_state:
        st.session_state.previous_model_type = None
    if "previous_model" not in st.session_state:
        st.session_state.previous_model = None

    # Model type and specific model selection dropdowns
    with st.sidebar:
        model_type = st.selectbox('Select model type', list(MODEL_CATEGORIES.keys()))
        if model_type:
            model = st.selectbox(f"Select {model_type} model:", MODEL_CATEGORIES[model_type])

    # Check if model or model type has changed
    model_changed = (model_type != st.session_state.previous_model_type) or (model != st.session_state.previous_model)

    # Initialize or reinitialize conversation if needed
    if model_changed or "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = initialize_conversation(model_type, model)
        st.session_state.previous_model_type = model_type
        st.session_state.previous_model = model

    # Clear previous chat messages
    for key in st.session_state.keys():
        if key.startswith('message_'):
            del st.session_state[key]

    # Display chat history
    for i, message in enumerate(st.session_state.conversation_manager.get_chat_history()):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        # Store the message in session state to prevent re-running
        # st.session_state[f'message_{i}'] = message["content"]

    # Get user input and process response
    if question := st.chat_input("Enter your message:"):
        st.session_state.conversation_manager.add_message("user", question)
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrive relevant context
                    context, docs = get_relevant_context(question)
                    if not context:
                        st.warning("No relevant context found. The response may be less accurate.")
                    
                    # Generate response with the chosen language model
                    response = get_llm_response(
                        context=context, 
                        question=question, 
                        model_type=model_type, 
                        model=model,
                        conversation_manager=st.session_state.conversation_manager
                    )
                    st.write(response)
                    st.session_state.conversation_manager.add_message("assistant", response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logging.error(f"Error in processing query: {e}", exc_info=True)

    # Clear chat button
    if st.button("Clear chat"):
        st.session_state.conversation_manager = initialize_conversation(model_type, model)
        st.rerun()

if __name__ == "__main__":
    main()