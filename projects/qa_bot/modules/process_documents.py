from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# TODO: Add a function to get the text from a single PDF
def get_pdf_text(pdf_docs):
    """
    Extracts the text content from a list of PDF documents.

    Args:
        pdf_docs (list): A list of PDF document file paths or file-like objects.

    Returns:
        str: The concatenated text content of all the PDF documents.
    """
    text = ""
    # Iterate through the list of PDF documents
    for pdf in pdf_docs:
        # Create a PdfReader object to read the PDF content
        reader = PdfReader(pdf)
        # Iterate through each page of the PDF document
        for page in reader.pages:
            # Extract the text content of the current page and append it to the text variable
            text += page.extract_text()
    # Return the concatenated text content of all the PDF documents
    return text


# TODO: Add a function to break the pdf in chunks
def get_text_chunks(text):
    """
    Splits the given text into smaller chunks with a specified chunk size and overlap.

    Args:
        text (str): The input text to be split into chunks.

    Returns:
        list: A list of text chunks.
    """
    # Create a RecursiveCharacterTextSplitter object with the following settings:
    # - Separators: newline-newline, newline, period, and space
    # - Chunk size: 1000 characters
    # - Chunk overlap: 100 characters
    # - Length function: len (default)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    # Split the input text into chunks using the text splitter
    chunks = text_splitter.split_text(text)

    # Return the list of text chunks
    return chunks