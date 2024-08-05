from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

# For LlamaIndex
from tqdm import tqdm
def extract_text_from_pdf(uploaded_files: list[str]) -> list:
    """
    Extract text from uploaded PDF files.

    Args:
        uploaded_files (list[str]): A list of file paths to the PDFs.

    Returns:
        list: A list of extracted text from each page of the PDFs.
    """
    results = []
    for uploaded_file in tqdm(uploaded_files, desc="Processing"):
        try:
            with open(uploaded_file, 'rb') as file:
                reader = PdfReader(file)
                num_pages = len(reader.pages)

                if num_pages == 0:
                    print(f"No pages found in {uploaded_file}. Skipping...")
                    continue

                for page in tqdm(range(num_pages), desc="Page"):
                    results.append(reader.pages[page].extract_text())

        except FileNotFoundError:
            print(f"File not found: {uploaded_file}")
        except Exception as e:
            print(f"An error occurred: {e}")

    return results
