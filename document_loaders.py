import os
import tempfile
from pathlib import Path 
from langchain_community.document_loaders import (
    TextLoader, WebBaseLoader,DirectoryLoader, UnstructuredFileLoader, UnstructuredURLLoader, PyPDFLoader) 
from dotenv import load_dotenv
from langchain_core.documents import Document
pdf_path = "documents/"  # Update with your PDF file path

load_dotenv()

def load_text_file():
    # Create a temporary text file for testing and demo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file for testing\n. \
                        This file is used to test text-loading function for LangChain.")
        temp_file_path = temp_file.name 
        
    try:
        # Load the text file using TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s)")
        print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")
       
        
        # Print the loaded documents
        # for doc in documents:
        #     print("Document Content")
        #     print(doc)
        #     print(doc.page_content)
    finally:        
        # Clean up temp file 
        os.remove(temp_file_path)
        
def web_loader():
    loader = WebBaseLoader("https://www.mortgageprocessinggroup.com/", bs_kwargs={"parse_only": None})
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from web")
    print(f"Source: {documents[0].metadata.get('source', 'N/A')}")
    print("Content length:", len(documents[0].page_content))
    print(f"Preview: {documents[0].page_content[:200]}...") 
      
def lazy_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temporary text file for testing and demo
        for i in range(5):
            path = Path(tmpdir) / f"doc_{i}.txt"
            path.write_text(f"This is sample text file {i} for testing lazy loading.\n"
                            f"This file is used to test lazy-loading function for LangChain.")
        
        loader = DirectoryLoader(tmpdir, glob="*.txt", 
                                       loader_cls=TextLoader)
        
        print("Initiliazing lazy loader for directory:", tmpdir)
        for doc in loader.lazy_load():
            print("Document Content Preview:", doc.page_content[:50], "...")
            print("Metadata:", doc.metadata["source"])

def doc_structure():
    doc = Document(page_content="This is a sample document content for testing.", 
                   metadata={"source": "test_source.txt", "author": "AO",
                        "length": 30,
                        "tags": ["sample", "test"],
                        "created_at": "2024-06-19T12:00:00Z"})
    print("Document Content:", doc.page_content)
    print("Document Metadata:", doc.metadata)
    return doc
                   
def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Preview:", doc.page_content[:100], "...")
        print(f"Document {i+1} Metadata:", doc.metadata)    
        
# Next PLACEHOLDER HERE #

if __name__ == "__main__":
    #load_text_file()
    # web_loader()
    #lazy_loader()
    #doc_structure()
    pdf_loader("documents/Aaron Otto - 2026-05-01.pdf")