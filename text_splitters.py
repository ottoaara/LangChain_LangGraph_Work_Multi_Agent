from langchain_text_splitters import(RecursiveCharacterTextSplitter, 
                                     CharacterTextSplitter, 
                                     TokenTextSplitter,
                                     SentenceTransformersTokenTextSplitter,
                                     MarkdownHeaderTextSplitter,
                                     Language,)
    
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

""" Text Splitters are used to break down large documents 
into smaller chunks that can be processed by language models.
"""

# sample documtes for testing

SAMPLE_TEXT = """

# Introduction to Machine Learning 

Machine Learning is a subset of artificial intelligence that focuses on building systems that can learn from data and improve 
their performance over time without being explicitly programmed. It has become an essential tool in various industries, 
including healthcare, finance, and marketing.

## Types of Machine Learning

### Supervised Learning
supervised learning involves training a model on a labeled dataset, where the input data is paired with the correct output. 
The model learns to make predictions based on this data.

Common algorithms include linear regression, logistic regression, and support vector machines.

### Unsupervised Learning
Unsupervised learning involves training a model on an unlabeled dataset, where the model must find patterns and relationships in the data without any guidance. 
Common algorithms include clustering and dimensionality reduction techniques

Common algorithms include k-means clustering, hierarchical clustering, and principal component analysis (PCA).

## Applications of Machine Learning

Machine learning has a wide range of applications, including:
1. Image and speech recognition
2. Natural language processing
3. Fraud detection
4. Recommendation systems
5. Autonomous vehicles

"""


SMAPLE_CODE = """
def quick_sort(arr):
    if len(arr) <= 1 
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]         
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right) 
    
def binary_search(arr, target):
    #P erforms binary search on a sorted array
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


"""

def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
        )
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Origional length: {len(SAMPLE_TEXT)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk Sizes: {[len(chunk) for chunk in chunks]}")
    print(f"First chunk preview: {chunks[0][:200]}...")
      
def chunk_size_comparison():
    sizes = [100, 200, 300, 400, 500]
    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, 
            chunk_overlap=size // 5,
            separators=["\n\n", "\n", " ", ""]
            )
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f"Chunk Size {size}: {len(chunks)} chunks") 
            
def overlap_importance():
    text = "The quick brown fox jumps over the lazy dog. The dog was not happy about it." * 10

    no_overlap_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0)
    
    with_overlap_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=20)
    
    print("Without Overlap:")
    print(f" Chunk 1 end: {no_overlap_splitter.split_text(text)[0][-20:]}")
    print(f" Chunk 2 start: {no_overlap_splitter.split_text(text)[1][:20]} ")
    
    print("With Overlap:")
    print(f" Chunk 1 end: {with_overlap_splitter.split_text(text)[0][-20:]}")
    print(f" Chunk 2 start: {with_overlap_splitter.split_text(text)[1][:20]} ")

def markdown_splitter():
    headers_to_split_on = [(
        "#","h1"),
         ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ]
    
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(SAMPLE_TEXT)
    print(f"Markdown Splitter Produced {len(chunks)}...\n")
    for i, chunk in enumerate(chunks):
        print(f"----Chunk {i+1}----\n{chunk.page_content[:200]}...\n")  
        print(f" Metadata: {chunk.metadata}\n")
        print(f"Content: {chunk.page_content[:200]}...\n")
        
def code_splitter():
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=500, chunk_overlap=50)
    
    chunks = python_splitter.split_text(SMAPLE_CODE)
    print(f"Code Splitter Produced {len(chunks)}...\n")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i} ({len(chunk)} charas):")
        print(chunk[:150] + "...\n" if len(chunk) > 150 else chunk)
     
def document_splitter():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
    
    loader = PyPDFLoader("/Users/aaronotto/Desktop/langc-course/documents/AO.pdf")
    docs = loader.load()
    
    print(f"Loaded {len(docs)} documents from PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    #split documents into chunks
    split_docs =splitter.split_documents(docs)
    
    print(f"Split into {len(split_docs)} chunks.")
    print(f"First chunk metadata: {split_docs[0].metadata}")
    print(f"First chunk content preview: {split_docs[0].page_content[:200]}...")
    print(f"\nLast chunk metadata: {split_docs[-1].metadata}")
    

if __name__ == "__main__":
    # print("=== Testing Recursive Character Text Splitter ===")
    # recursive_splitter()
    # print("=== Comparing Different Chunk Sizes ===")
    # chunk_size_comparison()
    # print("=== Importance of Overlap ===")
    # overlap_importance()  
    # print("=== Testing Markdown Header Text Splitter ===")
    #markdown_splitter()
    # print("=== Testing Code Splitter ===")
    # code_splitter()
    print("=== Testing Document Splitter ===")
    document_splitter()