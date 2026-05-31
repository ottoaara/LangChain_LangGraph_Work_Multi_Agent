from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import tempfile

from dotenv import load_dotenv

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
# sample documents
documents = [
    Document(page_content="LangChain is a framework for building applications with LLMs.", metadata={"source": "langchain_docs", 
            "topic": "overview"}),
    Document(page_content="LangGraph is a framework for building stateful, multi-actor applications.", metadata={"source": "langgraph_docs",
            "topic": "overview"}),
    Document(page_content="LangSmith is a tool for monitoring and debugging LLM applications.", metadata={"source": "langsmith_docs",
            "topic": "overview"}),
    Document(page_content="Vector stores are databases optimized for storing and querying high-dimensional vectors, such as embeddings.", metadata={"source": "vector_store_docs",
            "topic": "vector_stores"}),
    Document(page_content="Rag is a technique for retrieving relevant information from a knowledge base to augment the capabilities of LLMs.", metadata={"source": "rag_docs",
            "topic": "rag"}),
    Document(page_content="Chroma is an open-source vector database that provides efficient storage and retrieval of embeddings.", metadata={"source": "chroma_docs",
            "topic": "database"}),
    Document(page_content="Pinecone is a managed vector database service that provides scalable and efficient storage and retrieval of embeddings.", metadata={"source": "pinecone_docs",
            "topic": "database"}),
    Document(page_content="Embeddings are dense vector representations of text that capture semantic meaning and can be used for various NLP tasks.", metadata={"source": "embeddings_docs",
            "topic": "embeddings"}),
    Document(page_content="Text splitters are tools that break down large documents into smaller chunks for easier processing and embedding generation.", metadata={"source": "text_splitters_docs",
            "topic": "text_splitters"})
]

def chroma_basics():
        with tempfile.TemporaryDirectory() as temp_dir:
                # Initialize Chroma vector store
                vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=embeddings_model,
                        persist_directory=temp_dir,
                )

                print(f"Vector store created {vector_store._collection.count()} and persistent directory: {temp_dir}")

                # Perform similarity search
                query = "What is LangChain?"
                results = vector_store.similarity_search(query, k=2)

                print(f"Similarity search results for query (top 2) '{query}':")
                for i, result in enumerate(results):
                        print(f"Result {i + 1}: {result.page_content} (Source: {result.metadata['source']})")
                        print("*" * 50)

def similarity_search_with_scores():
        with tempfile.TemporaryDirectory() as temp_dir:
                # Create Vector store for documents
                vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=embeddings_model,
                        persist_directory=temp_dir,
                )

                # Perform similarity search with scores
                query = "Explain Vector Stores?"
                results_with_scores = vector_store.similarity_search_with_score(query, k=3)

                print(f"Top 3 results with scores for query '{query}':")
                for i, (result, score) in enumerate(results_with_scores):
                        print(
                                f"\nResult {i + 1}: {result.page_content} "
                                f"(Source: {result.metadata['source']}, Score: {score})"
                        )
                        print("*" * 50)

def meta_data_filtering():
        with tempfile.TemporaryDirectory() as temp_dir:
                # Create Vector store for documents
                vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=embeddings_model,
                        persist_directory=temp_dir,
                )   
                
                query ="What databases are available?"
                
                # Perform similarity search without metadata filtering

                results = vector_store.similarity_search(query, k=5)
                print(f" Results without metadata filtering for query '{query}':")
                for i, doc in enumerate(results):
                        print(f"Result {i + 1}: {doc.page_content} (Source: {doc.metadata['source']})")
                        print("*" * 50)                
     
                # with metadata filtering
                filter_criteria = {"topic": "database"}
                filtered_results = vector_store.similarity_search(query, k=5, filter=filter_criteria)
                print(f" Results with metadata filtering for query '{query}' and filter {filter_criteria}:")
                for i, doc in enumerate(filtered_results):
                    print(f"Result {i + 1}: {doc.page_content} (Source: {doc.metadata['source']})")

def as_retriever():   
        with tempfile.TemporaryDirectory() as temp_dir:
                # Create Vector store for documents
                vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=embeddings_model,
                        persist_directory=temp_dir,
                )   
                
                # basic retriever usage
                
                
                retriever = vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3})
                
                
                # use retriever to get relevant documents for a query
                query = "How do I build AI Applications?"
                
                docs =retriever.invoke(query)
                
                print(f"Retrieved documents for query '{query}':")
                for i, doc in enumerate(docs):
                    print(f"Document {i + 1}: {doc.page_content} (Source: {doc.metadata['source']})")
                    print("*" * 50)                 

                mmr_retriever = vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 3, "fetch_k": 5})
                mmr_docs = mmr_retriever.invoke(query)
                print(f"\nMMR Retrieved documents for query '{query}':")
                for i, doc in enumerate(mmr_docs):
                    print(f"Document {i + 1}: {doc.page_content} (Source: {doc.metadata['source']})")
                    print("*" * 50)

def persist_chroma():
        persist_dir = str(Path(__file__).resolve().parent / "chroma_db")
        
        # Create Vector store for documents
        vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings_model,
                persist_directory=persist_dir,
        )

        original_count = vector_store._collection.count()
        print(f"Vector store created with {original_count} documents.")
        print(f"Persistent directory: {persist_dir}")
        
        # simulator restart by loading from persist directory
        del vector_store
        
        reloaded = Chroma(
                embedding_function=embeddings_model,
                persist_directory=persist_dir,
        )
        reloaded_count = reloaded._collection.count()
        print(f"Vector store reloaded with {reloaded_count} documents.")
        
        # verify search still works after reload
        results = reloaded.similarity_search("LangChain?", k=2)
        print(f"Similarity search: {results[0].page_content[:50]}... (Source: {results[0].metadata['source']})")
        


def exercise_vector_store_setup():
        def create_retriever(raw_docs, chunk_size=200, chunk_overlap=50, k=3):
                splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                )
                docs_to_split = [Document(page_content=doc.strip()) for doc in raw_docs]
                split_docs = splitter.split_documents(docs_to_split)

                vector_store = Chroma.from_documents(
                        documents=split_docs,
                        embedding=embeddings_model,
                )

                return vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": k},
                )

        # Testing data
        docs = [
    """
    Vector stores are databases designed to store embeddings. An embedding is a numerical
    representation of text, images, or other data. Vector stores make it possible to search
    for information by meaning instead of exact keywords.
    """,

    """
    A retriever is the part of a RAG pipeline that searches the vector store and returns
    the most relevant chunks of text. The language model then uses those chunks as context
    to answer the user's question.
    """,

    """
    Text splitting is the process of breaking a large document into smaller pieces called
    chunks. Chunking helps the retriever find focused sections of text instead of searching
    through an entire long document at once.
    """,

    """
    Embeddings are created by an embedding model. The model converts text into a list of
    numbers that capture the meaning of the text. Similar ideas should have similar vectors
    even if they use different words.
    """,

    """
    Similarity search compares the vector of a user's question to the vectors stored in
    the database. The retriever returns the chunks with the closest vectors.
    """,

    """
    Maximal Marginal Relevance, or MMR, is a retrieval method that tries to balance relevance
    and diversity. Instead of returning five chunks that all say the same thing, MMR tries
    to return useful chunks that cover different angles.
    """,

    """
    In a business use case, a credit memo summarization system could store underwriting
    notes, financial statement sections, and borrower background information in a vector
    store. When the user asks a question, the retriever finds the most relevant supporting
    information.
    """,

    """
    A vector store is useful when users ask natural language questions. For example, a user
    might ask, 'Why did revenue decline last year?' The retriever can find related sections
    about revenue, pricing pressure, customer loss, or market conditions.
    """,

    """
    Chunk size matters because chunks that are too small may lose important context, while
    chunks that are too large may contain too much unrelated information. A good chunk size
    depends on the document type and the use case.
    """,

    """
    Metadata helps organize chunks in a vector store. Metadata might include the source file,
    page number, section name, company name, fiscal year, or document type.
    """
        ]

        retriever = create_retriever(docs, chunk_size=200, chunk_overlap=50, k=3)

        print("Testing Retriever:\n")
        queries = [
                "What is a vector store?",
                "How does a retriever work?",
                # "Why is text splitting important?",
                # "What are embeddings?",
                # "How does similarity search work?",
                # "What is MMR?",
        ]

        for query in queries:
                print(f"Query: {query}")
                results = retriever.invoke(query)
                for i, doc in enumerate(results):
                        preview = doc.page_content.replace("\n", " ").strip()[:120]
                        print(f"  {i + 1}. {preview}...")
                print("*" * 50)
     
        
        
        
if __name__ == "__main__":

    # chroma_basics()
    # similarity_search_with_scores()
    # meta_data_filtering()
    # persist_chroma()
    # as_retriever()
        exercise_vector_store_setup()