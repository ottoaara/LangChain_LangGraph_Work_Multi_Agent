from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np


load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def basic_embedding():
   
   # single test
    text = "What is Machine learning?"
    single_embedding = embeddings.embed_query(text)
    
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")
          
def batch_embedding():
    # multiple texts
    texts = [
            "What is Machine learning?", 
             "What is Deep learning?", 
             "What is Artificial Intelligence?"
             ]
    
    batch_embeddings = embeddings.embed_documents(texts)
    for i, embed in enumerate(batch_embeddings):
        print(f"Text {i +1} - Vector Dimensions: {len(embed)}")
        print(f"Text {i +1} - First 5 values: {embed[:5]}")
        print(f"Text {i +1} - Vector norm: {np.linalg.norm(embed):.4f}")
        print("-" * 50)
    
def similiarity_search():
    
    # Documents
    docs = [
        "Python is a popular programming language known for its simplicity and versatility.",
        "Javascript is used for web development and is known for its flexibility and wide range of libraries.",
        "Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data and make predictions or decisions without being explicitly programmed.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers to model complex patterns in data.",
        "Cats are popular pets known for their independence and playful behavior."
    ]
    
    query = "What programming languages exist?"

    # embed documents and query
    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    def cosine_similarity(doc_vector, query_vector):
        return np.dot(doc_vector, query_vector) / (np.linalg.norm(doc_vector) * np.linalg.norm(query_vector))
    
    similarities = [cosine_similarity(doc_vector, query_vector) for doc_vector in doc_vector]

    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print(f"Ranked by Similiary: ")
    for doc, score in ranked_docs:
        print(f"{score:.4f} - {doc}")
        print("-" * 50)
              
# Caching ---
def embedding_caching():
    from langchain_classic.embeddings import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore 
    import tempfile 
    
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LocalFileStore(root_path=temp_dir) 
        
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings = embeddings_model,
            document_embedding_cache = store,
            namespace = "embedding_cache"
        )
        
        text = "What is reinforcement learning?"
        
        first_embedding = cached_embeddings.embed_query(text)
        
        # First call hits API
        print(f"First call(API): ")
        vectors1 = cached_embeddings.embed_documents([text])
        print(f" Embeeded {len(vectors1)} documents.")     
        
        # Second call from cache
        print(f"Second call(Cache): ")
        vectors2 = cached_embeddings.embed_documents([text])
        
        print(f" Embeeded {len(vectors2)} documents.")
        
        # Verify same results
        print(f"\nSame vectors: {np.allclose(vectors1[0], vectors2[0])}")
           
        
        
if __name__ == "__main__":
   # basic_embedding()
   # batch_embedding()
   # similiarity_search()
    embedding_caching()