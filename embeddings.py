# libs
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# HuggingFaceEmbeddings is a wrapper around the Hugging Face Transformers library that allows you to easily generate embeddings for text data using pre-trained models. The "sentence-transformers/all-MiniLM-L6-v2" model is a popular choice for generating sentence embeddings, which are dense vector representations of sentences that capture their semantic meaning.
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Ollama 
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="llama3:latest")

# single text

text = "Hello this is a sample text to be embedded."
embedding = embeddings.embed_query(text)
print(f"Embedding for single text: {embedding}")

print(len(embedding))

# mutliple texts
embeds = embeddings.embed_documents(
    [text, text])

print(f"Embedding for multiple texts: {embeds}")
print(f"Number of embeddings: {len(embeds)}")
print(f"Length of each embedding: {len(embeds[0])}")