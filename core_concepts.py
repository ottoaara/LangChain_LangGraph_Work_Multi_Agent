from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model # New Way to import chat models

load_dotenv()


def demo_basic_chain():

    """ Demonstrates a basic chain using LCEL and Runables."""
    
    #Component 1: Define a prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the following question in one sentence: {question}"
    )
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser =StrOutputParser()
    
    # Compose with pipe operator
    chain = prompt | model | parser 
    
    # Execute the chain with an input
    
    result = chain.invoke({"question": "What is LangChain?"})
    
    print(f"Response: {result}")
    
    return chain 

def demo_batch_execution():
    """Demonstrates executing a chain on a batch of inputs."""
    prompt =ChatPromptTemplate.from_template(
        "Translate to French: {text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()
    
    chain = prompt | model | parser 
    
    # Batch - run with multiple inputs
    inputs = [{"text": "Hello, how are you?"}, 
              {"text": "What is your name?"},
              {"text": "Where is the nearest restaurant?"}]
    
    results = chain.batch(inputs)
    
    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")
    
def demo_streaming():
    
    """Demonstrates executing a chain on a batch of inputs."""
    prompt =ChatPromptTemplate.from_template(
        "Tell me a story about: {text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)
    parser = StrOutputParser()
    
    chain = prompt | model | parser 
    
    #Streaming
    print("Streaming output: ")
    for chunk in chain.stream([{"topic": "nature"}]):
        print(chunk, end ="", flush = True)
    
    print() # new line streaming

def demo_schema_inspection():
    """Demonstrates how to inspect the schema of a chain."""
    prompt =ChatPromptTemplate.from_template("Summarize the following text: {text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()
    
    chain = prompt | model | parser 
    
    #Inspect schema
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()
    
    print(f"Input schema: {input_schema}")
    print(f"Output schema: {output_schema}") 
                              
def exercise_first_chain():
    """Exercise: Build and execute a simple chain that takes a user's name as input and returns a greeting."""
    prompt = ChatPromptTemplate.from_template("Create a product tag line for: {product}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()
    
    chain = prompt | model | parser 
    
    result = chain.invoke({"product": "Fishing course for beginners"})
    
    print(f"This is your new product tagline: {result}")
      
# this new way to initialize chat models, we can see that it is more flexible and allows us to easily switch between different providers and models without changing the underlying code structure. This abstraction can help us write more modular and maintainable code, as we can easily swap out the model or provider without affecting the rest of our application. Additionally, this approach can make it easier to manage dependencies and configurations for different models and providers, as we can centralize this logic in one place. Overall, this new way of initializing chat models can enhance the scalability and adaptability of our applications that rely on language models.    
def new_way():
    model = init_chat_model("gpt-4o-mini", temperature=0.7, max_tokens=1500)
    # looking at this new way to initialize chat models, we can see that it is more flexible and allows us to easily switch between different providers and models without changing the underlying code structure. This abstraction can help us write more modular and maintainable code, as we can easily swap out the model or provider without affecting the rest of our application. Additionally, this approach can make it easier to manage dependencies and configurations for different models and providers, as we can centralize this logic in one place. Overall, this new way of initializing chat models can enhance the scalability and adaptability of our applications that rely on language models. 
                     
if __name__ == "__main__":
    # demo_basic_chain()
    #demo_batch_execution()
    # demo_streaming()
    # demo_schema_inspection()
     exercise_first_chain()