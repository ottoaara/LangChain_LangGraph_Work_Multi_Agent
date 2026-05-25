"""
Understanding Chains in LangChain v.1
LCEL patterns, composition, and debugging techniques.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", temperature=0.7)

def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in one sentence: {text}."
    )
    
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    
    result = chain.invoke(
        {
        
        "text": "LangChain is a powerful framework for building applications with language models. It provides tools for prompt management, output parsing, and chaining multiple components together."
        }
    )
    print(f"Summary: {result} ")


def demo_parallel_chain():
    """ Run multiple chains in parallel."""

    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize in two sentences: {text}."
    )

    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract 5 keywords from the following text: {text}\n return as a comma-separated list."
    )

    sentiment_prompt = ChatPromptTemplate.from_template(
        "What is the sentiment of the following text? {text}."
    )
        
        
    model = init_chat_model(model="gpt-4o-mini", temperature=0.7)

    parser = StrOutputParser()

    anlysis_chain = RunnableParallel(
        summary = summarize_prompt | model | parser,
        keywords = keywords_prompt | model | parser,
        sentiment = sentiment_prompt | model | parser,
)   

    text = """
    The new AI feature was released yesterday and it has received mixed reviews. Some users love the enhanced capabilities, 
    while others are concerned about privacy implications. Adoption rates are high.  Overall, the product launch has been a 
    huge success with record-breaking sales in the first week.  

"""

    results = anlysis_chain.invoke({"text": text})
    print("Analysis Results: ")
    print("Parallel Analysis Results: ")
    print(f"Summary: {results['summary']}")
    print(f"Keywords: {results['keywords']}")
    print(f"Sentiment: {results['sentiment']}")


def demo_passthrough_chain():
    """Demonstrate a chain that includes a passthrough component."""
    prompt = ChatPromptTemplate.from_template(
        "Original Question: {question}\n"
        "Context: {context}\n"
        "Answer the question based on the context provided."
    )

    # Simulate a retriever operation.
    def fake_retriever(_input_dict):
        return (
            "LangChain was created by Harrison Chase in 2022. "
            "It is a framework for building applications with language models. "
            "It provides tools for prompt management, output parsing, and chaining multiple components together."
        )

    chain = (
        RunnableParallel(
            context=RunnableLambda(fake_retriever),
            question=RunnablePassthrough(),
        )
        | RunnableLambda(
            lambda x: {
                "context": x["context"],
                "question": x["question"]["question"],
            }
        )
        | prompt
        | model
        | StrOutputParser()
    )

    result = chain.invoke({"question": "Who created LangChain?"})
    print(f"Answer: {result}")
    

def demo_branching_chain(): 
    
    # Different prompts for diferent intents
    code_prompt = ChatPromptTemplate.from_template(
        "You are a coding expert. Help with {input}."
    )
    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the question: {input}."   
    )
    
    # Classfier 
    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify the following question as either 'code' or 'general': {input}\nReturn only the label."
    )
    
    classifier = classifier_prompt | model | StrOutputParser()
    
    # Branching chain based on classification
    def is_code_question(input_dict):
        classification = classifier.invoke(input_dict)
        return "code" in classification.lower()
    
    branch = RunnableBranch(
        (is_code_question, code_prompt | model | StrOutputParser()), 
        general_prompt | model | StrOutputParser(), # default branch
    )
    
    # Test 
    questions = [
        "How do I reverse a list in Python?",
        "Who won world cup in 2018?"
    ]
    
    for question in questions:
        result = branch.invoke({"input": question})
        print(f"Question: {question}")
        print(f"Answer: {result[:100]}....\n")
    
    

if __name__ == "__main__":
    # demo_basic_chain()
   # demo_parallel_chain()
   # demo_passthrough_chain()
   demo_branching_chain()