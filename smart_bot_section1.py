"""
    Q&A Bot

"""
# -- Imports --

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from langsmith import traceable, Client 
import os

load_dotenv()

# -- LangSmith Configuration --
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "smart_bot_project")
    print(f"LangSmith is configured. ~ Project: {os.getenv('LANGSMITH_PROJECT')}")
    
class QAResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    confidence: str = Field(description="Confidence level: low, medium, high")
    reasoning: str = Field(description="The reasoning behind the answer provided")
    follow_up_questions: List[str] = Field(
        description="List of follow-up questions related to the topic",
        default_factory=list,
    )
    sources_needed: bool = Field(
        description="Whether the question requires citing sources or references",
        default=False,
    )

class SmartQABot:
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ): 
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        ).with_structured_output(QAResponse)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                 """You are a knowledgeable Q&A assistant.
                 
    Your guidelines:
    - Provide clear and concise answers to user questions.
    - If the question is ambiguous, ask for clarification.
    - If the question is complex, break down your answer into steps.
    - Provide clear reasoning for your answers.
    - Suggest relevant follow-up questions to encourage further discussion.
    - Indicate if external sources would help. 
    
    Always respond with accurate, helpful information. 
    """,
                ),
                ("human", "{question}"),
            ]
        )
        self.chain = self.prompt | self.model

    @traceable(name="ask_question", run_type="chain")
    def ask(self, question: str) -> QAResponse:
        try:
            response = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            return QAResponse(
                answer="Sorry, I encountered an error while processing your question.",
                confidence="low",
                reasoning=str(e),
                follow_up_questions=[],
                sources_needed=True,
            )

    @traceable(name="ask_batch", run_type="chain")
    def ask_batch(self, questions: List[str]) -> List[QAResponse]:
        """Ask multiple questions in parallel."""
        inputs = [{"question": q} for q in questions]
        return self.chain.batch(inputs)
            
# Demo Usage

def demo_qa_bot():
    bot = SmartQABot()
    questions =[
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What are the health benefits of meditation?",
        
        ] 
    
    print("=" * 60)
    print("SMART Q&A BOT DEMO")
    print("=" * 60)          
    
    for question in questions: 
        
        print(f"Question: {question}")
        print("-" * 40)
        
        response = bot.ask(question)
        
        print(f"Question: {question}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Follow-up Questions: {response.follow_up_questions}")
        print(f"Sources Needed: {response.sources_needed}")
        print("-" * 60)
        

@traceable(name="demo_error_handling", run_type="chain")
def demo_error_handling(question: str) -> dict:
    """Demonstrate error handling with explicit traced input/output."""

    bot = SmartQABot()
    
    print("=" * 60)
    print("ERROR HANDLING DEMO")
    print("=" * 60)          
    
    response = bot.ask(question)
    print(f"Handled gracefully: {response.confidence}") 
    return {
        "question": question,
        "response": response.model_dump(),
    }

@traceable(name="demo_batch_processing", run_type="chain")
def demo_batch_processing(questions: List[str]) -> dict:
    """Demonstrate batch processing with explicit traced input/output."""
    bot = SmartQABot()

    print("=" * 60)
    print("BATCH PROCESSING DEMO")
    print("=" * 60)          
    
   
    
    responses = bot.ask_batch(questions)
    
    for q, r in zip(questions, responses):
        print(f"Question: {q}")
        print(f"Answer: {r.answer}")
        print(f"Confidence: {r.confidence}")
        print("-" * 60)

    return {
        "questions": questions,
        "responses": [r.model_dump() for r in responses],
    }


if __name__ == "__main__":
    try:
        demo_qa_bot()
        demo_error_handling("What is the meaning of life? " * 100 + "important?")
        demo_batch_processing(
            [
                "What is the tallest mountain in the world?",
                "How do airplanes fly?",
                "What are the benefits of regular exercise?",
                "Who won the World Cup in 2018?",
                "What is quantum computing?",
            ]
        )
    finally:
        Client().flush()