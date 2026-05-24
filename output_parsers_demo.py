from langchain_core.prompts import (
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    
)

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage
)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model


load_dotenv()   

# String Output Parser

# parser = StrOutputParser()
# prompt = ChatPromptTemplate.from_template("Write a short poem about {topic}.")
# llm = init_chat_model(model="gpt-4o-mini", temperature=0.7)

# chain = prompt | llm | parser

# response = chain.invoke({"topic": "nature"})
# print(parser.invoke(response))


# Json OUtput Parser

# from langchain_core.output_parsers import JsonOutputParser

# json_parser = JsonOutputParser()
# prompt = ChatPromptTemplate.from_template("Return a JSON ojbect with 'name' and 'age' for:{description}.")
llm = init_chat_model(model="gpt-4o-mini", temperature=0.7)

# chain = prompt | llm | json_parser

# result  = chain.invoke({"description": "a 51 year old devloper named AO"})
# print(result)

# Pydantic Output Parser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# class Person(BaseModel):
#     name: str = Field(description="The person's name")
#     age: int = Field(description="The person's age")
#     occupation: str = Field(description="The person's occupation")

# py_parser = PydanticOutputParser(pydantic_object=Person)
# prompt = ChatPromptTemplate.from_template(
#     "{format_instructions}\nReturn a JSON object with name, age, and occupation for: {description}."
# ).partial(format_instructions=py_parser.get_format_instructions())

# llm = init_chat_model(model="gpt-4o-mini", temperature=0.7)

# chain = prompt | llm | py_parser

# result = chain.invoke(
#     {
#         "description": "a 51 year old developer named AO",
#         "format_instructions": py_parser.get_format_instructions(),
#     }
# )
# print(result)

# Structured Output Parser
class MovieReview(BaseModel):
    title: str = Field(description="The movie's title")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="The movie's rating out of 10")
    
structured_model = llm.with_structured_output(MovieReview)
result = structured_model.invoke("Review: Inception is a mind-bending thriller with stunning visuals and a captivating storyline. 9/10")
print(result)