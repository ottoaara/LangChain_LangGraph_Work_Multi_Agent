from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
import os       

load_dotenv()

#chatpprompttemplate
# prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")
# messages = prompt.format_messages(adjective="dark", topic="scooby doo")

# print(messages)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant that translates {input_language} to {output_language}."
#         ),
        
#         (
#             "human", "Translate the following text: {text}"
#         ),
#     ]
# )
# messages = prompt.format_messages(
#     input_language="English",
#     output_language="French",
#     text="I love programming."
# )                         

# print(messages)
# model = init_chat_model(model = "gpt-4o-mini", temperature=0.7)
# response = model.invoke(messages)
# #print(response.content)


# #Message Types
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# messages = [
#     SystemMessage(content="You are a helpful assistant that translates English to French."),
#     HumanMessage(content="Translate the following text: I love programming."),
#     AIMessage(content="J'aime programmer."),
#     ToolMessage(content="Tool used: translation_api", tool_call_id="translation_api_1"),
# ]


# # Few Shots examples

# from langchain_core.prompts import FewShotChatMessagePromptTemplate

# examples = [
#     {"input": "happy", "output": "sad"},
#     {"input": "tall", "output": "short"},
# ]

# example_prompt = ChatPromptTemplate.from_messages([
#   ("human", "{input}"), 
#   ("ai", "{output}"),
    
# ])

# fewshot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=example_prompt,
#     examples=examples,
# )

# final_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "Give the oppostite of each word."),
#         fewshot_prompt,
#         ("human", "{input}"),
#     ]
# )

# model = init_chat_model(model="gpt-4o-mini", temperature=0.7)
# response = model.invoke(final_prompt.format_messages(input = "fat"))

# print(response.content)


# Reusable components

system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {role}.")
    ])

user_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}")
])

# Combine
full_prompt = system_prompt + user_prompt

fin = full_prompt.format_messages(role="helpful assistant", question="What is AI?")
print(fin)