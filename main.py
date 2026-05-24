from dotenv import load_dotenv
load_dotenv()


from langchain_core import __version__ as core_version
from importlib.metadata import version as _pkg_version
lg_version = _pkg_version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")

def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    response = llm.invoke("Say 'setup comlplete!' in one word")
    print(f"Response from OpenAI: {response}")
    
    
    # #Test anthropic
    llm_anthropic = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    response_anthropic = llm_anthropic.invoke("Say 'setup comlplete!' in one word")
    print(f"Response from Anthropic: {response_anthropic}")


    print("Setup complete!")
    
if __name__ == "__main__":
    main()
