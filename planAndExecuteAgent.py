# Plan and Execute Agent 

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import numexpr

load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

search = SerpAPIWrapper()

@tool
def search_tool(query: str) -> str:
    """Search the web for up-to-date information."""
    return search.run(query)

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression using numexpr."""
    try:
        expression = expression.replace("^", "**")
        result = numexpr.evaluate(expression)
        return str(result.item())
    except Exception as e:
        return f"Error evaluating expression: {e}"
    
tools = [search_tool, calculator]

agent = create_agent(llm, tools)

query = (
    "First find where the next Summer Olympics will be hosted. "
    "Then find the population of that country. "
    "Then calculate population ** 0.43 using the calculator tool."
)

response = agent.invoke({"messages": [("user", query)]})
final_message = response["messages"][-1]
print(final_message.content)