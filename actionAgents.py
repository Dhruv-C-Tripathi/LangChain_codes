# There are 2 types of Agents: Action Agent and Plan-and-execute Agent.

# Proper Modern Agent using LangGraph

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
import numexpr as ne
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain.agents import create_agent

load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)


# Wikipedia tool
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)

# Calculator tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression using numexpr."""
    try:
        #print("Expression received:", expression)  

        expression = expression.replace("^", "**")
        result = ne.evaluate(expression)

        return str(result.item())
    except Exception as e:
        return f"Error evaluating expression: {e}"

tools = [wiki, calculator]


agent = create_agent(llm, tools)


query = (
    """Step 1: Find who was the 3rd president of India.
    Step 2: Extract his birth year.
    Step 3: Call the calculator tool using the exact birth year from Step 2 and compute birth_year ** 3.
    Do not guess the year."""
)

response = agent.invoke({"messages": [("user", query)]})
final_message = response["messages"][-1]
print(final_message.content)


