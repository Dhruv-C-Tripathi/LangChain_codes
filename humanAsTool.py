import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature= 0
)
# if we want to use human as a tool then we must provide the context
# first as in modern agents, this is handled naturally via conversation.
# i.e persistent memory and conversational context. 
# So it won't know anything unless we provide contect earlier

agent = create_agent(llm, tools=[])

query = (
    """what is the name of my friend who was born in 1990?"""
)

response = agent.invoke({"messages": [("user",query)]})
final_message = response["messages"][-1]
print(final_message.content)

