import os
from dotenv import load_dotenv 
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature= 0.9,
)

prompt = ChatPromptTemplate.from_template(
    "What is the name of the company that makes {product}?"
)

chain = prompt | llm

respose = chain.invoke({"product": "apple air pods"})

print(respose.content)

# response = llm.invoke([
#     HumanMessage(content="What is the name of the company " \
#     "that makes bright red socks")
# ])

# print(response.content)