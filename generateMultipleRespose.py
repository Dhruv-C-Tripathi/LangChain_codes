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
    "What are the top leading mobile companies?"
)

chain = prompt | llm

response = chain.batch([{} for _ in range(5)])

for res in response:
    print(res.content)
    print("----------")

