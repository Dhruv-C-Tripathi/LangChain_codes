import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature= 0,
)

template = "genreate only one catchy name for a company that makes {product}" \
"keep the response one word only and do not include any other text"

first_prompt = ChatPromptTemplate.from_template(template)

first_chain = first_prompt | llm

#print(first_chain.invoke({"product": "Notebooks"}).content)

template2 = "Write a catchy phrase for the following company: {company_name}"

second_prompt = ChatPromptTemplate.from_template(template2)
second_chain = second_prompt | llm

overall_chain = first_chain | second_chain

response = overall_chain.invoke({"product": "Notebooks"})
print(response.content)
