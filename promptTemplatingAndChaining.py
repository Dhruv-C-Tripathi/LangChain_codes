import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# PROMPT_TEMPLATE: reproducable way to generate a prompt, a text 
#                  string that accepts parameters and uses them 
#                  to format the final prompts.

load_dotenv()

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature= 0.9
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a {designation} for a new company.
    Suggest one creative brand name for a company that makes {product}.
    Only return the name.
    """
)

chain = prompt | llm 

response = chain.invoke({
    "product" : "Notebooks",
    "designation" : "naming consultant"
})

print(response.content)



# Structured JSON Output
#
# from langchain_core.output_parsers import JsonOutputParser
# from pydantic import BaseModel

# class CompanyName(BaseModel):
#     name: str
#     tagline: str

# parser = JsonOutputParser(pydantic_object=CompanyName)

# prompt = ChatPromptTemplate.from_template(
#     """
#     Suggest a company name and tagline for a company that makes {product}.
#     Return response in JSON format.
#     """
# )

# chain = prompt | llm | parser

# response = chain.invoke({"product": "Notebooks"})

# print(response)
