import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub

load_dotenv()

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={
        "temperature" : 0,
        "max_length" : 64
    }
)

prompt = "what are good fitness tips"

response = llm.invoke(prompt)

print(response)