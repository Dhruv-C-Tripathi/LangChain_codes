import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
#from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature= 0.9,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


chain = prompt | llm

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistoryhi:
    """Retrieve the message history for a given session ID."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chatBot = RunnableWithMessageHistory(
    chain, 
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

print("Welcome to your AI Chatbot! What's on your mind?")

config = {"configurable": {"session_id": "user_1"}}

for _ in range(0,40):
    human_input = input("You: ")

    response = chatBot.invoke(
        {"input": human_input},
        config=config
    )
    print("AI:", response.content)

    

# chatBot.invoke(
#     {"input": "What is your name?"},
#     config=config
# ).content


# print(chatBot.invoke(
#     {"input": "Can we talk about the weather?"},
#     config=config
# ).content)

# print(chatBot.invoke(
#     {"input": "It's a beautiful day"},
#     config=config
# ).content)