from API.api import API
from dotenv import load_dotenv
from pathlib import Path
from os import getenv
from langchain_openai import ChatOpenAI as ChatModel

from LLM.Chatbot import Chatbot
from Tools.CodeExecutor import Executor
from Retrieval.Storage import Storage

api_keys_path = Path(".secrets/api_keys.secrets")
load_dotenv(api_keys_path)
openAI_api_keys = getenv("OPENAI_API_KEY")

llm = ChatModel(
    api_key=openAI_api_keys,
    model="gpt-5-nano"
)

db = Storage()
chatBot = Chatbot(db=db, llm=llm)

api = API(chatBot=chatBot)
app = api.app

executor = Executor()
app.include_router(executor.router)