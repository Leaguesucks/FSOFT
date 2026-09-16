from API.api import API

from langchain_openai import ChatOpenAI as ChatModel

from LLM.Chatbot import Chatbot
from Tools.CodeExecutor import Executor
from Retrieval.Storage import Storage

db = Storage()
chatBot = Chatbot(db=db)

api = API(chatBot=chatBot)
app = api.app

executor = Executor()
app.include_router(executor.router)