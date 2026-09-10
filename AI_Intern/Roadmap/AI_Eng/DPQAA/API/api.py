import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from enum import Enum

from LLM.Chatbot import Chatbot

class ChatType(str, Enum):
    PLAIN_TEXT = "plain_text"
    HTTP = "http"

class Item(BaseModel):
    text: str
    chat_type: ChatType

class API:
    def __init__(self, chatBot: Chatbot):
        self.app = FastAPI()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

        self.app.get("/")(self.root)
        self.app.post("/chat")(self.chat)

        self.chatBot = chatBot

    def root(self):
        return {"message": "Welcome to the API!"}

    async def chat(self, item: Item):
        def generate():
            for chunk in self.chatBot.answer_stream(
                query=item.text,
                session_id="user_123"
            ):
                yield json.dumps(chunk) + "\n"

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson"
        )