import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from enum import Enum

from Graph.graph.Graph_Main import Main_Graph

class ChatType(str, Enum):
    PLAIN_TEXT = "plain_text"
    HTTP = "http"

class Item(BaseModel):
    text: str
    chat_type: ChatType
    user_id: str
    session_id: str

class API:
    def __init__(self, graph: Main_Graph=Main_Graph()):
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

        self.graph = graph

    def root(self):
        return {"message": "Welcome to the API!"}

    def chat(self, item: Item):
        def generate():
            for event in self.graph.stream(
                query=item.text,
                user_id=item.user_id,
                session_id=item.session_id
            ):
                yield json.dumps(event) + "\n"

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson"
        )