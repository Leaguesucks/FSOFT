import json
import Prompt.Prompt as Prompt

from pydantic import BaseModel, Field
from enum import Enum
from groq import Groq

class RouteType(str, Enum):
    SEARCH_FPT = "search_fpt"
    COMPUTE = "compute"
    CODE = "code"
    GREETING = "greeting"
    CHIT_CHAT = "chit_chat"
    OTHER = "other"

class Task(BaseModel):
    query: str
    route: RouteType

class QueryPlan(BaseModel):
    tasks: list[Task]

class QueryPlanner:
    def __init__(self):
        self.client = Groq()

    def decompose(self, query: str) -> QueryPlan:

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": Prompt.query_parallelizer_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "query_plan",
                    "strict": True,
                    "schema": {
                        "type": "object",

                        "properties": {
                            "tasks": {
                                "type": "array",

                                "items": {
                                    "type": "object",

                                    "properties": {
                                        "query": {
                                            "type": "string"
                                        },

                                        "route": {
                                            "type": "string",
                                            "enum": [
                                                "search_fpt",
                                                "compute",
                                                "code",
                                                "greeting",
                                                "harmful",
                                                "chit_chat",
                                                "other"
                                            ]
                                        }
                                    },

                                    "required": [
                                        "query",
                                        "route"
                                    ],

                                    "additionalProperties": False
                                }
                            }
                        },

                        "required": [
                            "tasks"
                        ],

                        "additionalProperties": False
                    }
                }
            }
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Groq returned empty content")

        data = json.loads(content)

        return QueryPlan.model_validate(data)