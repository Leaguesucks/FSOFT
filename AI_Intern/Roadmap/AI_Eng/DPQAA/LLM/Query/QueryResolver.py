import json
import Prompt.Prompt as Prompt

from pydantic import BaseModel
from enum import Enum
from groq import Groq, BadRequestError

class ResolverStatus(str, Enum):
    OK = "ok"
    HARMFUL = "harmful"

class ResolvedQuery(BaseModel):
    status: ResolverStatus

    query: str
    context: str

    rejection_reason: str

class QueryResolver:
    def __init__(self):
        self.client = Groq()

    def resolve(
        self,
        conversation: str,
        latest_query: str,
    ) -> ResolvedQuery:

        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": Prompt.history_resolver_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"""
                            CONVERSATION HISTORY:
                            {conversation}

                            LATEST USER MESSAGE:
                            {latest_query}
                        """,
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "resolved_query",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "ok",
                                        "harmful",
                                    ],
                                },
                                "query": {
                                    "type": "string",
                                },
                                "context": {
                                    "type": "string",
                                },
                                "rejection_reason": {
                                    "type": "string",
                                },
                            },
                            "required": [
                                "status",
                                "query",
                                "context",
                                "rejection_reason",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
            )
        except BadRequestError:
            return ResolvedQuery(
                status=ResolverStatus.HARMFUL,
                query="",
                context="",
                rejection_reason="The request was rejected in accordance with the system's safty guidelines"
            )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                "Query resolver returned empty content"
            )

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Query resolver returned invalid JSON: {content}"
            ) from e

        return ResolvedQuery.model_validate(data)