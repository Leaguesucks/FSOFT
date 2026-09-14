from typing import TypedDict, List, Annotated

from langgraph.graph import add_messages
from langchain.messages import AnyMessage

from LLM.QueryRoute import QueryType

class ChatState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]

    user_id: str
    session_id: str

    resolved_query: str
    previous_context: str

    route: QueryType

    retrieved_context: str
    route_instruction: str

    generation_context: str
    response: str

    lack_context: bool=False