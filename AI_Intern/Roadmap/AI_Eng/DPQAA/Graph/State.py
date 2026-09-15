from typing import TypedDict, List, Annotated
from operator import add

from langgraph.graph import add_messages
from langchain.messages import AnyMessage

class TaskResult(TypedDict, total=False):
    task: str
    route: str
    context: str
    instruction: str

class ChatState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]

    user_id: str
    session_id: str

    resolved_query: str
    previous_context: str

    tasks: List[str]

    task_results: Annotated[List[TaskResult], add]

    route: str

    retrieved_context: str
    route_instruction: str

    generation_context: str
    response: str

    lack_context: bool=False