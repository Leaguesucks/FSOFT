from typing import TypedDict, List, Annotated, Dict
from operator import add
from enum import Enum

from langgraph.graph import add_messages
from langchain.messages import AnyMessage

from Query.QueryParallelizer import RouteType

def merge_search_results(
    left: dict[str, SearchTaskResult],
    right: dict[str, SearchTaskResult]
) -> dict[str, SearchTaskResult]:
    merged = dict(left)
    merged.update(right)

    return merged

class ResultType(str, Enum):
    DOCUMENTS = "documents"
    CANDIDATE_ANSWER = "candidate_answer"
    ERROR = "error"

class SearchType(str, Enum):
    INTERNAL = "internal"
    WEB = "web"

class TaskResult(TypedDict, total=False):
    task_id: int
    task: str
    result_type: str

    content: str

    instruction: str
    error: str

class SearchTaskResult(TypedDict, total=False):
    content: str
    sources: list[str]

class FPTTaskState(TypedDict, total=False):
    task: str
    task_id: int

    search_results: Annotated[Dict[str, SearchTaskResult], merge_search_results]
    task_results: Annotated[List[TaskResult], add]

class ChatState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]

    user_id: str
    session_id: str

    resolved_query: str
    previous_context: str

    tasks: Dict[str, RouteType]

    task_results: Annotated[List[TaskResult], add]

    response: str