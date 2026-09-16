from pydantic import BaseModel, Field
from enum import Enum

class RouteType(str, Enum):
    SEARCH_FPT = "search_fpt"
    COMPUTE = "compute"
    CODE = "code"
    GREETING = "greeting"
    HARMFUL = "harmful"
    CHIT_CHAT = "chit_chat"
    OTHER = "other"


class QueryPlan(BaseModel):
    tasks: dict[str, RouteType] = Field(
        # description=(
        #     "Independent tasks extracted from the user's query. "
        #     "Each task must be assigned exactly one execution route. "
        #     "Use the enum values such as 'search_fpt', "
        #     "'compute', 'code', 'greeting', 'chit_chat', "
        #     "'harmful', or 'other'."
        # )

        description=(
            "Map each task to exactly one route. "
            "Use exact lowercase enum values."
        )
    )