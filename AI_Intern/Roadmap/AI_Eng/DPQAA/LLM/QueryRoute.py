from enum import Enum
from pydantic import BaseModel, Field

class QueryType(str, Enum):
    FPT = "FPT"
    COMPUTE = "COMPUTE"
    CHIT_CHAT = "CHIT CHAT"
    RUBBISH = "RUBBISH"
    CODE = "CODE"
    GREETING = "GREETING"
    HARMFUL = "HARMFUL"
    OTHER = "OTHER"

class QueryRoute(BaseModel):
    query_type: QueryType = Field(description="The type of query")

    reasoning: str = Field(description="Brief explanation of why this route was selected")

class QueryRouteFPT(BaseModel):
    lack_context: bool = Field(
        description="True if an answer cannot be generated based on the retrieved documents"
    )