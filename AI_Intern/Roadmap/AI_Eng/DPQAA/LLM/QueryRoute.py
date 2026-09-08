from enum import Enum
from pydantic import BaseModel, Field

class QueryType(str, Enum):
    DOCUMENT = "DOCUMENT"
    FPT = "FPT"
    COMPUTE = "COMPUTE"
    GENERAL = "GENERAL"
    RUBBISH = "RUBBISH"
    LACK_CONTEXT = "LACK CONTEXT"
    CODE = "CODE"

class QueryRoute(BaseModel):
    query_type: QueryType = Field(description="The type of query")

    reasoning: str = Field(description="Brief explanation of why this route was selected")