from pydantic import BaseModel, Field

class ResolvedQuery(BaseModel):
    query: str = Field(
        description="A standalone version of the user's latest query"
    )

    required_context: bool = Field(
        description="Whether the query depends on the previous conversation"
    )

    context: str = Field(
        description="Only the previous conversation information needed to understand the query"
    )