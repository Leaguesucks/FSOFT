from pydantic import BaseModel, Field

class QueryPlan(BaseModel):
    is_parallel: bool = Field(
        description="Whether the query contains multiple independent tasks"
    )

    tasks: list[str] = Field(
        description="Independent tasks extracted from the user's query"
    )