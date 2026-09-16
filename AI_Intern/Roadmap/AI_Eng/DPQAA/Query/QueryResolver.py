from pydantic import BaseModel, Field

class ResolvedQuery(BaseModel):
    query: str

    context: str