from pydantic import BaseModel, Field

class Candidate_Summary(BaseModel):
    candidate: str = Field(
        description="A summarization candidate answer to the user's query using the retrieved documents"
    )

    internal_src: str | None = Field(
        description="Internal sources used or None if not sufficient"
    )

    web_src: str | None = Field(
        description="External sources retrieved from the web or None if not sufficient"
    )

    use_external: bool = Field(
        description="True if external sources needed to answer the query and internal sources was in-sufficient. False otherwise"
    )