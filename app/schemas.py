from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Broad research topic")
    session_id: str | None = Field(None, description="Reuse session for follow-up")


class FeedbackRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Correction or guidance mid-research")


class ResearchResponse(BaseModel):
    session_id: str
    status: str = "started"
