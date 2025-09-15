from typing import Optional

from pydantic import BaseModel


class CreateQuery(BaseModel):
    id: int = 0


class StepQuery(BaseModel):
    env_idx: int
    action: str


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool


class ResetQuery(BaseModel):
    env_idx: int
    id: Optional[int] = None 