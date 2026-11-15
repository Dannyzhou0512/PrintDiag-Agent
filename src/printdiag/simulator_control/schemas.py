"""Simulator control API schemas."""

from typing import Literal
from pydantic import BaseModel, Field


class ArmRequest(BaseModel):
    confirm_local_simulator: bool = Field(default=False)


class JogRequest(BaseModel):
    axis: Literal["x", "y", "z"]
    direction: Literal["+", "-"]
    step: float = Field(ge=0.1, le=10.0)
