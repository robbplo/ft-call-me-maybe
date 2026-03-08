from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    prompt: str = Field(min_length=1)
    fn_name: str = Field(min_length=1)
    args: dict[str, str | int | float | bool] = Field(default_factory=list)

