from __future__ import annotations

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    """Pydantic model for the structured output of the function pipeline.

    Attributes:
        prompt: The original natural-language request.
        fn_name: Name of the function to invoke.
        args: Mapping from argument name to its value.
    """

    prompt: str = Field(min_length=1)
    fn_name: str = Field(min_length=1)
    args: dict[str, str | int | float | bool] = Field(default_factory=dict)
