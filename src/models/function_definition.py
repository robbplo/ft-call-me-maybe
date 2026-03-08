from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, RootModel, model_validator

PrimitiveType = Literal["float", "int", "str", "bool"]


class FunctionDefinition(BaseModel):
    fn_name: str = Field(min_length=1)
    args_names: list[str] = Field(default_factory=list)
    args_types: dict[str, PrimitiveType] = Field(default_factory=dict)
    return_type: PrimitiveType

    @model_validator(mode="after")
    def validate_arguments(self) -> FunctionDefinition:
        if self.args_names != list(self.args_types.keys()):
            raise ValueError(
                "args_names must match args_types keys in the same order."
            )
        return self


class FunctionDefinitions(RootModel[list[FunctionDefinition]]):
    @classmethod
    def from_file(cls, path: str | Path) -> FunctionDefinitions:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
