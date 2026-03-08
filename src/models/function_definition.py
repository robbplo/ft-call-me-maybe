from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field, RootModel, model_validator

PrimitiveType = Literal["float", "int", "str", "bool"]


class FunctionDefinition(BaseModel):
    """Pydantic model representing a single callable function.

    Attributes:
        fn_name: Unique name of the function.
        args_names: Ordered list of argument names.
        args_types: Mapping from argument name to its primitive type.
        return_type: Primitive type of the function's return value.
    """

    fn_name: str = Field(min_length=1)
    args_names: list[str] = Field(default_factory=list)
    args_types: dict[str, PrimitiveType] = Field(default_factory=dict)
    return_type: PrimitiveType

    @model_validator(mode="after")
    def validate_arguments(self) -> FunctionDefinition:
        """Ensure *args_names* and *args_types* keys are identical and ordered.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the names and keys do not match.
        """
        if self.args_names != list(self.args_types.keys()):
            raise ValueError(
                "args_names must match args_types keys in the same order."
            )
        return self


class FunctionDefinitions(RootModel[list[FunctionDefinition]]):
    """Pydantic root model wrapping a list of :class:`FunctionDefinition`."""

    @classmethod
    def from_file(cls, path: str | Path) -> FunctionDefinitions:
        """Load and validate function definitions from a JSON file.

        Args:
            path: Path to the JSON file with an array of function definitions.

        Returns:
            A :class:`FunctionDefinitions` instance populated from the file.
        """
        return cast(
            FunctionDefinitions,
            cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
        )
