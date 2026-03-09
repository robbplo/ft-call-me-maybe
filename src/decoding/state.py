from dataclasses import dataclass, field
from enum import Enum, auto

from src.models.function_definition import PrimitiveType


class JsonState(Enum):
    """States for the JSON FSM used during constrained decoding."""

    START = 0
    EXPECT_VALUE = auto()
    EXPECT_KEY = auto()
    EXPECT_COMMA_OR_END = auto()
    EXPECT_COLON = auto()
    IN_KEY = auto()
    IN_STRING = auto()
    NUMBER_AFTER_MINUS = auto()
    IN_NUMBER_ZERO = auto()
    IN_NUMBER = auto()
    NUMBER_AFTER_DOT = auto()
    IN_FRACTION = auto()
    NUMBER_AFTER_EXP = auto()
    NUMBER_AFTER_EXP_SIGN = auto()
    IN_EXPONENT = auto()
    IN_TRUE_T = auto()
    IN_TRUE_TR = auto()
    IN_TRUE_TRU = auto()
    IN_FALSE_F = auto()
    IN_FALSE_FA = auto()
    IN_FALSE_FAL = auto()
    IN_FALSE_FALS = auto()
    INVALID = auto()
    END = auto()


@dataclass()
class State:
    """Mutable decoding state tracking JSON structure and schema constraints.

    Attributes:
        s: Current JSON parser state.
        depth: Current nesting depth in the JSON object.
        allowed_keys: Exhaustive list of keys permitted at depth 2.
        allowed_types: Mapping from allowed keys to their primitive types.
        keys: Keys that have already been emitted at depth 2.
        current_key: Key string being accumulated character by character.
        current_value_key: Key whose value is currently being decoded.
        current_value_buffer: Prefix of the current value for type validation.
    """

    s: JsonState
    depth: int
    allowed_keys: list[str]
    allowed_types: dict[str, PrimitiveType] = field(default_factory=dict)
    keys: list[str] = field(default_factory=list)
    current_key: str = ""
    current_value_key: str | None = None
    current_value_buffer: str = ""

    def remaining_keys(self, keys: list[str] | None = None) -> list[str]:
        """Return the keys that have not been emitted yet."""
        used_keys = self.keys if keys is None else keys
        return [key for key in self.allowed_keys if key not in used_keys]

    def add_key_char(
        self, char: str
    ) -> tuple[bool, list[str], str, str | None]:
        """Validate and accumulate one character of a key at depth 2.

        Args:
            char: The next character to append to the current key, or ``"``
                to signal that the key has ended.

        Returns:
            A four-tuple ``(allowed, keys, current_key, current_value_key)``
            where *allowed* indicates whether the character is valid given
            the schema, *keys* is the updated list of completed keys,
            *current_key* is the updated in-progress key string, and
            *current_value_key* is the key whose value should be decoded next.
        """
        # Key schema only applies to depth = 2
        if self.depth != 2 or self.s != JsonState.IN_KEY:
            return (
                True,
                self.keys,
                self.current_key,
                self.current_value_key,
            )
        # End key
        remaining_keys = self.remaining_keys()
        keys = self.keys.copy()
        current_key = self.current_key
        current_value_key = self.current_value_key
        if char == '"':
            allowed = self.current_key in remaining_keys
            if allowed:
                keys.append(self.current_key)
                current_value_key = self.current_key
                current_key = ""
            return (allowed, keys, current_key, current_value_key)
        # Add char
        index = len(self.current_key)
        remaining_keys = [
            key for key in remaining_keys
            if self.current_key == key[:index] and len(key) > index
        ]
        chars = [key[index] for key in remaining_keys]
        allowed = char in chars
        if allowed:
            current_key += char
        return (allowed, keys, current_key, current_value_key)
