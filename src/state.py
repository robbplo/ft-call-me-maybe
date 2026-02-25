from dataclasses import dataclass, field
from enum import StrEnum, auto

class JsonState(StrEnum):
    START = auto()
    EXPECT_VALUE = auto()
    EXPECT_KEY = auto()
    EXPECT_COMMA_OR_END = auto()
    EXPECT_COLON = auto()
    IN_KEY = auto()
    IN_STRING = auto()
    IN_NUMBER = auto()
    INVALID = auto()
    END = auto()

@dataclass()
class State:
    s: JsonState
    allowed_keys: list[str]
    keys: list[str] = field(default_factory=list)
    current_key: str = ""

    def key_char_valid(self, char: str) -> bool:
        index = len(self.current_key)
        keys = [key for key in self.allowed_keys if self.current_key == key[:index] and len(key) > index]
        chars = [key[index] for key in keys]
        return char in chars
