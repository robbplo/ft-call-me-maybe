from dataclasses import dataclass, field
from enum import Enum, auto

class JsonState(Enum):
    START = 0
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
    depth: int
    allowed_keys: list[str]
    keys: list[str] = field(default_factory=list)
    current_key: str = ""

    def add_key_char(self, char: str) -> tuple[bool, list[str], str]:
        # Key schema only applies to depth = 2
        if self.depth != 2 or self.s != JsonState.IN_KEY:
            return (True, self.keys, self.current_key)
        # End key
        remaining_keys = [key for key in self.allowed_keys if key not in self.keys]
        keys = self.keys.copy()
        current_key = self.current_key
        if char == '"':
            allowed = self.current_key in remaining_keys
            if allowed:
                keys.append(self.current_key)
                current_key = ""
            return (allowed, keys, current_key)
        # Add char
        index = len(self.current_key)
        remaining_keys = [key for key in remaining_keys if self.current_key == key[:index] and len(key) > index]
        chars = [key[index] for key in remaining_keys]
        allowed = char in chars
        if allowed:
            current_key += char
        return (allowed, keys, current_key)
