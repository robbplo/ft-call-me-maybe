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

    def add_key_char(self, char: str) -> bool:
        # End key
        keys = [key for key in self.allowed_keys if key not in self.keys]
        if char == '"':
            allowed = self.current_key in keys
            if allowed:
                self.keys.append(self.current_key)
                self.allowed_keys.remove(self.current_key)
                self.current_key = ""
            return allowed
        # Add char
        index = len(self.current_key)
        keys = [key for key in keys if self.current_key == key[:index] and len(key) > index]
        chars = [key[index] for key in keys]
        allowed = char in chars
        if allowed:
            self.current_key += char
        return allowed
