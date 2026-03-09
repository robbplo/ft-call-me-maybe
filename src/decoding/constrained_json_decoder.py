from typing import Protocol

from .state import JsonState, State
from src.vocabulary import Vocabulary


WHITESPACE = set(" \t\n\r")
DIGITS = set("0123456789")
NON_ZERO_DIGITS = set("123456789")
MAX_DEPTH = 2
NUMBER_STATES = {
    JsonState.NUMBER_AFTER_MINUS,
    JsonState.IN_NUMBER_ZERO,
    JsonState.IN_NUMBER,
    JsonState.NUMBER_AFTER_DOT,
    JsonState.IN_FRACTION,
    JsonState.NUMBER_AFTER_EXP,
    JsonState.NUMBER_AFTER_EXP_SIGN,
    JsonState.IN_EXPONENT,
}
COMPLETE_NUMBER_STATES = {
    JsonState.IN_NUMBER_ZERO,
    JsonState.IN_NUMBER,
    JsonState.IN_FRACTION,
    JsonState.IN_EXPONENT,
}
BOOLEAN_STATES = {
    JsonState.IN_TRUE_T,
    JsonState.IN_TRUE_TR,
    JsonState.IN_TRUE_TRU,
    JsonState.IN_FALSE_F,
    JsonState.IN_FALSE_FA,
    JsonState.IN_FALSE_FAL,
    JsonState.IN_FALSE_FALS,
}


class Tokenizer(Protocol):
    """Protocol for objects that can decode token IDs to strings."""

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs to a string."""
        ...


class ConstrainedJSONDecoder:
    """Token-by-token JSON decoder with structural and schema constraints."""

    def __init__(self, tokenizer: Tokenizer, vocab: Vocabulary) -> None:
        """Initialize the decoder and pre-compute structural masks."""
        self.model: Tokenizer = tokenizer
        self.vocab: Vocabulary = vocab
        self.token_bytes: list[str] = [
            tokenizer.decode([i]) for i in range(vocab.size)]
        self.structural_masks: dict[tuple[JsonState, int], list[float]] = {
            (state, depth): self._get_structural_mask(state, depth)
            for state in JsonState
            for depth in range(MAX_DEPTH + 1)
        }

    def get_logit_mask(self, state: State) -> list[float]:
        """Return the combined structural and schema logit mask."""
        structural_mask = self.structural_masks[state.s, state.depth]
        schema_mask = self._get_schema_mask(state)
        return [a + b for a, b in zip(structural_mask, schema_mask)]

    def _get_schema_mask(self, state: State) -> list[float]:
        """Compute the schema-level logit mask for depth-2 args decoding."""
        if state.depth != 2:
            return [0.0] * self.vocab.size

        mask: list[float] = [-float("inf")] * self.vocab.size
        for token_id, token_str in enumerate(self.token_bytes):
            allowed, _, _, _, _ = self._simulate_schema(state, token_str)
            if allowed:
                mask[token_id] = 0.0
        return mask

    def _simulate_schema(
        self, state: State, token_str: str, prints: bool = False
    ) -> tuple[bool, list[str], str, str | None, str]:
        """Simulate appending *token_str* and check schema validity."""
        del prints

        orig_s, orig_depth = state.s, state.depth
        orig_keys, orig_current_key = state.keys, state.current_key
        orig_current_value_key = state.current_value_key
        orig_current_value_buffer = state.current_value_buffer

        allowed = True
        keys = state.keys.copy()
        current_key = state.current_key
        current_value_key = state.current_value_key
        current_value_buffer = state.current_value_buffer
        s = state.s
        depth = state.depth

        for char in token_str:
            if depth == 2:
                (
                    allowed,
                    keys,
                    current_key,
                    current_value_key,
                    current_value_buffer,
                ) = self._simulate_schema_char(
                    state,
                    s,
                    depth,
                    char,
                    keys,
                    current_key,
                    current_value_key,
                    current_value_buffer,
                )
                if not allowed:
                    break

            s, depth = self._simulate_structure_char(s, char, depth)
            if s == JsonState.INVALID:
                allowed = False
                break

        state.s, state.depth = orig_s, orig_depth
        state.keys, state.current_key = orig_keys, orig_current_key
        state.current_value_key = orig_current_value_key
        state.current_value_buffer = orig_current_value_buffer
        return (
            allowed,
            keys,
            current_key,
            current_value_key,
            current_value_buffer,
        )

    def _simulate_schema_char(
        self,
        state: State,
        json_state: JsonState,
        depth: int,
        char: str,
        keys: list[str],
        current_key: str,
        current_value_key: str | None,
        current_value_buffer: str,
    ) -> tuple[bool, list[str], str, str | None, str]:
        """Simulate one schema-relevant character at depth 2."""
        state.s = json_state
        state.depth = depth
        state.keys = keys
        state.current_key = current_key
        state.current_value_key = current_value_key
        state.current_value_buffer = current_value_buffer

        if json_state == JsonState.IN_KEY:
            allowed, keys, current_key, current_value_key = state.add_key_char(
                char)
            return (
                allowed,
                keys,
                current_key,
                current_value_key,
                current_value_buffer,
            )

        allowed, current_value_key, current_value_buffer = (
            self._consume_value_char(state, json_state, char)
        )
        if not allowed:
            return (
                False,
                keys,
                current_key,
                current_value_key,
                current_value_buffer,
            )

        if self._is_depth_2_object_close(json_state, char):
            if state.remaining_keys(keys):
                return (
                    False,
                    keys,
                    current_key,
                    current_value_key,
                    current_value_buffer,
                )

        return (
            True,
            keys,
            current_key,
            current_value_key,
            current_value_buffer,
        )

    def _consume_value_char(
        self, state: State, json_state: JsonState, char: str
    ) -> tuple[bool, str | None, str]:
        """Validate a character against the active depth-2 value type."""
        current_value_key = state.current_value_key
        current_value_buffer = state.current_value_buffer

        if current_value_key is None:
            return True, current_value_key, current_value_buffer

        value_type = state.allowed_types.get(current_value_key)
        if value_type is None:
            return False, current_value_key, current_value_buffer

        if json_state == JsonState.EXPECT_COLON:
            return True, current_value_key, current_value_buffer

        if json_state == JsonState.EXPECT_VALUE:
            if char in WHITESPACE:
                return True, current_value_key, current_value_buffer
            return self._start_typed_value(value_type, char, current_value_key)

        if json_state == JsonState.IN_STRING:
            if value_type != "str":
                return False, current_value_key, current_value_buffer
            current_value_buffer += char
            if len(current_value_buffer) > 1 and current_value_buffer[-1] == '"':
                return True, None, ""
            return True, current_value_key, current_value_buffer

        if json_state in BOOLEAN_STATES:
            if value_type != "bool":
                return False, current_value_key, current_value_buffer
            current_value_buffer += char
            if not any(
                literal.startswith(current_value_buffer)
                for literal in ("true", "false")
            ):
                return False, current_value_key, current_value_buffer
            if current_value_buffer in {"true", "false"}:
                return True, None, ""
            return True, current_value_key, current_value_buffer

        if json_state in NUMBER_STATES:
            if value_type not in {"int", "float"}:
                return False, current_value_key, current_value_buffer
            return self._consume_numeric_char(
                value_type,
                json_state,
                char,
                current_value_key,
                current_value_buffer,
            )

        return True, current_value_key, current_value_buffer

    def _start_typed_value(
        self, value_type: str, char: str, current_value_key: str
    ) -> tuple[bool, str | None, str]:
        """Validate the first non-whitespace character of a typed value."""
        match value_type:
            case "str":
                if char != '"':
                    return False, current_value_key, ""
            case "bool":
                if char not in {"t", "f"}:
                    return False, current_value_key, ""
            case "int" | "float":
                if char not in DIGITS | {"-"}:
                    return False, current_value_key, ""
            case _:
                return False, current_value_key, ""
        return True, current_value_key, char

    def _consume_numeric_char(
        self,
        value_type: str,
        json_state: JsonState,
        char: str,
        current_value_key: str,
        current_value_buffer: str,
    ) -> tuple[bool, str | None, str]:
        """Validate a numeric character or numeric delimiter."""
        if char in WHITESPACE or char in {",", "}"}:
            if json_state not in COMPLETE_NUMBER_STATES:
                return False, current_value_key, current_value_buffer
            if value_type == "int" and not self._is_integer_number(
                current_value_buffer
            ):
                return False, current_value_key, current_value_buffer
            return True, None, ""

        if value_type == "int" and char in {".", "e", "E"}:
            return False, current_value_key, current_value_buffer

        current_value_buffer += char
        return True, current_value_key, current_value_buffer

    def _is_integer_number(self, value: str) -> bool:
        """Return whether the current number prefix is integer-only."""
        return (
            bool(value)
            and value != "-"
            and "." not in value
            and "e" not in value
            and "E" not in value
        )

    def _is_depth_2_object_close(
        self, json_state: JsonState, char: str
    ) -> bool:
        """Return whether *char* can close the depth-2 args object."""
        return char == "}" and (
            json_state == JsonState.EXPECT_COMMA_OR_END
            or json_state in COMPLETE_NUMBER_STATES
        )

    def _get_structural_mask(
        self, json_state: JsonState, depth: int
    ) -> list[float]:
        """Pre-compute the structural logit mask for a given state and depth."""
        mask: list[float] = [-float("inf")] * self.vocab.size
        for token_id, token_str in enumerate(self.token_bytes):
            next_state, _ = self._simulate_structure(
                json_state, token_str, depth)
            if next_state is not JsonState.INVALID:
                mask[token_id] = 0.0
        return mask

    def _simulate_structure(
        self, json_state: JsonState, token: str, depth: int
    ) -> tuple[JsonState, int]:
        """Simulate appending *token* character by character through the FSM."""
        for char in token:
            json_state, depth = self._simulate_structure_char(
                json_state, char, depth)
        return json_state, depth

    def _simulate_structure_char(
        self, json_state: JsonState, char: str, depth: int
    ) -> tuple[JsonState, int]:
        """Advance the JSON structural FSM by one character."""
        next_state = json_state
        match json_state:
            case JsonState.INVALID:
                pass
            case JsonState.START:
                match char:
                    case "{":
                        next_state = JsonState.EXPECT_KEY
                        depth += 1
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.EXPECT_KEY:
                match char:
                    case '"':
                        next_state = JsonState.IN_KEY
                    case char if char in WHITESPACE:
                        next_state = JsonState.EXPECT_KEY
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.IN_KEY:
                match char:
                    case '"':
                        next_state = JsonState.EXPECT_COLON
                    case _:
                        next_state = JsonState.IN_KEY
            case JsonState.IN_STRING:
                match char:
                    case '"':
                        next_state = JsonState.EXPECT_COMMA_OR_END
                    case _:
                        next_state = JsonState.IN_STRING
            case JsonState.NUMBER_AFTER_MINUS:
                match char:
                    case "0":
                        next_state = JsonState.IN_NUMBER_ZERO
                    case char if char in NON_ZERO_DIGITS:
                        next_state = JsonState.IN_NUMBER
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.IN_NUMBER_ZERO:
                match char:
                    case ".":
                        next_state = JsonState.NUMBER_AFTER_DOT
                    case "e" | "E":
                        next_state = JsonState.NUMBER_AFTER_EXP
                    case char if char in WHITESPACE:
                        next_state = JsonState.EXPECT_COMMA_OR_END
                    case ",":
                        next_state = JsonState.EXPECT_KEY
                    case "}":
                        depth -= 1
                        if depth == 0:
                            next_state = JsonState.END
                        else:
                            next_state = JsonState.EXPECT_COMMA_OR_END
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.EXPECT_COLON:
                match char:
                    case ":":
                        next_state = JsonState.EXPECT_VALUE
                    case char if char in WHITESPACE:
                        next_state = JsonState.EXPECT_COLON
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.EXPECT_VALUE:
                match char:
                    case "{" if depth < MAX_DEPTH:
                        next_state = JsonState.EXPECT_KEY
                        depth += 1
                    case '"':
                        next_state = JsonState.IN_STRING
                    case "-":
                        next_state = JsonState.NUMBER_AFTER_MINUS
                    case "0":
                        next_state = JsonState.IN_NUMBER_ZERO
                    case "t":
                        next_state = JsonState.IN_TRUE_T
                    case "f":
                        next_state = JsonState.IN_FALSE_F
                    case char if char in WHITESPACE:
                        next_state = JsonState.EXPECT_VALUE
                    case char if char in NON_ZERO_DIGITS:
                        next_state = JsonState.IN_NUMBER
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.EXPECT_COMMA_OR_END:
                match char:
                    case ",":
                        next_state = JsonState.EXPECT_KEY
                    case "}":
                        depth -= 1
                        if depth == 0:
                            next_state = JsonState.END
                        else:
                            next_state = JsonState.EXPECT_COMMA_OR_END
                    case char if char in WHITESPACE:
                        next_state = JsonState.EXPECT_COMMA_OR_END
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.IN_NUMBER:
                match char:
                    case char if char in DIGITS:
                        next_state = JsonState.IN_NUMBER
                    case ".":
                        next_state = JsonState.NUMBER_AFTER_DOT
                    case "e" | "E":
                        next_state = JsonState.NUMBER_AFTER_EXP
                    case char if char in WHITESPACE:
                        next_state = JsonState.EXPECT_COMMA_OR_END
                    case ",":
                        next_state = JsonState.EXPECT_KEY
                    case "}":
                        depth -= 1
                        if depth == 0:
                            next_state = JsonState.END
                        else:
                            next_state = JsonState.EXPECT_COMMA_OR_END
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.NUMBER_AFTER_DOT:
                match char:
                    case char if char in DIGITS:
                        next_state = JsonState.IN_FRACTION
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.IN_FRACTION:
                match char:
                    case char if char in DIGITS:
                        next_state = JsonState.IN_FRACTION
                    case "e" | "E":
                        next_state = JsonState.NUMBER_AFTER_EXP
                    case char if char in WHITESPACE:
                        next_state = JsonState.EXPECT_COMMA_OR_END
                    case ",":
                        next_state = JsonState.EXPECT_KEY
                    case "}":
                        depth -= 1
                        if depth == 0:
                            next_state = JsonState.END
                        else:
                            next_state = JsonState.EXPECT_COMMA_OR_END
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.NUMBER_AFTER_EXP:
                match char:
                    case "+" | "-":
                        next_state = JsonState.NUMBER_AFTER_EXP_SIGN
                    case char if char in DIGITS:
                        next_state = JsonState.IN_EXPONENT
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.NUMBER_AFTER_EXP_SIGN:
                match char:
                    case char if char in DIGITS:
                        next_state = JsonState.IN_EXPONENT
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.IN_EXPONENT:
                match char:
                    case char if char in DIGITS:
                        next_state = JsonState.IN_EXPONENT
                    case char if char in WHITESPACE:
                        next_state = JsonState.EXPECT_COMMA_OR_END
                    case ",":
                        next_state = JsonState.EXPECT_KEY
                    case "}":
                        depth -= 1
                        if depth == 0:
                            next_state = JsonState.END
                        else:
                            next_state = JsonState.EXPECT_COMMA_OR_END
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.IN_TRUE_T:
                match char:
                    case "r":
                        next_state = JsonState.IN_TRUE_TR
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.IN_TRUE_TR:
                match char:
                    case "u":
                        next_state = JsonState.IN_TRUE_TRU
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.IN_TRUE_TRU:
                match char:
                    case "e":
                        next_state = JsonState.EXPECT_COMMA_OR_END
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.IN_FALSE_F:
                match char:
                    case "a":
                        next_state = JsonState.IN_FALSE_FA
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.IN_FALSE_FA:
                match char:
                    case "l":
                        next_state = JsonState.IN_FALSE_FAL
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.IN_FALSE_FAL:
                match char:
                    case "s":
                        next_state = JsonState.IN_FALSE_FALS
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.IN_FALSE_FALS:
                match char:
                    case "e":
                        next_state = JsonState.EXPECT_COMMA_OR_END
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.END:
                if char in WHITESPACE:
                    next_state = JsonState.END
                else:
                    next_state = JsonState.INVALID

        return next_state, depth
