from .common import (
    MASK_BLOCKED,
    Tokenizer,
    build_token_mask,
    decode_vocab_tokens,
)
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
LITERAL_TRANSITIONS = {
    JsonState.IN_TRUE_T: ("r", JsonState.IN_TRUE_TR),
    JsonState.IN_TRUE_TR: ("u", JsonState.IN_TRUE_TRU),
    JsonState.IN_TRUE_TRU: ("e", JsonState.EXPECT_COMMA_OR_END),
    JsonState.IN_FALSE_F: ("a", JsonState.IN_FALSE_FA),
    JsonState.IN_FALSE_FA: ("l", JsonState.IN_FALSE_FAL),
    JsonState.IN_FALSE_FAL: ("s", JsonState.IN_FALSE_FALS),
    JsonState.IN_FALSE_FALS: ("e", JsonState.EXPECT_COMMA_OR_END),
}


class ConstrainedJSONDecoder:
    """Token-by-token JSON decoder with structural and schema constraints."""

    def __init__(self, tokenizer: Tokenizer, vocab: Vocabulary) -> None:
        """Initialize the decoder and pre-compute structural masks."""
        self.vocab: Vocabulary = vocab
        self.token_bytes: list[str] = decode_vocab_tokens(tokenizer, vocab)
        self.structural_masks: dict[tuple[JsonState, int], list[float]] = {
            (state, depth): self._get_structural_mask(state, depth)
            for state in JsonState
            for depth in range(MAX_DEPTH + 1)
        }

    def get_logit_mask(self, state: State) -> list[float]:
        """Return the structural logit mask for the current state."""
        return self.structural_masks[state.s, state.depth]

    def find_valid_token(
        self, logits: list[float], state: State
    ) -> tuple[int, State]:
        """Return the highest-logit structurally- and schema-valid token.

        Applies the structural mask, then walks tokens in descending logit
        order and returns the first one that also passes the schema check.
        For greedy decoding this is equivalent to the old full-mask approach
        but avoids calling advance_state for every vocabulary token.
        """
        structural_mask = self.structural_masks[state.s, state.depth]
        masked = [a + b for a, b in zip(logits, structural_mask)]
        ordered = sorted(
            range(len(masked)),
            key=lambda i: masked[i], reverse=True
        )
        for idx in ordered:
            if masked[idx] == MASK_BLOCKED:
                break
            valid, next_state = self.advance_state(
                state, self.token_bytes[idx]
            )
            if valid:
                return idx, next_state
        raise ValueError("No valid token found — state machine is stuck.")

    def advance_state(self, state: State, text: str) -> tuple[bool, State]:
        """Return whether *text* is valid and the resulting copied state."""
        next_state = state.copy()

        for char in text:
            previous_depth = next_state.depth
            if next_state.depth == 2 and not self._advance_schema_char(
                next_state, char
            ):
                return False, next_state

            next_state.s, next_state.depth = self._simulate_structure_char(
                next_state.s,
                char,
                next_state.depth,
            )
            if next_state.s == JsonState.INVALID:
                return False, next_state
            if (
                previous_depth == 2
                and char == "}"
                and next_state.depth == 1
                and next_state.remaining_keys()
            ):
                return False, next_state

        return True, next_state

    def _advance_schema_char(self, state: State, char: str) -> bool:
        """Apply depth-2 schema tracking for one character."""
        if state.s == JsonState.IN_KEY:
            allowed, state.keys, state.current_key, state.current_value_key = (
                state.add_key_char(char)
            )
            return allowed

        return self._consume_value_char(state, char)

    def _consume_value_char(
        self, state: State, char: str
    ) -> bool:
        """Validate a character against the active depth-2 value type."""
        if state.current_value_key is None:
            return True

        value_type = state.allowed_types.get(state.current_value_key)
        if value_type is None:
            return False

        if state.s == JsonState.EXPECT_COLON:
            return True

        if state.s == JsonState.EXPECT_VALUE:
            if char in WHITESPACE:
                return True
            return self._start_typed_value(state, value_type, char)

        if state.s == JsonState.IN_STRING:
            if value_type != "str":
                return False
            state.current_value_buffer += char
            if (
                len(state.current_value_buffer) > 1
                and state.current_value_buffer[-1] == '"'
            ):
                state.current_value_key = None
                state.current_value_buffer = ""
            return True

        if state.s in BOOLEAN_STATES:
            if value_type != "bool":
                return False
            state.current_value_buffer += char
            if not any(
                literal.startswith(state.current_value_buffer)
                for literal in ("true", "false")
            ):
                return False
            if state.current_value_buffer in {"true", "false"}:
                state.current_value_key = None
                state.current_value_buffer = ""
            return True

        if state.s in NUMBER_STATES:
            if value_type not in {"int", "float"}:
                return False
            return self._consume_numeric_char(
                state,
                value_type,
                char,
            )

        return True

    def _start_typed_value(
        self, state: State, value_type: str, char: str
    ) -> bool:
        """Validate the first non-whitespace character of a typed value."""
        match value_type:
            case "str":
                if char != '"':
                    return False
            case "bool":
                if char not in {"t", "f"}:
                    return False
            case "int" | "float":
                if char not in DIGITS | {"-"}:
                    return False
            case _:
                return False
        state.current_value_buffer = char
        return True

    def _consume_numeric_char(
        self,
        state: State,
        value_type: str,
        char: str,
    ) -> bool:
        """Validate a numeric character or numeric delimiter."""
        if char in WHITESPACE or char in {",", "}"}:
            if state.s not in COMPLETE_NUMBER_STATES:
                return False
            if value_type == "int" and not self._is_integer_number(
                state.current_value_buffer
            ):
                return False
            state.current_value_key = None
            state.current_value_buffer = ""
            return True

        if value_type == "int" and char in {".", "e", "E"}:
            return False

        state.current_value_buffer += char
        return True

    def _is_integer_number(self, value: str) -> bool:
        """Return whether the current number prefix is integer-only."""
        return (
            bool(value)
            and value != "-"
            and "." not in value
            and "e" not in value
            and "E" not in value
        )

    def _get_structural_mask(
        self, json_state: JsonState, depth: int
    ) -> list[float]:
        """Compute the structural logit mask for a given state and depth."""
        return build_token_mask(
            self.token_bytes,
            lambda token_str: self._simulate_structure(
                json_state,
                token_str,
                depth,
            )[0]
            is not JsonState.INVALID,
        )

    def _simulate_structure(
        self, json_state: JsonState, token: str, depth: int
    ) -> tuple[JsonState, int]:
        """Simulate appending character by character through the FSM."""
        for char in token:
            json_state, depth = self._simulate_structure_char(
                json_state, char, depth)
        return json_state, depth

    def _simulate_structure_char(
        self, json_state: JsonState, char: str, depth: int
    ) -> tuple[JsonState, int]:
        """Advance the JSON structural FSM by one character."""
        match json_state:
            case JsonState.INVALID:
                return JsonState.INVALID, depth
            case JsonState.START:
                match char:
                    case "{":
                        return JsonState.EXPECT_KEY, depth + 1
                    case _:
                        return JsonState.INVALID, depth
            case JsonState.EXPECT_KEY:
                match char:
                    case '"':
                        return JsonState.IN_KEY, depth
                    case char if char in WHITESPACE:
                        return JsonState.EXPECT_KEY, depth
                    case _:
                        return JsonState.INVALID, depth
            case JsonState.IN_KEY:
                match char:
                    case '"':
                        return JsonState.EXPECT_COLON, depth
                    case _:
                        return JsonState.IN_KEY, depth
            case JsonState.IN_STRING:
                match char:
                    case '"':
                        return JsonState.EXPECT_COMMA_OR_END, depth
                    case _:
                        return JsonState.IN_STRING, depth
            case JsonState.NUMBER_AFTER_MINUS:
                match char:
                    case "0":
                        return JsonState.IN_NUMBER_ZERO, depth
                    case char if char in NON_ZERO_DIGITS:
                        return JsonState.IN_NUMBER, depth
                    case _:
                        return JsonState.INVALID, depth
            case JsonState.IN_NUMBER_ZERO:
                return self._advance_number_state(
                    json_state,
                    char,
                    depth,
                    allow_digits=False,
                    allow_fraction=True,
                    allow_exponent=True,
                )
            case JsonState.EXPECT_COLON:
                match char:
                    case ":":
                        return JsonState.EXPECT_VALUE, depth
                    case char if char in WHITESPACE:
                        return JsonState.EXPECT_COLON, depth
                    case _:
                        return JsonState.INVALID, depth
            case JsonState.EXPECT_VALUE:
                match char:
                    case "{" if depth < MAX_DEPTH:
                        return JsonState.EXPECT_KEY, depth + 1
                    case '"':
                        return JsonState.IN_STRING, depth
                    case "-":
                        return JsonState.NUMBER_AFTER_MINUS, depth
                    case "0":
                        return JsonState.IN_NUMBER_ZERO, depth
                    case "t":
                        return JsonState.IN_TRUE_T, depth
                    case "f":
                        return JsonState.IN_FALSE_F, depth
                    case char if char in WHITESPACE:
                        return JsonState.EXPECT_VALUE, depth
                    case char if char in NON_ZERO_DIGITS:
                        return JsonState.IN_NUMBER, depth
                    case _:
                        return JsonState.INVALID, depth
            case JsonState.EXPECT_COMMA_OR_END:
                match char:
                    case ",":
                        return JsonState.EXPECT_KEY, depth
                    case "}":
                        return self._close_object(depth)
                    case char if char in WHITESPACE:
                        return JsonState.EXPECT_COMMA_OR_END, depth
                    case _:
                        return JsonState.INVALID, depth
            case JsonState.IN_NUMBER:
                return self._advance_number_state(
                    json_state,
                    char,
                    depth,
                    allow_digits=True,
                    allow_fraction=True,
                    allow_exponent=True,
                )
            case JsonState.NUMBER_AFTER_DOT:
                match char:
                    case char if char in DIGITS:
                        return JsonState.IN_FRACTION, depth
                    case _:
                        return JsonState.INVALID, depth
            case JsonState.IN_FRACTION:
                return self._advance_number_state(
                    json_state,
                    char,
                    depth,
                    allow_digits=True,
                    allow_fraction=False,
                    allow_exponent=True,
                )
            case JsonState.NUMBER_AFTER_EXP:
                match char:
                    case "+" | "-":
                        return JsonState.NUMBER_AFTER_EXP_SIGN, depth
                    case char if char in DIGITS:
                        return JsonState.IN_EXPONENT, depth
                    case _:
                        return JsonState.INVALID, depth
            case JsonState.NUMBER_AFTER_EXP_SIGN:
                match char:
                    case char if char in DIGITS:
                        return JsonState.IN_EXPONENT, depth
                    case _:
                        return JsonState.INVALID, depth
            case JsonState.IN_EXPONENT:
                return self._advance_number_state(
                    json_state,
                    char,
                    depth,
                    allow_digits=True,
                    allow_fraction=False,
                    allow_exponent=False,
                )
            case state if state in LITERAL_TRANSITIONS:
                return self._advance_literal_state(state, char, depth)
            case JsonState.END:
                if char in WHITESPACE:
                    return JsonState.END, depth
                return JsonState.INVALID, depth

        return JsonState.INVALID, depth

    def _advance_number_state(
        self,
        json_state: JsonState,
        char: str,
        depth: int,
        *,
        allow_digits: bool,
        allow_fraction: bool,
        allow_exponent: bool,
    ) -> tuple[JsonState, int]:
        """Advance a numeric state with shared delimiter handling."""
        if allow_digits and char in DIGITS:
            return json_state, depth
        if allow_fraction and char == ".":
            return JsonState.NUMBER_AFTER_DOT, depth
        if allow_exponent and char in {"e", "E"}:
            return JsonState.NUMBER_AFTER_EXP, depth
        return self._advance_completed_value(char, depth)

    def _advance_completed_value(
        self, char: str, depth: int
    ) -> tuple[JsonState, int]:
        """Advance after a complete literal or numeric token."""
        if char in WHITESPACE:
            return JsonState.EXPECT_COMMA_OR_END, depth
        if char == ",":
            return JsonState.EXPECT_KEY, depth
        if char == "}":
            return self._close_object(depth)
        return JsonState.INVALID, depth

    def _advance_literal_state(
        self,
        json_state: JsonState,
        char: str,
        depth: int,
    ) -> tuple[JsonState, int]:
        """Advance one step through the true/false literal FSM."""
        expected_char, next_state = LITERAL_TRANSITIONS[json_state]
        if char != expected_char:
            return JsonState.INVALID, depth
        return next_state, depth

    def _close_object(self, depth: int) -> tuple[JsonState, int]:
        """Close the current object and move to the parent context."""
        depth -= 1
        if depth == 0:
            return JsonState.END, depth
        return JsonState.EXPECT_COMMA_OR_END, depth
