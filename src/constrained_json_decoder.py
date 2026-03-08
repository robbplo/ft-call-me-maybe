from typing import Protocol

from src.state import State, JsonState
from src.vocabulary import Vocabulary


WHITESPACE = set(" \t\n\r")
DIGITS = set("0123456789")
MAX_DEPTH = 2


class Tokenizer(Protocol):
    """Structural protocol satisfied by any object that can decode token IDs."""

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into a string.

        Args:
            ids: List of integer token IDs.

        Returns:
            The decoded text string.
        """
        ...


class ConstrainedJSONDecoder:
    """Token-by-token JSON decoder with structural and schema-level constraints.

    At each generation step the decoder pre-computes which tokens keep the
    output both syntactically valid (correct JSON structure) and semantically
    valid (only allowed argument keys appear at depth 2).  Invalid tokens
    receive a ``-inf`` logit offset so greedy selection never picks them.

    Attributes:
        model: Tokenizer used to decode individual token IDs.
        vocab: Vocabulary providing the full token list and its size.
        token_bytes: Pre-decoded string representation of every token ID.
        structural_masks: Cache mapping ``(JsonState, depth)`` pairs to their
            pre-computed structural logit mask and resulting depth.
    """

    def __init__(self, tokenizer: Tokenizer, vocab: Vocabulary) -> None:
        """Initialize the decoder and pre-compute all structural masks.

        Args:
            tokenizer: Tokenizer used to decode token IDs to strings.
            vocab: Vocabulary describing the full token set.
        """
        self.model: Tokenizer = tokenizer
        self.vocab: Vocabulary = vocab
        self.token_bytes: list[str] = [tokenizer.decode([i]) for i in range(vocab.size)]
        self.structural_masks: dict[tuple[JsonState, int], tuple[list[float], int]] = {
            (state, depth): self._get_structural_mask(state, depth)
            for state in JsonState
            for depth in range(MAX_DEPTH + 1)}

    def get_logit_mask(self, state: State) -> list[float]:
        """Return the combined structural and schema logit mask for *state*.

        Args:
            state: Current decoding state holding JSON position and schema info.

        Returns:
            A list of float offsets of length ``vocab.size``.  ``0.0`` means
            the token is allowed; ``-inf`` means it is forbidden.
        """
        structural_mask, depth = self.structural_masks[state.s, state.depth]
        state.depth = depth
        schema_mask = self._get_schema_mask(state)
        mask = [a + b for a, b in zip(structural_mask, schema_mask)]
        return mask

    def _get_schema_mask(self, state: State) -> list[float]:
        """Compute the schema-level logit mask enforcing allowed argument keys.

        Only applied at depth 2; at all other depths every token is allowed.

        Args:
            state: Current decoding state.

        Returns:
            A list of float offsets of length ``vocab.size``.
        """
        if state.depth != 2:
            return [0.0] * self.vocab.size
        mask: list[float] = [-float('inf')] * self.vocab.size
        for token_id, token_str in enumerate(self.token_bytes):
            allowed, _, _ = self._simulate_schema(state, token_str)
            if allowed:
                mask[token_id] = 0.0
        return mask

    def _simulate_schema(
        self, state: State, token_str: str, prints: bool = False
    ) -> tuple[bool, list[str], str]:
        """Simulate appending *token_str* and check schema validity.

        Does not mutate *state*; all changes are applied to local copies and
        the original state is restored before returning.

        Args:
            state: Current decoding state (read-only during simulation).
            token_str: Token string to simulate appending.
            prints: Unused debug flag kept for call-site compatibility.

        Returns:
            A three-tuple ``(allowed, keys, current_key)`` where *allowed*
            is ``True`` if the token is schema-valid, *keys* is the updated
            list of completed argument keys, and *current_key* is the updated
            in-progress key string.
        """
        orig_s, orig_depth = state.s, state.depth
        orig_keys, orig_current_key = state.keys, state.current_key

        allowed = True
        keys = state.keys.copy()
        current_key = state.current_key
        s = state.s
        depth = state.depth

        for char in token_str:
            if depth == 2:
                match s:
                    case JsonState.IN_KEY:
                        state.s, state.depth = s, depth
                        state.keys, state.current_key = keys, current_key
                        allowed, keys, current_key = state.add_key_char(char)
                        if not allowed:
                            break
                    case JsonState.EXPECT_COMMA_OR_END | JsonState.IN_NUMBER if char == '}':
                        remaining_keys = [k for k in state.allowed_keys if k not in keys]
                        if any(k not in keys for k in remaining_keys):
                            allowed = False
                            break

            s, depth = self._simulate_structure_char(s, char, depth)
            if s == JsonState.INVALID:
                allowed = False
                break

        state.s, state.depth = orig_s, orig_depth
        state.keys, state.current_key = orig_keys, orig_current_key
        return allowed, keys, current_key

    def _get_structural_mask(
        self, json_state: JsonState, depth: int
    ) -> tuple[list[float], int]:
        """Pre-compute the structural logit mask for a given state and depth.

        Args:
            json_state: The JSON parser state to compute the mask for.
            depth: The current nesting depth.

        Returns:
            A two-tuple of the mask list and the resulting depth after applying
            the last simulated token.
        """
        mask: list[float] = [-float('inf')] * self.vocab.size
        next_depth: int = depth
        for token_id, token_str in enumerate(self.token_bytes):
            next_state, next_depth = self._simulate_structure(json_state, token_str, depth)
            if next_state is not JsonState.INVALID:
                mask[token_id] = 0.0
        return mask, next_depth

    def _simulate_structure(
        self, json_state: JsonState, token: str, depth: int
    ) -> tuple[JsonState, int]:
        """Simulate appending *token* character by character through the FSM.

        Args:
            json_state: Starting JSON parser state.
            token: Token string whose characters will be fed into the FSM.
            depth: Starting nesting depth.

        Returns:
            The resulting ``(JsonState, depth)`` after processing all characters.
        """
        for char in token:
            json_state, depth = self._simulate_structure_char(json_state, char, depth)
        return json_state, depth

    def _simulate_structure_char(
        self, json_state: JsonState, char: str, depth: int
    ) -> tuple[JsonState, int]:
        """Advance the JSON structural FSM by one character.

        Args:
            json_state: Current parser state.
            char: Single character to process.
            depth: Current nesting depth.

        Returns:
            The resulting ``(JsonState, depth)`` after processing *char*.
        """
        next_state = json_state
        match json_state:
            case JsonState.INVALID:
                pass
            case JsonState.START:
                match char:
                    case '{':
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
            case JsonState.EXPECT_COLON:
                match char:
                    case ':':
                        next_state = JsonState.EXPECT_VALUE
                    case char if char in WHITESPACE:
                        next_state = JsonState.EXPECT_COLON
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.EXPECT_VALUE:
                match char:
                    case '{' if depth < MAX_DEPTH:
                        next_state = JsonState.EXPECT_KEY
                        depth += 1
                    case '"':
                        next_state = JsonState.IN_STRING
                    case char if char in WHITESPACE:
                        next_state = JsonState.EXPECT_VALUE
                    case char if char in DIGITS:
                        next_state = JsonState.IN_NUMBER
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.EXPECT_COMMA_OR_END:
                match char:
                    case ',':
                        next_state = JsonState.EXPECT_KEY
                    case '}':
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
                    case char if char in WHITESPACE:
                        next_state = JsonState.EXPECT_COMMA_OR_END
                    case ',':
                        next_state = JsonState.EXPECT_KEY
                    case '}':
                        depth -= 1
                        if depth == 0:
                            next_state = JsonState.END
                        else:
                            next_state = JsonState.EXPECT_COMMA_OR_END
                    case _:
                        next_state = JsonState.INVALID
            case JsonState.END:
                if char in WHITESPACE:
                    next_state = JsonState.END
                else:
                    next_state = JsonState.INVALID

        return next_state, depth
