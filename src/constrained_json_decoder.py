from src.state import State, JsonState
from copy import deepcopy
from typing import Protocol, cast

import torch
from src.vocabulary import Vocabulary


WHITESPACE = set(" \t\n\r")
DIGITS = set("0123456789")
MAX_DEPTH = 2

class Tokenizer(Protocol):
    def decode(self, ids: torch.Tensor | list[int]) -> str:
        ...

class ConstrainedJSONDecoder:
    def __init__(self, tokenizer: Tokenizer, vocab: Vocabulary):
        self.model: Tokenizer = tokenizer
        self.vocab: Vocabulary = vocab
        self.token_bytes: list[str] = [tokenizer.decode([i]) for i in range(vocab.size)]
        self.structural_masks: dict[tuple[JsonState, int], tuple[list[float], int]] = {
            (state, depth): self._get_structural_mask(state, depth)
            for state in JsonState
            for depth in range(MAX_DEPTH + 1)}

    def get_logit_mask(self, state: State) -> list[float]:
        structural_mask, depth = self.structural_masks[state.s, state.depth]
        state.depth = depth
        schema_mask = self._get_schema_mask(state)
        mask = [a + b for a, b in zip(structural_mask, schema_mask)]
        return mask

    def _get_schema_mask(self, state: State) -> list[float]:
        if state.depth != 2:
            return [0.0] * self.vocab.size
        mask: list[float] = [-float('inf')] * self.vocab.size
        for token_id, token_str in enumerate(self.token_bytes):
            allowed, _, _ = self._simulate_schema(state, token_str)
            if allowed:
                mask[token_id] = 0.0
        return mask

    def _simulate_schema(self, state: State, token_str: str) -> tuple[bool, list[str], str]:
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
                    case JsonState.EXPECT_COMMA_OR_END | JsonState.IN_NUMBER if char == 'f':
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

    def _get_structural_mask(self, json_state: JsonState, depth: int) -> tuple[list[float], int]:
        mask: list[float] = [-float('inf')] * self.vocab.size
        for token_id, token_str in enumerate(self.token_bytes):
            next_state, next_depth = self._simulate_structure(json_state, token_str, depth)
            if next_state is not JsonState.INVALID:
                mask[token_id] = 0.0
        return mask, next_depth

    def _simulate_structure(self, json_state: JsonState, token: str, depth: int) -> tuple[JsonState, int]:
        for char in token:
            json_state, depth = self._simulate_structure_char(json_state, char, depth)
        return json_state, depth

    def _simulate_structure_char(self, json_state: JsonState, char: str, depth: int) -> tuple[JsonState, int]:
        next_state = json_state
        match json_state:
            case JsonState.INVALID:
                pass
            case JsonState.START:
                match char:
                    case '{': 
                        next_state = JsonState.EXPECT_KEY
                        depth += 1
                    case _: next_state = JsonState.INVALID
            case JsonState.EXPECT_KEY:
                match char:
                    case '"': next_state = JsonState.IN_KEY
                    case char if char in WHITESPACE: next_state = JsonState.EXPECT_KEY
                    case _: next_state = JsonState.INVALID
            case JsonState.IN_KEY:
                match char:
                    case '"': next_state = JsonState.EXPECT_COLON
                    case _: next_state = JsonState.IN_KEY
            case JsonState.IN_STRING:
                match char:
                    case '"': next_state = JsonState.EXPECT_COMMA_OR_END
                    case _: next_state = JsonState.IN_STRING
            case JsonState.EXPECT_COLON:
                match char:
                    case ':': next_state = JsonState.EXPECT_VALUE
                    case char if char in WHITESPACE: next_state = JsonState.EXPECT_COLON
                    case _: next_state = JsonState.INVALID
            case JsonState.EXPECT_VALUE:
                match char:
                    case '{' if depth < MAX_DEPTH: 
                        next_state = JsonState.EXPECT_KEY
                        depth += 1
                    case '"': next_state = JsonState.IN_STRING
                    case char if char in WHITESPACE: next_state = JsonState.EXPECT_VALUE
                    case char if char in DIGITS: next_state = JsonState.IN_NUMBER
                    case _: next_state = JsonState.INVALID
            case JsonState.EXPECT_COMMA_OR_END:
                match char:
                    case ',': next_state = JsonState.EXPECT_KEY
                    case '}': 
                        depth -= 1
                        if depth == 0:
                            next_state = JsonState.END
                        else:
                            next_state = JsonState.EXPECT_COMMA_OR_END
                    case char if char in WHITESPACE: next_state = JsonState.EXPECT_COMMA_OR_END
                    case _: next_state = JsonState.INVALID
            case JsonState.IN_NUMBER:
                match char:
                    case char if char in DIGITS: next_state = JsonState.IN_NUMBER
                    case char if char in WHITESPACE: next_state = JsonState.EXPECT_COMMA_OR_END
                    case ',': next_state = JsonState.EXPECT_KEY
                    case '}': 
                        depth -= 1
                        if depth == 0:
                            next_state = JsonState.END
                        else:
                            next_state = JsonState.EXPECT_COMMA_OR_END
                    case _: next_state = JsonState.INVALID
            case JsonState.END:
                if char in WHITESPACE:
                    next_state = JsonState.END
                else:
                    next_state = JsonState.INVALID

        return next_state, depth



