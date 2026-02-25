from src.state import State, JsonState
import copy
from typing import Protocol

import torch
from src.vocabulary import Vocabulary


WHITESPACE = set(" \t\n\r")
DIGITS = set("0123456789")

class Tokenizer(Protocol):
    def decode(self, ids: torch.Tensor | list[int]) -> str:
        ...

class ConstrainedDecoder:
    def __init__(self, tokenizer: Tokenizer, vocab: Vocabulary):
        self.model: Tokenizer = tokenizer
        self.vocab: Vocabulary = vocab
        self.token_bytes: list[str] = [tokenizer.decode([i]) for i in range(vocab.size)]

    def get_logit_mask(self, state: State) -> list[float]:
        mask = [-float('inf')] * self.vocab.size
        for token_id, token_str in enumerate(self.token_bytes):
            next_state = self.simulate(copy.deepcopy(state), token_str)
            if next_state.s is not JsonState.INVALID:
                mask[token_id] = 0.0
        return mask

    def simulate(self, state: State, token: str) -> State:
        for char in token:
            state = self._simulate_char(state, char)
            if state.s is JsonState.INVALID:
                return state
        return state

    def _simulate_char(self, state: State, char: str) -> State:
        match state.s:
            case JsonState.START:
                match char:
                    case '{': state.s = JsonState.EXPECT_KEY
                    case char if char in WHITESPACE: state.s = JsonState.START
                    case _: state.s = JsonState.INVALID
            case JsonState.EXPECT_KEY:
                match char:
                    case '"': state.s = JsonState.IN_KEY
                    case char if char in WHITESPACE: state.s = JsonState.EXPECT_KEY
                    case _: state.s = JsonState.INVALID
            case JsonState.IN_KEY:
                key_index = len(state.current_key)
                if char in [key[key_index] for key in state.allowed_keys]:
                    state.current_key += char
                    
                match char:
                    case '"': state.s = JsonState.EXPECT_COLON
                    case _: state.s = JsonState.IN_KEY
            case JsonState.IN_STRING:
                match char:
                    case '"': state.s = JsonState.EXPECT_COMMA_OR_END
                    case _: state.s = JsonState.IN_STRING
            case JsonState.EXPECT_COLON:
                match char:
                    case ':': state.s = JsonState.EXPECT_VALUE
                    case char if char in WHITESPACE: state.s = JsonState.EXPECT_COLON
                    case _: state.s = JsonState.INVALID
            case JsonState.EXPECT_VALUE:
                match char:
                    case '"': state.s = JsonState.IN_STRING
                    case char if char in WHITESPACE: state.s = JsonState.EXPECT_VALUE
                    case char if char in DIGITS: state.s = JsonState.IN_NUMBER
                    case _: state.s = JsonState.INVALID
            case JsonState.EXPECT_COMMA_OR_END:
                match char:
                    case ',': state.s = JsonState.EXPECT_KEY
                    case '}': state.s = JsonState.END
                    case char if char in WHITESPACE: state.s = JsonState.EXPECT_COMMA_OR_END
                    case _: state.s = JsonState.INVALID
            case JsonState.IN_NUMBER:
                match char:
                    case char if char in DIGITS: state.s = JsonState.IN_NUMBER
                    case char if char in WHITESPACE: state.s = JsonState.EXPECT_COMMA_OR_END
                    case ',': state.s = JsonState.EXPECT_KEY
                    case '}': state.s = JsonState.END
                    case _: state.s = JsonState.INVALID
            case JsonState.END :
                state.s = JsonState.INVALID
            case JsonState.INVALID:
                pass

        return state



