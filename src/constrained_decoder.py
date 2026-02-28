from src.state import State, JsonState
from copy import deepcopy
from typing import Protocol, cast

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
        self.structural_masks: list[list[float]] = [self._get_structural_mask(state) for state in JsonState]

    def get_logit_mask(self, state: State) -> list[float]:
        return self.structural_masks[state.s.value]

    def _get_structural_mask(self, json_state: JsonState) -> list[float]:
        mask = [-float('inf')] * self.vocab.size
        for token_id, token_str in enumerate(self.token_bytes):
            next_state = self._simulate_structure(json_state, token_str)
            if next_state is not JsonState.INVALID:
                mask[token_id] = 0.0
        return cast(list[float], mask)

    def _simulate_structure(self, json_state: JsonState, token: str) -> JsonState:
        for char in token:
            json_state = self._simulate_structure_char(json_state , char)
        return json_state

    def _simulate_structure_char(self, json_state: JsonState, char: str) -> JsonState:
        match json_state:
            case JsonState.INVALID:
                pass
            case JsonState.START:
                match char:
                    case '{': return JsonState.EXPECT_KEY
                    case _: return JsonState.INVALID
            case JsonState.EXPECT_KEY:
                match char:
                    case '"': return JsonState.IN_KEY
                    case char if char in WHITESPACE: return JsonState.EXPECT_KEY
                    case _: return JsonState.INVALID
            case JsonState.IN_KEY:
                match char:
                    case '"': return JsonState.EXPECT_COLON
                    case _: return JsonState.IN_KEY
            case JsonState.IN_STRING:
                match char:
                    case '"': return JsonState.EXPECT_COMMA_OR_END
                    case _: return JsonState.IN_STRING
            case JsonState.EXPECT_COLON:
                match char:
                    case ':': return JsonState.EXPECT_VALUE
                    case char if char in WHITESPACE: return JsonState.EXPECT_COLON
                    case _: return JsonState.INVALID
            case JsonState.EXPECT_VALUE:
                match char:
                    case '"': return JsonState.IN_STRING
                    case char if char in WHITESPACE: return JsonState.EXPECT_VALUE
                    case char if char in DIGITS: return JsonState.IN_NUMBER
                    case _: return JsonState.INVALID
            case JsonState.EXPECT_COMMA_OR_END:
                match char:
                    case ',': return JsonState.EXPECT_KEY
                    case '}': return JsonState.END
                    case char if char in WHITESPACE: return JsonState.EXPECT_COMMA_OR_END
                    case _: return JsonState.INVALID
            case JsonState.IN_NUMBER:
                match char:
                    case char if char in DIGITS: return JsonState.IN_NUMBER
                    case char if char in WHITESPACE: return JsonState.EXPECT_COMMA_OR_END
                    case ',': return JsonState.EXPECT_KEY
                    case '}': return JsonState.END
                    case _: return JsonState.INVALID
            case JsonState.END:
                return JsonState.INVALID

        return json_state



