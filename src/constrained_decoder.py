from enum import Enum, auto
from typing import Protocol

import torch
from src.vocabulary import Vocabulary

class S(Enum):
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

    def get_logit_mask(self, state: S):
        mask = [-float('inf')] * self.vocab.size
        for token_id, token_str in enumerate(self.token_bytes):
            next_state = self.simulate(state, token_str)
            if next_state is not S.INVALID:
                mask[token_id] = 0.0
        return mask

    def simulate(self, state: S, token: str) -> S:
        for char in token:
            state = self._simulate_char(state, char)
            if state is S.INVALID:
                return state
        return state

    def _simulate_char(self, state: S, char: str) -> S:
        match state:
            case S.START:
                match char:
                    case '{': return S.EXPECT_KEY
                    case char if char in WHITESPACE: return S.START
                    case _: return S.INVALID
            case S.EXPECT_KEY:
                match char:
                    case '"': return S.IN_KEY
                    case char if char in WHITESPACE: return S.EXPECT_KEY
                    case _: return S.INVALID
            case S.IN_KEY:
                match char:
                    case '"': return S.EXPECT_COLON
                    case _: return S.IN_KEY
            case S.IN_STRING:
                match char:
                    case '"': return S.EXPECT_COMMA_OR_END
                    case _: return S.IN_STRING
            case S.EXPECT_COLON:
                match char:
                    case ':': return S.EXPECT_VALUE
                    case char if char in WHITESPACE: return S.EXPECT_COLON
                    case _: return S.INVALID
            case S.EXPECT_VALUE:
                match char:
                    case '"': return S.IN_STRING
                    case char if char in WHITESPACE: return S.EXPECT_VALUE
                    case char if char in DIGITS: return S.IN_NUMBER
                    case _: return S.INVALID
            case S.EXPECT_COMMA_OR_END:
                match char:
                    case ',': return S.EXPECT_KEY
                    case '}': return S.END
                    case char if char in WHITESPACE: return S.EXPECT_COMMA_OR_END
                    case _: return S.INVALID
            case S.IN_NUMBER:
                match char:
                    case char if char in DIGITS: return S.IN_NUMBER
                    case char if char in WHITESPACE: return S.EXPECT_COMMA_OR_END
                    case ',': return S.EXPECT_KEY
                    case _: return S.INVALID
            case S.END :
                return S.INVALID
            case S.INVALID:
                return S.INVALID



