from enum import Enum, auto
from typing import Protocol, cast

import torch

from src.vocabulary import Vocabulary


class Tokenizer(Protocol):
    def decode(self, ids: torch.Tensor | list[int]) -> str:
        ...


class StringState(Enum):
    START = 0
    IN_STRING = auto()
    COMPLETE = auto()
    INVALID = auto()


class ConstrainedStringDecoder:
    def __init__(
        self,
        tokenizer: Tokenizer,
        vocab: Vocabulary,
        allowed_strings: list[str],
    ):
        if not allowed_strings:
            raise ValueError("allowed_strings must contain at least one value.")

        self.model: Tokenizer = tokenizer
        self.vocab: Vocabulary = vocab
        self.allowed_strings: set[str] = set(allowed_strings)
        self.prefixes: set[str] = {
            allowed[:index]
            for allowed in self.allowed_strings
            for index in range(len(allowed) + 1)
        }
        self.token_bytes: list[str] = [tokenizer.decode([i]) for i in range(vocab.size)]

    def get_logit_mask(self, value: str, string_state: StringState) -> list[float]:
        mask = [-float("inf")] * self.vocab.size
        for token_id, token_str in enumerate(self.token_bytes):
            _, next_state = self._simulate_structure(value, string_state, token_str)
            if next_state is not StringState.INVALID:
                mask[token_id] = 0.0
        return cast(list[float], mask)

    def _simulate_structure(
        self,
        value: str,
        string_state: StringState,
        token: str,
    ) -> tuple[str, StringState]:
        next_value = value
        next_state = string_state
        if next_state is StringState.INVALID:
            return next_value, next_state

        for char in token:
            next_value, next_state = self._simulate_structure_char(
                next_value,
                next_state,
                char,
            )
            if next_state is StringState.INVALID:
                return next_value, next_state

        if next_value in self.allowed_strings:
            next_state = StringState.COMPLETE
        elif next_value == "":
            next_state = StringState.START
        else:
            next_state = StringState.IN_STRING
        return next_value, next_state

    def _simulate_structure_char(
        self,
        value: str,
        string_state: StringState,
        char: str,
    ) -> tuple[str, StringState]:
        if string_state is StringState.INVALID:
            return value, string_state

        next_value = value + char
        if next_value not in self.prefixes:
            return value, StringState.INVALID

        if next_value in self.allowed_strings:
            string_state = StringState.COMPLETE
        else:
            string_state = StringState.IN_STRING
        return next_value, string_state
