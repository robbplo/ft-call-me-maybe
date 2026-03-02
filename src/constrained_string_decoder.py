from typing import Protocol, cast

import torch

from src.vocabulary import Vocabulary


class Tokenizer(Protocol):
    def decode(self, ids: torch.Tensor | list[int]) -> str:
        ...

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

    def get_logit_mask(self, value: str) -> list[float]:
        mask = [-float("inf")] * self.vocab.size
        for token_id, token_str in enumerate(self.token_bytes):
            if self._simulate_structure(value, token_str):
                mask[token_id] = 0.0
        return cast(list[float], mask)

    def _simulate_structure(
        self,
        value: str,
        token: str,
    ) -> bool:
        next_value = value

        for char in token:
            next_value += char
            if next_value not in self.prefixes:
                return False

        return True
