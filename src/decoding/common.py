from typing import Callable, Protocol

from src.vocabulary import Vocabulary


MASK_ALLOWED = 0.0
MASK_BLOCKED = -float("inf")


class Tokenizer(Protocol):
    """Protocol for objects that can decode token IDs to strings."""

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs to a string."""
        ...


def decode_vocab_tokens(tokenizer: Tokenizer, vocab: Vocabulary) -> list[str]:
    """Decode each token in *vocab* once and cache its string form."""
    return [tokenizer.decode([index]) for index in range(vocab.size)]


def build_token_mask(
    token_bytes: list[str],
    is_allowed: Callable[[str], bool],
) -> list[float]:
    """Return a 0 / -inf additive mask based on *is_allowed*."""
    mask = [MASK_BLOCKED] * len(token_bytes)
    for token_id, token_str in enumerate(token_bytes):
        if is_allowed(token_str):
            mask[token_id] = MASK_ALLOWED
    return mask
