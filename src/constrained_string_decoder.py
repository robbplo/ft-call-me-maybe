from typing import Protocol, cast

from src.vocabulary import Vocabulary


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


class ConstrainedStringDecoder:
    """Greedy decoder that constrains generation to a fixed set of allowed strings.

    At each step only tokens whose concatenation with the current prefix
    remains a valid prefix of at least one allowed string are assigned a
    zero logit mask; all others receive ``-inf``.

    Attributes:
        model: Tokenizer used to decode individual token IDs.
        vocab: Vocabulary providing the full token list and its size.
        allowed_strings: Set of strings the decoder may produce.
        prefixes: All valid prefixes of every allowed string (including empty
            string and the full strings themselves).
        token_bytes: Pre-decoded string representation of every token ID.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        vocab: Vocabulary,
        allowed_strings: list[str],
    ) -> None:
        """Initialize the decoder with a tokenizer, vocabulary, and allowed strings.

        Args:
            tokenizer: Tokenizer used to decode token IDs to strings.
            vocab: Vocabulary describing the full token set.
            allowed_strings: Non-empty list of strings the decoder may produce.

        Raises:
            ValueError: If *allowed_strings* is empty.
        """
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
        """Compute the additive logit mask for the current generation prefix.

        Tokens that would extend *value* into a valid prefix of an allowed
        string receive ``0.0``; all others receive ``-inf``.

        Args:
            value: The text generated so far.

        Returns:
            A list of float offsets of length ``vocab.size`` to add to raw
            logits before greedy selection.
        """
        mask = [-float("inf")] * self.vocab.size
        for token_id, token_str in enumerate(self.token_bytes):
            if value + token_str in self.prefixes:
                mask[token_id] = 0.0
        return cast(list[float], mask)
