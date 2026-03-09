from typing import cast

from .common import Tokenizer, build_token_mask, decode_vocab_tokens
from src.vocabulary import Vocabulary


class ConstrainedStringDecoder:
    """Greedy decoder constraining generation to a set of allowed strings.

    At each step only tokens whose concatenation with the current prefix
    remains a valid prefix of at least one allowed string are assigned a
    zero logit mask; all others receive ``-inf``.

    Attributes:
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
        """Initialize the decoder with a tokenizer, vocab, and allowed strings.

        Args:
            tokenizer: Tokenizer used to decode token IDs to strings.
            vocab: Vocabulary describing the full token set.
            allowed_strings: Non-empty list of strings the decoder may produce.

        Raises:
            ValueError: If *allowed_strings* is empty.
        """
        if not allowed_strings:
            raise ValueError(
                "allowed_strings must contain at least one value.")

        self.vocab: Vocabulary = vocab
        self.allowed_strings: set[str] = set(allowed_strings)
        self.prefixes: set[str] = {
            allowed[:index]
            for allowed in self.allowed_strings
            for index in range(len(allowed) + 1)
        }
        self.token_bytes: list[str] = decode_vocab_tokens(tokenizer, vocab)

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
        return cast(
            list[float],
            build_token_mask(
                self.token_bytes,
                lambda token_str: value + token_str in self.prefixes,
            ),
        )
