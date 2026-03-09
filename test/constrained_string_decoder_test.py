import pytest
import torch

from src.decoding import (
    ConstrainedStringDecoder,
)
from src.vocabulary import Vocabulary


class _FakeModel:
    def __init__(self, token_map: dict[str, int]) -> None:
        self._token_map: dict[int, str] = {v: k for k, v in token_map.items()}

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        if len(ids) != 1 or isinstance(ids, torch.Tensor):
            raise ValueError("Fake decoder expects a single token id.")
        return self._token_map[ids[0]]


def _make_decoder(
    token_map: dict[str, int],
    allowed_strings: list[str],
) -> ConstrainedStringDecoder:
    model = _FakeModel(token_map)
    vocab = Vocabulary(token_map)
    return ConstrainedStringDecoder(model, vocab, allowed_strings)


def test_get_logit_mask_by_prefix() -> None:
    allowed_strings = ["cat", "car", "dog"]
    token_map = {
        "c": 0,
        "d": 1,
        "x": 2,
        "a": 3,
        "o": 4,
        "t": 5,
        "r": 6,
        "g": 7,
        "ca": 8,
        "cat": 9,
        "dog": 10,
    }
    decoder = _make_decoder(token_map, allowed_strings)

    start_mask = decoder.get_logit_mask("")
    start_valid = {token for token,
                   idx in token_map.items() if start_mask[idx] == 0.0}
    assert start_valid == {"c", "d", "ca", "cat", "dog"}

    ca_mask = decoder.get_logit_mask("ca")
    ca_valid = {token for token, idx in token_map.items()
                if ca_mask[idx] == 0.0}
    assert ca_valid == {"t", "r"}

    cat_mask = decoder.get_logit_mask("cat")
    cat_valid = {token for token, idx in token_map.items()
                 if cat_mask[idx] == 0.0}
    assert cat_valid == set()


def test_complete_value_can_still_continue_when_prefix_overlaps() -> None:
    token_map = {"a": 0, "b": 1, "x": 2}
    allowed_strings = ["a", "ab"]
    decoder = _make_decoder(token_map, allowed_strings)

    start_mask = decoder.get_logit_mask("")
    start_valid = {token for token,
                   idx in token_map.items() if start_mask[idx] == 0.0}
    assert start_valid == {"a"}

    mask = decoder.get_logit_mask("a")
    valid = {token for token, idx in token_map.items() if mask[idx] == 0.0}
    assert valid == {"b"}

    complete_mask = decoder.get_logit_mask("ab")
    complete_valid = {token for token,
                      idx in token_map.items() if complete_mask[idx] == 0.0}
    assert complete_valid == set()


def test_empty_allowed_strings_raises() -> None:
    with pytest.raises(ValueError):
        _make_decoder({"a": 0}, [])
