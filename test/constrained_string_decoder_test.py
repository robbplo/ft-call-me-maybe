import pytest
import torch

from src.constrained_string_decoder import (
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
) -> ConstrainedStringDecoder:
    model = _FakeModel(token_map)
    vocab = Vocabulary(token_map)
    return ConstrainedStringDecoder(model, vocab)


def _prefixes(allowed_strings: list[str]) -> set[str]:
    allowed_set = set(allowed_strings)
    return {
        allowed[:index]
        for allowed in allowed_set
        for index in range(len(allowed) + 1)
    }


@pytest.mark.parametrize(
    ("initial_value", "token", "expected_valid"),
    [
        ("", "c", True),
        ("", "ca", True),
        ("", "cat", True),
        ("", "dog", True),
        ("", "z", False),
        ("ca", "r", True),
        ("cat", "x", False),
    ],
)
def test_simulate_structure(
    initial_value,
    token,
    expected_valid,
):
    decoder = _make_decoder({"c": 0, "a": 1, "t": 2, "r": 3, "d": 4, "o": 5, "g": 6, "x": 7, "z": 8})
    prefixes = _prefixes(["cat", "car", "dog"])
    assert decoder._simulate_structure(initial_value, token, prefixes) is expected_valid


def test_get_logit_mask_by_prefix():
    token_map = {"c": 0, "d": 1, "x": 2, "a": 3, "o": 4, "t": 5, "r": 6, "g": 7}
    decoder = _make_decoder(token_map)
    allowed_strings = ["cat", "car", "dog"]

    start_mask = decoder.get_logit_mask("", allowed_strings)
    start_valid = {token for token, idx in token_map.items() if start_mask[idx] == 0.0}
    assert start_valid == {"c", "d"}

    ca_mask = decoder.get_logit_mask("ca", allowed_strings)
    ca_valid = {token for token, idx in token_map.items() if ca_mask[idx] == 0.0}
    assert ca_valid == {"t", "r"}

    cat_mask = decoder.get_logit_mask("cat", allowed_strings)
    cat_valid = {token for token, idx in token_map.items() if cat_mask[idx] == 0.0}
    assert cat_valid == set()


def test_complete_value_can_still_continue_when_prefix_overlaps():
    token_map = {"a": 0, "b": 1, "x": 2}
    decoder = _make_decoder(token_map)
    allowed_strings = ["a", "ab"]

    assert decoder._simulate_structure("", "a", _prefixes(allowed_strings)) is True

    mask = decoder.get_logit_mask("a", allowed_strings)
    valid = {token for token, idx in token_map.items() if mask[idx] == 0.0}
    assert valid == {"b"}

    assert decoder._simulate_structure("a", "b", _prefixes(allowed_strings)) is True


def test_empty_allowed_strings_raises():
    with pytest.raises(ValueError):
        _make_decoder({"a": 0}).get_logit_mask("", [])
