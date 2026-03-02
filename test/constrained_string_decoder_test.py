import pytest
import torch

from src.constrained_string_decoder import (
    ConstrainedStringDecoder,
    StringState,
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


@pytest.mark.parametrize(
    ("initial_value", "initial_state", "token", "expected_state", "expected_value"),
    [
        ("", StringState.START, "c", StringState.IN_STRING, "c"),
        ("", StringState.START, "ca", StringState.IN_STRING, "ca"),
        ("", StringState.START, "cat", StringState.COMPLETE, "cat"),
        ("", StringState.START, "dog", StringState.COMPLETE, "dog"),
        ("", StringState.START, "z", StringState.INVALID, ""),
        ("ca", StringState.IN_STRING, "r", StringState.COMPLETE, "car"),
        ("cat", StringState.COMPLETE, "x", StringState.INVALID, "cat"),
    ],
)
def test_simulate_structure(
    initial_value,
    initial_state,
    token,
    expected_state,
    expected_value,
):
    decoder = _make_decoder(
        {"c": 0, "a": 1, "t": 2, "r": 3, "d": 4, "o": 5, "g": 6, "x": 7, "z": 8},
        ["cat", "car", "dog"],
    )
    value, state = decoder._simulate_structure(initial_value, initial_state, token)
    assert state is expected_state
    assert value == expected_value


def test_get_logit_mask_by_prefix():
    token_map = {"c": 0, "d": 1, "x": 2, "a": 3, "o": 4, "t": 5, "r": 6, "g": 7}
    decoder = _make_decoder(token_map, ["cat", "car", "dog"])

    start_mask = decoder.get_logit_mask("", StringState.START)
    start_valid = {token for token, idx in token_map.items() if start_mask[idx] == 0.0}
    assert start_valid == {"c", "d"}

    ca_value, ca_state = decoder._simulate_structure("", StringState.START, "ca")
    ca_mask = decoder.get_logit_mask(ca_value, ca_state)
    ca_valid = {token for token, idx in token_map.items() if ca_mask[idx] == 0.0}
    assert ca_valid == {"t", "r"}

    cat_value, cat_state = decoder._simulate_structure("", StringState.START, "cat")
    cat_mask = decoder.get_logit_mask(cat_value, cat_state)
    cat_valid = {token for token, idx in token_map.items() if cat_mask[idx] == 0.0}
    assert cat_valid == set()


def test_complete_value_can_still_continue_when_prefix_overlaps():
    token_map = {"a": 0, "b": 1, "x": 2}
    decoder = _make_decoder(token_map, ["a", "ab"])

    value, state = decoder._simulate_structure("", StringState.START, "a")
    assert state is StringState.COMPLETE
    assert value == "a"

    mask = decoder.get_logit_mask(value, state)
    valid = {token for token, idx in token_map.items() if mask[idx] == 0.0}
    assert valid == {"b"}

    next_value, next_state = decoder._simulate_structure(value, state, "b")
    assert next_state is StringState.COMPLETE
    assert next_value == "ab"


def test_empty_allowed_strings_raises():
    with pytest.raises(ValueError):
        _make_decoder({"a": 0}, [])
