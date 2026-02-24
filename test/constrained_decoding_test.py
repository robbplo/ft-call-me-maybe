from __future__ import annotations

from pathlib import Path
import pytest
import torch

from src.vocabulary import Vocabulary
from src.constrained_decoder import ConstrainedDecoder, S

class _FakeModel:
    def __init__(self, token_map: dict[str, int]) -> None:
        self._token_map: dict[int, str] = {v: k for k, v in token_map.items()}

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        if len(ids) != 1 or isinstance(ids, torch.Tensor):
            raise ValueError("Fake decoder expects a single token id.")
        return self._token_map[ids[0]]

token_map = {"a": 0, "b": 1, "c": 2}
model = _FakeModel(token_map)
vocab = Vocabulary(token_map)
decoder = ConstrainedDecoder(model, vocab)

@pytest.mark.parametrize(
    ("start_state", "input_text", "expected_state"),
    [
        # Start
        (S.START, "a", S.INVALID),
        (S.START, " b", S.INVALID),
        (S.START, "c ", S.INVALID),
        (S.START, "       ", S.START),
        (S.START, "}", S.INVALID),
        (S.START, "{", S.EXPECT_KEY),
        (S.START, " {", S.EXPECT_KEY),
        # Expect key
        (S.EXPECT_KEY, "a", S.INVALID),
        (S.EXPECT_KEY, " b", S.INVALID),
        (S.EXPECT_KEY, "c ", S.INVALID),
        (S.EXPECT_KEY, "\"", S.IN_KEY),
        (S.EXPECT_KEY, "       ", S.EXPECT_KEY),
        # In key
        (S.IN_KEY, "a", S.IN_KEY),
        (S.IN_KEY, " b", S.IN_KEY),
        (S.IN_KEY, "c ", S.IN_KEY),
        (S.IN_KEY, "\"", S.EXPECT_COLON),
        # Expect colon
        (S.EXPECT_COLON, "a", S.INVALID),
        (S.EXPECT_COLON, " ", S.EXPECT_COLON),
        (S.EXPECT_COLON, ":", S.EXPECT_VALUE),
        # Expect value
        (S.EXPECT_VALUE, "a", S.INVALID),
        (S.EXPECT_VALUE, " ", S.EXPECT_VALUE),
        (S.EXPECT_VALUE, "1", S.IN_NUMBER),
        (S.EXPECT_VALUE, "\"", S.IN_STRING),
        # In number
        (S.IN_NUMBER, "a", S.INVALID),
        (S.IN_NUMBER, "1", S.IN_NUMBER),
        (S.IN_NUMBER, " ", S.EXPECT_COMMA_OR_END),
        (S.IN_NUMBER, ",", S.EXPECT_KEY),
        # In string
        (S.IN_STRING, "a", S.IN_STRING),
        (S.IN_STRING, "1", S.IN_STRING),
        (S.IN_STRING, " ", S.IN_STRING),
        (S.IN_STRING, ",", S.IN_STRING),
        (S.IN_STRING, "\"", S.EXPECT_COMMA_OR_END),
        # Expect comma or end
        (S.EXPECT_COMMA_OR_END, "a", S.INVALID),
        (S.EXPECT_COMMA_OR_END, " ", S.EXPECT_COMMA_OR_END),
        (S.EXPECT_COMMA_OR_END, ",", S.EXPECT_KEY),
        (S.EXPECT_COMMA_OR_END, "}", S.END),
        # Multiple states in one token
        (S.START, "{a", S.INVALID),
        (S.START, "{\n\"", S.IN_KEY),

    ],
)
def test_simulate_json(start_state, input_text, expected_state):
    assert decoder.simulate(start_state, input_text) == expected_state
