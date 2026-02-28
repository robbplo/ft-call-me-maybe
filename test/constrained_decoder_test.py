from __future__ import annotations

from pathlib import Path
import pytest
import torch

from src.vocabulary import Vocabulary
from src.constrained_decoder import ConstrainedDecoder, JsonState, State

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
    ("json_state", "input_text", "expected_state"),
    [
        # Start
        (JsonState.START, 'a', JsonState.INVALID),
        (JsonState.START, ' b', JsonState.INVALID),
        (JsonState.START, 'c ', JsonState.INVALID),
        (JsonState.START, '       ', JsonState.INVALID),
        (JsonState.START, '}', JsonState.INVALID),
        (JsonState.START, '{', JsonState.EXPECT_KEY),
        (JsonState.START, ' {', JsonState.INVALID),
        # Expect key
        (JsonState.EXPECT_KEY, 'a', JsonState.INVALID),
        (JsonState.EXPECT_KEY, ' b', JsonState.INVALID),
        (JsonState.EXPECT_KEY, 'c ', JsonState.INVALID),
        (JsonState.EXPECT_KEY, '"', JsonState.IN_KEY),
        (JsonState.EXPECT_KEY, '       ', JsonState.EXPECT_KEY),
        # In key
        (JsonState.IN_KEY, 'a', JsonState.IN_KEY),
        (JsonState.IN_KEY, 'aaa"', JsonState.EXPECT_COLON),
        # Expect colon
        (JsonState.EXPECT_COLON, 'a', JsonState.INVALID),
        (JsonState.EXPECT_COLON, ' ', JsonState.EXPECT_COLON),
        (JsonState.EXPECT_COLON, ':', JsonState.EXPECT_VALUE),
        # Expect value
        (JsonState.EXPECT_VALUE, 'a', JsonState.INVALID),
        (JsonState.EXPECT_VALUE, ' ', JsonState.EXPECT_VALUE),
        (JsonState.EXPECT_VALUE, '1', JsonState.IN_NUMBER),
        (JsonState.EXPECT_VALUE, '"', JsonState.IN_STRING),
        # In number
        (JsonState.IN_NUMBER, 'a', JsonState.INVALID),
        (JsonState.IN_NUMBER, '1', JsonState.IN_NUMBER),
        (JsonState.IN_NUMBER, ' ', JsonState.EXPECT_COMMA_OR_END),
        (JsonState.IN_NUMBER, ',', JsonState.EXPECT_KEY),
        (JsonState.IN_NUMBER, '}', JsonState.END),
        # In string
        (JsonState.IN_STRING, 'a', JsonState.IN_STRING),
        (JsonState.IN_STRING, '1', JsonState.IN_STRING),
        (JsonState.IN_STRING, ' ', JsonState.IN_STRING),
        (JsonState.IN_STRING, ',', JsonState.IN_STRING),
        (JsonState.IN_STRING, '"', JsonState.EXPECT_COMMA_OR_END),
        # Expect comma or end
        (JsonState.EXPECT_COMMA_OR_END, 'a', JsonState.INVALID),
        (JsonState.EXPECT_COMMA_OR_END, ' ', JsonState.EXPECT_COMMA_OR_END),
        (JsonState.EXPECT_COMMA_OR_END, ',', JsonState.EXPECT_KEY),
        (JsonState.EXPECT_COMMA_OR_END, '}', JsonState.END),
        # Multiple states in one token
        (JsonState.START, '{a', JsonState.INVALID),
        (JsonState.START, '{\n"aaa', JsonState.IN_KEY),
        (JsonState.START, '{"aaa": 1', JsonState.IN_NUMBER),
        (JsonState.START, '{"aaa": 1}', JsonState.END),
        (JsonState.START, '{"aaa": 1, "bbb": "', JsonState.IN_STRING),
        (JsonState.START, '{"aaa": 1, "bbb": "t', JsonState.IN_STRING),
        (JsonState.START, '{"aaa": 1, "bbb": "t"', JsonState.EXPECT_COMMA_OR_END),
        (JsonState.START, '{"aaa": 1, "bbb": "t"}', JsonState.END),

    ],
)
def test_simulate_json(json_state, input_text, expected_state):
    assert decoder._simulate_structure(json_state, input_text) == expected_state

# tokens = ["a", "b", "c", "{", "}", '"', ":", ","]
# token_map = {t: i for i, t in enumerate(tokens)}
# model = _FakeModel(token_map)
# vocab = Vocabulary(token_map)
# decoder = ConstrainedDecoder(model, vocab)
#
# @pytest.mark.parametrize(
#     ("json_state", "valid_tokens"),
#     [
#         (JsonState.START, ["{"]),
#         (JsonState.IN_KEY, ["f", "a"]),
#     ],
# )
# def test_get_logit_mask(json_state, valid_tokens):
#     mask = decoder.get_logit_mask(State(json_state, ["function", "arguments"]))
#     print(mask)
#     for i, token in enumerate(tokens):
#         if token not in valid_tokens:
#             assert mask[i] == float('-inf')
#         else:
#             assert mask[i] == 0
