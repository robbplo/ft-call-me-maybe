from __future__ import annotations

import pytest
import torch

from src.vocabulary import Vocabulary
from src.constrained_decoder import ConstrainedJSONDecoder, JsonState

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
decoder = ConstrainedJSONDecoder(model, vocab)

@pytest.mark.parametrize(
    ("json_state", "depth", "input_text", "expected_state", "expected_depth"),
    [
        # Start
        (JsonState.START, 0, 'a', JsonState.INVALID, 0),
        (JsonState.START, 0, ' b', JsonState.INVALID, 0),
        (JsonState.START, 0, 'c ', JsonState.INVALID, 0),
        (JsonState.START, 0, '       ', JsonState.INVALID, 0),
        (JsonState.START, 0, '}', JsonState.INVALID, 0),
        (JsonState.START, 0, '{', JsonState.EXPECT_KEY, 1),
        (JsonState.START, 0, ' {', JsonState.INVALID, 0),
        # Expect key
        (JsonState.EXPECT_KEY, 1, 'a', JsonState.INVALID, 1),
        (JsonState.EXPECT_KEY, 1, ' b', JsonState.INVALID, 1),
        (JsonState.EXPECT_KEY, 1, 'c ', JsonState.INVALID, 1),
        (JsonState.EXPECT_KEY, 1, '"', JsonState.IN_KEY, 1),
        (JsonState.EXPECT_KEY, 1, '       ', JsonState.EXPECT_KEY, 1),
        # In key
        (JsonState.IN_KEY, 1, 'a', JsonState.IN_KEY, 1),
        (JsonState.IN_KEY, 1, 'aaa"', JsonState.EXPECT_COLON, 1),
        # Expect colon
        (JsonState.EXPECT_COLON, 1, 'a', JsonState.INVALID, 1),
        (JsonState.EXPECT_COLON, 1, ' ', JsonState.EXPECT_COLON, 1),
        (JsonState.EXPECT_COLON, 1, ':', JsonState.EXPECT_VALUE, 1),
        # Expect value
        (JsonState.EXPECT_VALUE, 1, 'a', JsonState.INVALID, 1),
        (JsonState.EXPECT_VALUE, 1, ' ', JsonState.EXPECT_VALUE, 1),
        (JsonState.EXPECT_VALUE, 1, '1', JsonState.IN_NUMBER, 1),
        (JsonState.EXPECT_VALUE, 1, '"', JsonState.IN_STRING, 1),
        (JsonState.EXPECT_VALUE, 1, '{', JsonState.EXPECT_KEY, 2),
        # In number
        (JsonState.IN_NUMBER, 1, 'a', JsonState.INVALID, 1),
        (JsonState.IN_NUMBER, 1, '1', JsonState.IN_NUMBER, 1),
        (JsonState.IN_NUMBER, 1, ' ', JsonState.EXPECT_COMMA_OR_END, 1),
        (JsonState.IN_NUMBER, 1, ',', JsonState.EXPECT_KEY, 1),
        (JsonState.IN_NUMBER, 1, '}', JsonState.END, 0),
        (JsonState.IN_NUMBER, 2, '}', JsonState.EXPECT_COMMA_OR_END, 1),
        # In string
        (JsonState.IN_STRING, 1, 'a', JsonState.IN_STRING, 1),
        (JsonState.IN_STRING, 1, '1', JsonState.IN_STRING, 1),
        (JsonState.IN_STRING, 1, ' ', JsonState.IN_STRING, 1),
        (JsonState.IN_STRING, 1, ',', JsonState.IN_STRING, 1),
        (JsonState.IN_STRING, 1, '"', JsonState.EXPECT_COMMA_OR_END, 1),
        # Expect comma or end
        (JsonState.EXPECT_COMMA_OR_END, 1, 'a', JsonState.INVALID, 1),
        (JsonState.EXPECT_COMMA_OR_END, 1, ' ', JsonState.EXPECT_COMMA_OR_END, 1),
        (JsonState.EXPECT_COMMA_OR_END, 1, ',', JsonState.EXPECT_KEY, 1),
        (JsonState.EXPECT_COMMA_OR_END, 1, '}', JsonState.END, 0),
        (JsonState.EXPECT_COMMA_OR_END, 2, '}', JsonState.EXPECT_COMMA_OR_END, 1),
        # Multiple states in one token
        (JsonState.START, 0, '{a', JsonState.INVALID, 1),
        (JsonState.START, 0, '{\n"aaa', JsonState.IN_KEY, 1),
        (JsonState.START, 0, '{"aaa": 1', JsonState.IN_NUMBER, 1),
        (JsonState.START, 0, '{"aaa": 1}', JsonState.END, 0),
        (JsonState.START, 0, '{"aaa": 1, "bbb": "', JsonState.IN_STRING, 1),
        (JsonState.START, 0, '{"aaa": 1, "bbb": "t', JsonState.IN_STRING, 1),
        (JsonState.START, 0, '{"aaa": 1, "bbb": "t"', JsonState.EXPECT_COMMA_OR_END, 1),
        (JsonState.START, 0, '{"aaa": 1, "bbb": "t"}', JsonState.END, 0),
        (JsonState.START, 0, '{"aaa": {"bbb": 1}', JsonState.EXPECT_COMMA_OR_END, 1),
        (JsonState.START, 0, '{"aaa": {"bbb": 1}}', JsonState.END, 0),
        (JsonState.START, 0, '{"aaa": {"bbb": 1}, "ccc": {"ddd": "x"}}', JsonState.END, 0),

    ],
)
def test_simulate_json(json_state, depth, input_text, expected_state, expected_depth):
    assert decoder._simulate_structure(json_state, input_text, depth) == (
        expected_state,
        expected_depth,
    )

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
