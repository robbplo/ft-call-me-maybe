from __future__ import annotations
from src.state import State

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

tokens = ["a", "b", "c", "{", "}", '"', ":", ",", "ff"]
token_map = {t: i for i, t in enumerate(tokens)}
model = _FakeModel(token_map)
vocab = Vocabulary(token_map)
decoder = ConstrainedJSONDecoder(model, vocab)

@pytest.mark.parametrize(
    ("json_state", "valid_tokens"),
    [
        (JsonState.START, ["{"]),
        (JsonState.IN_KEY, ["f", "a"]),
    ],
)
def test_get_logit_mask(json_state, valid_tokens):
    mask = decoder.get_logit_mask(State(json_state, allowed_keys=["function", "arguments"], depth=2))
    print(mask)
    for i, token in enumerate(tokens):
        if token not in valid_tokens:
            assert mask[i] == float('-inf')
        else:
            assert mask[i] == 0


# --- Multi-character token schema tests ---

multi_tokens = ["a", "r", "1", '{', '}', '"', ':', ',', ' ', '1, "r', '1, "a', '1}', ', "r']
multi_token_map = {t: i for i, t in enumerate(multi_tokens)}
multi_model = _FakeModel(multi_token_map)
multi_vocab = Vocabulary(multi_token_map)
multi_decoder = ConstrainedJSONDecoder(multi_model, multi_vocab)


class TestSimulateSchemaMultiChar:
    """Tests for _simulate_schema with tokens spanning multiple state transitions."""

    def test_multi_char_token_transitions_into_key(self):
        """Token '1, "r' transitions IN_NUMBER -> EXPECT_KEY -> IN_KEY and should check key prefix."""
        state = State(
            s=JsonState.IN_NUMBER,
            depth=2,
            allowed_keys=["role", "arg"],
            keys=[],
            current_key="",
        )
        # 'r' is a valid prefix for "role"
        allowed, keys, current_key = multi_decoder._simulate_schema(state, '1, "r')
        assert allowed is True
        assert current_key == "r"

    def test_multi_char_token_rejects_invalid_key_prefix(self):
        """Token '1, "a' should be rejected when only 'role' is allowed."""
        state = State(
            s=JsonState.IN_NUMBER,
            depth=2,
            allowed_keys=["role"],
            keys=[],
            current_key="",
        )
        allowed, _, _ = multi_decoder._simulate_schema(state, '1, "a')
        assert allowed is False

    def test_multi_char_token_accepts_valid_key_prefix(self):
        """Token '1, "a' should be accepted when 'arg' is allowed."""
        state = State(
            s=JsonState.IN_NUMBER,
            depth=2,
            allowed_keys=["arg"],
            keys=[],
            current_key="",
        )
        allowed, _, current_key = multi_decoder._simulate_schema(state, '1, "a')
        assert allowed is True
        assert current_key == "a"

    def test_closing_brace_blocked_when_required_keys_missing(self):
        """Token '1}' should be rejected at depth 2 when required keys are missing."""
        state = State(
            s=JsonState.IN_NUMBER,
            depth=2,
            allowed_keys=["role", "arg"],
            keys=["role"],
            current_key="",
        )
        allowed, _, _ = multi_decoder._simulate_schema(state, '1}')
        assert allowed is False

    def test_closing_brace_allowed_when_all_keys_present(self):
        """Token '1}' should be accepted at depth 2 when all required keys are present."""
        state = State(
            s=JsonState.IN_NUMBER,
            depth=2,
            allowed_keys=["role", "arg"],
            keys=["role", "arg"],
            current_key="",
        )
        allowed, _, _ = multi_decoder._simulate_schema(state, '1}')
        assert allowed is True

    def test_closing_brace_from_expect_comma_or_end(self):
        """Token '}' from EXPECT_COMMA_OR_END should be rejected when keys are missing."""
        state = State(
            s=JsonState.EXPECT_COMMA_OR_END,
            depth=2,
            allowed_keys=["role", "arg"],
            keys=["role"],
            current_key="",
        )
        allowed, _, _ = multi_decoder._simulate_schema(state, '}')
        assert allowed is False

    def test_state_not_mutated_after_simulate(self):
        """_simulate_schema should not mutate the input state."""
        state = State(
            s=JsonState.IN_NUMBER,
            depth=2,
            allowed_keys=["role", "arg"],
            keys=["role"],
            current_key="",
        )
        multi_decoder._simulate_schema(state, '1, "r')
        assert state.s == JsonState.IN_NUMBER
        assert state.depth == 2
        assert state.keys == ["role"]
        assert state.current_key == ""

    def test_comma_then_key_start(self):
        """Token ', "r' from EXPECT_COMMA_OR_END should accept valid key prefix."""
        state = State(
            s=JsonState.EXPECT_COMMA_OR_END,
            depth=2,
            allowed_keys=["role", "arg"],
            keys=[],
            current_key="",
        )
        allowed, _, current_key = multi_decoder._simulate_schema(state, ', "r')
        assert allowed is True
        assert current_key == "r"

    def test_empty_keys_should_be_rejected(self):
        """_simulate_schema should not produce empty keys"""
        state = State(
            s=JsonState.IN_KEY,
            depth=2,
            allowed_keys=["a"],
            keys=[],
            current_key="a",
        )
        # aState(s=<JsonState.IN_KEY: 5>, depth=2, allowed_keys=['a'], keys=[''], current_key='a')
        allowed, keys, current_key = multi_decoder._simulate_schema(state, '":')
        assert allowed is True
        assert keys == ["a"]
        assert current_key == ""
