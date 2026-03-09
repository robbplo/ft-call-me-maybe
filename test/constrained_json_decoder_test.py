from __future__ import annotations

import pytest
import torch

from src.decoding import ConstrainedJSONDecoder, JsonState, State
from src.vocabulary import Vocabulary


class _FakeModel:
    def __init__(self, token_map: dict[str, int]) -> None:
        self._token_map: dict[int, str] = {v: k for k, v in token_map.items()}

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        if len(ids) != 1 or isinstance(ids, torch.Tensor):
            raise ValueError("Fake decoder expects a single token id.")
        return self._token_map[ids[0]]


def _make_decoder(tokens: list[str]) -> ConstrainedJSONDecoder:
    token_map = {token: index for index, token in enumerate(tokens)}
    model = _FakeModel(token_map)
    vocab = Vocabulary(token_map)
    return ConstrainedJSONDecoder(model, vocab)


decoder = _make_decoder(["a", "b", "c"])


@pytest.mark.parametrize(
    ("json_state", "depth", "input_text", "expected_state", "expected_depth"),
    [
        (JsonState.START, 0, "a", JsonState.INVALID, 0),
        (JsonState.START, 0, " b", JsonState.INVALID, 0),
        (JsonState.START, 0, "c ", JsonState.INVALID, 0),
        (JsonState.START, 0, "       ", JsonState.INVALID, 0),
        (JsonState.START, 0, "}", JsonState.INVALID, 0),
        (JsonState.START, 0, "{", JsonState.EXPECT_KEY, 1),
        (JsonState.START, 0, " {", JsonState.INVALID, 0),
        (JsonState.EXPECT_KEY, 1, "a", JsonState.INVALID, 1),
        (JsonState.EXPECT_KEY, 1, " b", JsonState.INVALID, 1),
        (JsonState.EXPECT_KEY, 1, "c ", JsonState.INVALID, 1),
        (JsonState.EXPECT_KEY, 1, '"', JsonState.IN_KEY, 1),
        (JsonState.EXPECT_KEY, 1, "       ", JsonState.EXPECT_KEY, 1),
        (JsonState.IN_KEY, 1, "a", JsonState.IN_KEY, 1),
        (JsonState.IN_KEY, 1, 'aaa"', JsonState.EXPECT_COLON, 1),
        (JsonState.EXPECT_COLON, 1, "a", JsonState.INVALID, 1),
        (JsonState.EXPECT_COLON, 1, " ", JsonState.EXPECT_COLON, 1),
        (JsonState.EXPECT_COLON, 1, ":", JsonState.EXPECT_VALUE, 1),
        (JsonState.EXPECT_VALUE, 1, "a", JsonState.INVALID, 1),
        (JsonState.EXPECT_VALUE, 1, " ", JsonState.EXPECT_VALUE, 1),
        (JsonState.EXPECT_VALUE, 1, "-", JsonState.NUMBER_AFTER_MINUS, 1),
        (JsonState.EXPECT_VALUE, 1, "0", JsonState.IN_NUMBER_ZERO, 1),
        (JsonState.EXPECT_VALUE, 1, "1", JsonState.IN_NUMBER, 1),
        (JsonState.EXPECT_VALUE, 1, '"', JsonState.IN_STRING, 1),
        (JsonState.EXPECT_VALUE, 1, "{", JsonState.EXPECT_KEY, 2),
        (JsonState.EXPECT_VALUE, 2, "{", JsonState.INVALID, 2),
        (JsonState.EXPECT_VALUE, 1, "t", JsonState.IN_TRUE_T, 1),
        (JsonState.EXPECT_VALUE, 1, "f", JsonState.IN_FALSE_F, 1),
        (JsonState.NUMBER_AFTER_MINUS, 1, "0", JsonState.IN_NUMBER_ZERO, 1),
        (JsonState.NUMBER_AFTER_MINUS, 1, "2", JsonState.IN_NUMBER, 1),
        (JsonState.IN_NUMBER_ZERO, 1, ".", JsonState.NUMBER_AFTER_DOT, 1),
        (JsonState.IN_NUMBER_ZERO, 1, "}", JsonState.END, 0),
        (JsonState.IN_NUMBER, 1, "1", JsonState.IN_NUMBER, 1),
        (JsonState.IN_NUMBER, 1, ".", JsonState.NUMBER_AFTER_DOT, 1),
        (JsonState.IN_NUMBER, 1, "e", JsonState.NUMBER_AFTER_EXP, 1),
        (JsonState.IN_NUMBER, 1, " ", JsonState.EXPECT_COMMA_OR_END, 1),
        (JsonState.IN_NUMBER, 1, ",", JsonState.EXPECT_KEY, 1),
        (JsonState.IN_NUMBER, 1, "}", JsonState.END, 0),
        (JsonState.IN_NUMBER, 2, "}", JsonState.EXPECT_COMMA_OR_END, 1),
        (JsonState.NUMBER_AFTER_DOT, 1, "5", JsonState.IN_FRACTION, 1),
        (JsonState.IN_FRACTION, 1, "e", JsonState.NUMBER_AFTER_EXP, 1),
        (JsonState.IN_FRACTION, 1, "}", JsonState.END, 0),
        (JsonState.NUMBER_AFTER_EXP, 1, "+", JsonState.NUMBER_AFTER_EXP_SIGN, 1),
        (JsonState.NUMBER_AFTER_EXP, 1, "2", JsonState.IN_EXPONENT, 1),
        (JsonState.NUMBER_AFTER_EXP_SIGN, 1, "2", JsonState.IN_EXPONENT, 1),
        (JsonState.IN_EXPONENT, 1, "}", JsonState.END, 0),
        (JsonState.IN_STRING, 1, "a", JsonState.IN_STRING, 1),
        (JsonState.IN_STRING, 1, "1", JsonState.IN_STRING, 1),
        (JsonState.IN_STRING, 1, " ", JsonState.IN_STRING, 1),
        (JsonState.IN_STRING, 1, ",", JsonState.IN_STRING, 1),
        (JsonState.IN_STRING, 1, '"', JsonState.EXPECT_COMMA_OR_END, 1),
        (JsonState.IN_TRUE_T, 1, "r", JsonState.IN_TRUE_TR, 1),
        (JsonState.IN_TRUE_TR, 1, "u", JsonState.IN_TRUE_TRU, 1),
        (JsonState.IN_TRUE_TRU, 1, "e", JsonState.EXPECT_COMMA_OR_END, 1),
        (JsonState.IN_FALSE_F, 1, "a", JsonState.IN_FALSE_FA, 1),
        (JsonState.IN_FALSE_FA, 1, "l", JsonState.IN_FALSE_FAL, 1),
        (JsonState.IN_FALSE_FAL, 1, "s", JsonState.IN_FALSE_FALS, 1),
        (JsonState.IN_FALSE_FALS, 1, "e", JsonState.EXPECT_COMMA_OR_END, 1),
        (JsonState.EXPECT_COMMA_OR_END, 1, "a", JsonState.INVALID, 1),
        (JsonState.EXPECT_COMMA_OR_END, 1, " ", JsonState.EXPECT_COMMA_OR_END, 1),
        (JsonState.EXPECT_COMMA_OR_END, 1, ",", JsonState.EXPECT_KEY, 1),
        (JsonState.EXPECT_COMMA_OR_END, 1, "}", JsonState.END, 0),
        (JsonState.EXPECT_COMMA_OR_END, 2, "}", JsonState.EXPECT_COMMA_OR_END, 1),
        (JsonState.START, 0, "{a", JsonState.INVALID, 1),
        (JsonState.START, 0, '{\n"aaa', JsonState.IN_KEY, 1),
        (JsonState.START, 0, '{"aaa": 1', JsonState.IN_NUMBER, 1),
        (JsonState.START, 0, '{"aaa": 1}', JsonState.END, 0),
        (JsonState.START, 0, '{"aaa": 1, "bbb": "', JsonState.IN_STRING, 1),
        (JsonState.START, 0, '{"aaa": 1, "bbb": "t', JsonState.IN_STRING, 1),
        (
            JsonState.START,
            0,
            '{"aaa": 1, "bbb": "t"',
            JsonState.EXPECT_COMMA_OR_END,
            1,
        ),
        (JsonState.START, 0, '{"aaa": 1, "bbb": "t"}', JsonState.END, 0),
        (JsonState.START, 0, '{"aaa": true}', JsonState.END, 0),
        (JsonState.START, 0, '{"aaa": -0.5e+2}', JsonState.END, 0),
        (JsonState.START, 0, '{"aaa": {"bbb": 1}', JsonState.EXPECT_COMMA_OR_END, 1),
        (JsonState.START, 0, '{"aaa": {"bbb": 1}}', JsonState.END, 0),
        (
            JsonState.START,
            0,
            '{"aaa": {"bbb": 1}, "ccc": {"ddd": "x"}}',
            JsonState.END,
            0,
        ),
    ],
)
def test_simulate_json(
    json_state: JsonState,
    depth: int,
    input_text: str,
    expected_state: JsonState,
    expected_depth: int,
) -> None:
    assert decoder._simulate_structure(json_state, input_text, depth) == (
        expected_state,
        expected_depth,
    )


mask_tokens = ["a", "b", "c", "{", "}", '"', ":", ",", "ff"]
mask_decoder = _make_decoder(mask_tokens)


@pytest.mark.parametrize(
    ("json_state", "valid_tokens"),
    [
        (JsonState.START, ["{"]),
        (JsonState.IN_KEY, ["a"]),
    ],
)
def test_get_logit_mask(
    json_state: JsonState, valid_tokens: list[str]
) -> None:
    mask = mask_decoder.get_logit_mask(
        State(json_state, allowed_keys=["function", "arguments"], depth=2)
    )
    for index, token in enumerate(mask_tokens):
        if token not in valid_tokens:
            assert mask[index] == float("-inf")
        else:
            assert mask[index] == 0


value_tokens = ['"', "1", "-", "t", "f", "{"] 
value_decoder = _make_decoder(value_tokens)


@pytest.mark.parametrize(
    ("arg_type", "valid_tokens"),
    [
        ("str", {'"'}),
        ("int", {"1", "-"}),
        ("float", {"1", "-"}),
        ("bool", {"t", "f"}),
    ],
)
def test_get_logit_mask_constrains_value_type(
    arg_type: str, valid_tokens: set[str]
) -> None:
    state = State(
        s=JsonState.EXPECT_VALUE,
        depth=2,
        allowed_keys=["arg"],
        allowed_types={"arg": arg_type},
        keys=["arg"],
        current_value_key="arg",
    )

    mask = value_decoder.get_logit_mask(state)

    for index, token in enumerate(value_tokens):
        if token in valid_tokens:
            assert mask[index] == 0.0
        else:
            assert mask[index] == float("-inf")


multi_tokens = [
    "a",
    "r",
    "1",
    "{",
    "}",
    '"',
    ":",
    ",",
    " ",
    '1, "r',
    '1, "a',
    "1}",
    ', "r',
]
multi_decoder = _make_decoder(multi_tokens)


class TestSimulateSchemaMultiChar:
    def test_multi_char_token_transitions_into_key(self) -> None:
        state = State(
            s=JsonState.IN_NUMBER,
            depth=2,
            allowed_keys=["role", "arg"],
            keys=[],
            current_key="",
        )

        allowed, next_state = multi_decoder.advance_state(state, '1, "r')

        assert allowed is True
        assert next_state.keys == []
        assert next_state.current_key == "r"
        assert next_state.current_value_key is None
        assert next_state.current_value_buffer == ""

    def test_multi_char_token_rejects_invalid_key_prefix(self) -> None:
        state = State(
            s=JsonState.IN_NUMBER,
            depth=2,
            allowed_keys=["role"],
            keys=[],
            current_key="",
        )

        allowed, _ = multi_decoder.advance_state(state, '1, "a')

        assert allowed is False

    def test_multi_char_token_accepts_valid_key_prefix(self) -> None:
        state = State(
            s=JsonState.IN_NUMBER,
            depth=2,
            allowed_keys=["arg"],
            keys=[],
            current_key="",
        )

        allowed, next_state = multi_decoder.advance_state(state, '1, "a')

        assert allowed is True
        assert next_state.current_key == "a"

    def test_closing_brace_blocked_when_required_keys_missing(self) -> None:
        state = State(
            s=JsonState.IN_NUMBER,
            depth=2,
            allowed_keys=["role", "arg"],
            keys=["role"],
            current_key="",
        )

        allowed, _ = multi_decoder.advance_state(state, "1}")

        assert allowed is False

    def test_closing_brace_allowed_when_all_keys_present(self) -> None:
        state = State(
            s=JsonState.IN_NUMBER,
            depth=2,
            allowed_keys=["role", "arg"],
            keys=["role", "arg"],
            current_key="",
        )

        allowed, next_state = multi_decoder.advance_state(state, "1}")

        assert allowed is True
        assert next_state.depth == 1
        assert next_state.s == JsonState.EXPECT_COMMA_OR_END

    def test_closing_brace_from_expect_comma_or_end(self) -> None:
        state = State(
            s=JsonState.EXPECT_COMMA_OR_END,
            depth=2,
            allowed_keys=["role", "arg"],
            keys=["role"],
            current_key="",
        )

        allowed, _ = multi_decoder.advance_state(state, "}")

        assert allowed is False

    def test_state_not_mutated_after_advance(self) -> None:
        state = State(
            s=JsonState.IN_NUMBER,
            depth=2,
            allowed_keys=["role", "arg"],
            allowed_types={"role": "int", "arg": "bool"},
            keys=["role"],
            current_key="",
            current_value_key="role",
            current_value_buffer="1",
        )

        multi_decoder.advance_state(state, '1, "r')

        assert state.s == JsonState.IN_NUMBER
        assert state.depth == 2
        assert state.keys == ["role"]
        assert state.current_key == ""
        assert state.current_value_key == "role"
        assert state.current_value_buffer == "1"

    def test_comma_then_key_start(self) -> None:
        state = State(
            s=JsonState.EXPECT_COMMA_OR_END,
            depth=2,
            allowed_keys=["role", "arg"],
            keys=[],
            current_key="",
        )

        allowed, next_state = multi_decoder.advance_state(state, ', "r')

        assert allowed is True
        assert next_state.current_key == "r"

    def test_key_close_sets_current_value_key(self) -> None:
        state = State(
            s=JsonState.IN_KEY,
            depth=2,
            allowed_keys=["a"],
            allowed_types={"a": "int"},
            keys=[],
            current_key="a",
        )

        allowed, next_state = multi_decoder.advance_state(state, '":')

        assert allowed is True
        assert next_state.keys == ["a"]
        assert next_state.current_key == ""
        assert next_state.current_value_key == "a"
        assert next_state.current_value_buffer == ""


typed_decoder = _make_decoder(["x"])


def test_string_value_must_start_with_quote() -> None:
    state = State(
        s=JsonState.EXPECT_VALUE,
        depth=2,
        allowed_keys=["name"],
        allowed_types={"name": "str"},
        keys=["name"],
        current_value_key="name",
    )

    allowed, _ = typed_decoder.advance_state(state, "1")

    assert allowed is False


def test_string_value_completion_clears_current_value_key() -> None:
    state = State(
        s=JsonState.EXPECT_VALUE,
        depth=2,
        allowed_keys=["name"],
        allowed_types={"name": "str"},
        keys=["name"],
        current_value_key="name",
    )

    allowed, next_state = typed_decoder.advance_state(state, '"ok"')

    assert allowed is True
    assert next_state.current_value_key is None
    assert next_state.current_value_buffer == ""


def test_int_value_rejects_decimal_suffix() -> None:
    state = State(
        s=JsonState.IN_NUMBER,
        depth=2,
        allowed_keys=["n"],
        allowed_types={"n": "int"},
        keys=["n"],
        current_value_key="n",
        current_value_buffer="1",
    )

    allowed, _ = typed_decoder.advance_state(state, ".")

    assert allowed is False


def test_float_value_accepts_exponent_prefix() -> None:
    state = State(
        s=JsonState.IN_NUMBER,
        depth=2,
        allowed_keys=["n"],
        allowed_types={"n": "float"},
        keys=["n"],
        current_value_key="n",
        current_value_buffer="1",
    )

    allowed, next_state = typed_decoder.advance_state(state, "e+2")

    assert allowed is True
    assert next_state.current_value_key == "n"
    assert next_state.current_value_buffer == "1e+2"


def test_bool_value_accepts_split_prefix_and_completion() -> None:
    state = State(
        s=JsonState.IN_TRUE_T,
        depth=2,
        allowed_keys=["flag"],
        allowed_types={"flag": "bool"},
        keys=["flag"],
        current_value_key="flag",
        current_value_buffer="t",
    )

    allowed, next_state = typed_decoder.advance_state(state, "rue")

    assert allowed is True
    assert next_state.current_value_key is None
    assert next_state.current_value_buffer == ""


def test_mixed_keys_switch_value_type_in_one_token() -> None:
    state = State(
        s=JsonState.IN_NUMBER,
        depth=2,
        allowed_keys=["a", "b"],
        allowed_types={"a": "int", "b": "bool"},
        keys=["a"],
        current_value_key="a",
        current_value_buffer="1",
    )

    allowed, next_state = typed_decoder.advance_state(state, ', "b": f')

    assert allowed is True
    assert next_state.keys == ["a", "b"]
    assert next_state.current_key == ""
    assert next_state.current_value_key == "b"
    assert next_state.current_value_buffer == "f"


def test_advance_state_primes_existing_generator_prefix() -> None:
    state = State(
        s=JsonState.START,
        depth=0,
        allowed_keys=["arg"],
        allowed_types={"arg": "int"},
    )

    allowed, next_state = typed_decoder.advance_state(
        state,
        '{"prompt": "Q","fn_name": "f","args": {',
    )

    assert allowed is True
    assert next_state.s == JsonState.EXPECT_KEY
    assert next_state.depth == 2
