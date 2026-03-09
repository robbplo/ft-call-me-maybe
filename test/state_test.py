import pytest

from src.decoding import JsonState, State


@pytest.mark.parametrize(
    ("depth", "steps"),
    [
        (
            2,
            [
                ("d", False, [], ""),
                ('"', False, [], ""),
                ("a", True, [], "a"),
                ("b", False, [], "a"),
                ("a", True, [], "aa"),
                ("a", True, [], "aaa"),
                ("a", False, [], "aaa"),
                ('"', True, ["aaa"], ""),
                ("a", False, ["aaa"], ""),
                ("c", True, ["aaa"], "c"),
                ("c", True, ["aaa"], "cc"),
                ("c", False, ["aaa"], "cc"),
                ('"', True, ["aaa", "cc"], ""),
            ],
        ),
        (
            1,
            [
                ("d", True, [], ""),
                ('"', True, [], ""),
                ("a", True, [], ""),
                ("b", True, [], ""),
                ("a", True, [], ""),
                ("a", True, [], ""),
                ("a", True, [], ""),
                ('"', True, [], ""),
                ("a", True, [], ""),
                ("c", True, [], ""),
                ("c", True, [], ""),
                ("c", True, [], ""),
                ('"', True, [], ""),
            ],
        ),
    ],
    ids=["depth_2", "depth_1"],
)
def test_add_key_char(
    depth: int, steps: list[tuple[str, bool, list[str], str]]
) -> None:
    state = State(JsonState.IN_KEY, depth=depth,
                  allowed_keys=["aaa", "bbb", "cc"])

    for char, expected_allowed, expected_keys, expected_current_key in steps:
        allowed, keys, current_key = state.add_key_char(char)

        assert (allowed, keys, current_key) == (
            expected_allowed,
            expected_keys,
            expected_current_key,
        )

        state.keys = keys
        state.current_key = current_key
