from src.state import State, JsonState

def test_key_char_valid():
    state = State(JsonState.IN_KEY, ["aaa", "bbb", "cc"])
    assert state.key_char_valid("a")
    assert state.key_char_valid("b")
    assert state.key_char_valid("c")
    assert not state.key_char_valid("d")

    state.current_key = "aa"
    assert state.key_char_valid("a")
    assert not state.key_char_valid("b")
    assert not state.key_char_valid("c")
    assert not state.key_char_valid("d")

    state.current_key = "bbb"
    assert not state.key_char_valid("a")
    assert not state.key_char_valid("b")
    assert not state.key_char_valid("c")
    assert not state.key_char_valid("d")
