from src.state import State, JsonState

# def test_add_key_char():
#     state = State(JsonState.IN_KEY, ["aaa", "bbb", "cc"])
#     assert not state.add_key_char("d")
#     assert not state.add_key_char('"')
#     assert state.current_key == ""
#
#     assert state.add_key_char("a")
#     assert not state.add_key_char("b")
#     assert state.add_key_char("a")
#     assert state.add_key_char("a")
#     assert not state.add_key_char("a")
#     assert state.current_key == "aaa"
#     assert state.add_key_char('"')
#     assert state.keys == ["aaa"]
#     assert state.current_key == ""
#
#     assert not state.add_key_char("a")
#     assert state.add_key_char("c")
#     assert state.add_key_char("c")
#     assert not state.add_key_char("c")
#     assert state.add_key_char('"')
#     assert state.keys == ["aaa", "cc"]
#     assert state.current_key == ""
#
#     # state.current_key = "aa"
#     # assert state.add_key_char("a")
#     # assert not state.add_key_char("b")
#     # assert not state.add_key_char("c")
#     # assert not state.add_key_char("d")
#     #
#     # state.current_key = "bbb"
#     # assert not state.add_key_char("a")
#     # assert not state.add_key_char("b")
#     # assert not state.add_key_char("c")
#     # assert not state.add_key_char("d")
