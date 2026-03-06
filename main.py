from src.constrained_string_decoder import ConstrainedStringDecoder
from src.constrained_decoder import ConstrainedJSONDecoder, JsonState, State
from src.vocabulary import Vocabulary
from llm_sdk import Small_LLM_Model
import json

def load_token_map(vocab_path) -> dict[str, int]:
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab

tests = []
with open("./data/input/function_calling_tests.json", "r") as f:
    for obj in json.load(f):
        tests.append(obj["prompt"])

def pick_functions(model: Small_LLM_Model, vocabulary: Vocabulary):
    functions = [
      "fn_add_numbers",
      "fn_get_square_root",
      "fn_greet",
      "fn_is_even",
      "fn_multiply_numbers",
      "fn_reverse_string",
      "fn_substitute_string_with_regex"
    ]
    decoder = ConstrainedStringDecoder(model, vocabulary,functions)
    for prompt in tests:
        print(prompt)
        full_prompt = f"""
I know these functions:
- {'\n- '.join(functions)}
QUESTION: {prompt}
The function which applies best to the question is: """
        result = ""
        for _ in range(10):
            ids = model.encode(full_prompt + result)
            logits = model.get_logits_from_input_ids(ids.tolist()[0])
            mask = decoder.get_logit_mask(result)
            masked_logits = [a + b for a, b in zip(logits, mask)]

            max_index = masked_logits.index(max(masked_logits))
            token = model.decode([max_index])
            result += token

            substrings = [f for f in functions if f.startswith(result)]
            if len(substrings) == 1:
                result = substrings[0]
                break
        print(result)

def generate_json(model: Small_LLM_Model, vocabulary: Vocabulary):
    decoder = ConstrainedJSONDecoder(model, vocabulary)

    question = "Replace all vowels in 'Programming is fun' with asterisks"
    prompt = f"""
PROMPT: {question}
ARGUMENTS: 
- regex
- source_string
- replacement
FUNCTION CALL: """
    print(prompt)

    result = '{' + \
    f'"prompt": "{question}",' + \
    '"fn_name": "fn_substitute_string_with_regex",' + \
    '"args": {"regex": "'
    print(result, end="")
    # Apply schema state of initial result
    state = State(JsonState.START, depth=0, allowed_keys=["regex", "source_string", "replacement"],
                  keys=["regex"], current_key="")
    # Apply initial structure
    next_s, next_depth = decoder._simulate_structure(state.s, result, state.depth)
    state.s, state.depth = next_s, next_depth
    # Apply initial schema
    # for char in result:
    #     _, keys, current_key = decoder._simulate_schema(state, char)
    #     print(state)
    #     state.keys, state.current_key = keys, current_key
    for _ in range(100):
        ids = model.encode(prompt + result)
        logits = model.get_logits_from_input_ids(ids.tolist()[0])

        mask = decoder.get_logit_mask(state)
        masked_logits = [a + b for a, b in zip(logits, mask)]
        max_index = masked_logits.index(max(masked_logits))

        token = model.decode([max_index])
        result += token

        # update json structure
        next_s, next_depth = decoder._simulate_structure(state.s, token, state.depth)
        state.s = next_s
        state.depth = next_depth
        # update json schema
        _, keys, current_key = decoder._simulate_schema(state, token)
        state.keys = keys
        state.current_key = current_key
        print(token, end="", flush=True)
        # print(keys, current_key)
        if state.s == JsonState.END:
            break
        assert state.s != JsonState.INVALID
    print("Final result: \n")
    print(result)
    print(state)

def main():
    model = Small_LLM_Model()
    token_map = load_token_map(model.get_path_to_vocabulary_json())
    vocabulary = Vocabulary(token_map)

    # pick_functions(model, vocabulary)
    generate_json(model, vocabulary)


if __name__ == "__main__":
    main()
