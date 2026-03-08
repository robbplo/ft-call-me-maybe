from src.models.function_definition import FunctionDefinition
from src.models.function_call import FunctionCall
from src.function_selector import FunctionSelector
from src.constrained_string_decoder import ConstrainedStringDecoder
from src.constrained_decoder import ConstrainedJSONDecoder, JsonState, State
from src.vocabulary import Vocabulary
from src.llm_sdk import Small_LLM_Model
import json


def generate_json(model: Small_LLM_Model, decoder: ConstrainedJSONDecoder, prompt: str, result: str, function: FunctionDefinition):
    # Apply schema state of initial result
    state = State(JsonState.START, depth=0, allowed_keys=function.args_names,
                  keys=[""], current_key="")
    # Apply initial structure
    next_s, next_depth = decoder._simulate_structure(state.s, result, state.depth)
    state.s, state.depth = next_s, next_depth
    for _ in range(100):
        ids = model.encode(prompt + result)
        logits = model.get_logits_from_input_ids(ids.tolist()[0])

        mask = decoder.get_logit_mask(state)
        masked_logits = [a + b for a, b in zip(logits, mask)]
        max_index = masked_logits.index(max(masked_logits))

        token = model.decode([max_index])
        result += token

        # update schema
        _, keys, current_key = decoder._simulate_schema(state, token, prints=True)
        state.keys = keys
        state.current_key = current_key

        # update json structure
        next_s, next_depth = decoder._simulate_structure(state.s, token, state.depth)
        state.s = next_s
        state.depth = next_depth

        print(token, end="", flush=True)

        if state.s == JsonState.END:
            break
        assert state.s != JsonState.INVALID
    return result

def load_token_map(vocab_path) -> dict[str, int]:
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab

def build_prompt(question: str, function: FunctionDefinition) -> tuple[str, str]:
    prompt = f"""PROMPT: {question}
    ARGUMENTS: 
    - {'- \n'.join(function.args_names)}
    FUNCTION CALL: """

    initial_result = '{' + \
    f'"prompt": "{question}",' + \
    f'"fn_name": "{function.fn_name}",' + \
    '"args": {'
    return prompt, initial_result

def main():
    model = Small_LLM_Model()
    token_map = load_token_map(model.get_path_to_vocabulary_json())
    vocab = Vocabulary(token_map)
    function_selector = FunctionSelector(model, vocab)

    questions = []
    with open("data/input/function_calling_tests.json", "r") as f:
        questions = [q["prompt"] for q in json.load(f)]


    function_calls = []

    decoder = ConstrainedJSONDecoder(model, vocab)
    for question in questions:
        function = function_selector.select_function(question)
        prompt, result = build_prompt(question, function)

        print(question)
        print(result, end="")


        result = generate_json(model, decoder, prompt, result, function)
        function_call = FunctionCall.model_validate_json(result)
        function_calls.append(function_call)

    with open("data/output/function_calling_results.json", "w") as f:
        function_call_dicts = [fc.model_dump() for fc in function_calls]
        json.dump(function_call_dicts, f, indent=2)


if __name__ == "__main__":
    main()
