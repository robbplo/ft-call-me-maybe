from src.constrained_decoder import ConstrainedJSONDecoder, JsonState, State
from src.vocabulary import Vocabulary
from llm_sdk import Small_LLM_Model
import json

def load_token_map(vocab_path) -> dict[str, int]:
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab

def main():
    model = Small_LLM_Model()

    token_map = load_token_map(model.get_path_to_vocabulary_json())

    vocabulary = Vocabulary(token_map)
    decoder = ConstrainedJSONDecoder(model, vocabulary)

    prompt = "say 'hello'"
    print(prompt)

    result = ""
    state = State(JsonState.START, allowed_keys=["hello"])
    for _ in range(10):
        ids = model.encode(prompt + result)
        logits = model.get_logits_from_input_ids(ids.tolist()[0])
        mask = decoder.get_logit_mask(state)
        masked_logits = [a + b for a, b in zip(logits, mask)]

        max_index = masked_logits.index(max(masked_logits))
        # print([vocabulary[i] for i, l in enumerate(masked_logits) if l != float('-inf')])
        token = model.decode([max_index])
        result += token
        # print(state.s, token, max_index)
        state.s = decoder._simulate_structure(state.s, token)
        if state.s == JsonState.END:
            break
        assert state.s != JsonState.INVALID
        print(token)
    print(result)

    # token = decoder.token_bytes[max_index]
    # print(token)







if __name__ == "__main__":
    main()
