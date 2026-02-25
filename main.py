from src.constrained_decoder import ConstrainedDecoder, JsonState, State
from src.vocabulary import Vocabulary
from llm_sdk import Small_LLM_Model
import json

def load_token_map(vocab_path) -> dict[str, int]:
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab

def apply_mask(logits: list[float], mask: list[float]) -> list[float]:
    # mask_len = len(mask)
    for i in range(len(logits)):
        print("before:", i, logits[i], mask[i])
        logits[i] += mask[i]
        print("after:", i, logits[i])

    return logits

def main():
    model = Small_LLM_Model()

    token_map = load_token_map(model.get_path_to_vocabulary_json())

    vocabulary = Vocabulary(token_map)
    decoder = ConstrainedDecoder(model, vocabulary)

    prompt = "say 'hello'"
    print(prompt)

    result = ""
    state = State(JsonState.START, ["hello"])
    for _ in range(10):
        ids = model.encode(prompt + result)
        logits = model.get_logits_from_input_ids(ids.tolist()[0])
        mask = decoder.get_logit_mask(state)
        masked_logits = [a + b for a, b in zip(logits, mask)]

        print(mask[:100])
        print([vocabulary[i] for i, logit in enumerate(mask[:100]) if logit >= 0])
        max_index = masked_logits.index(max(masked_logits))
        # if masked_logits[max_index] < 0:
        #     raise ValueError("No masked logit greater than 0")
        token = model.decode([max_index])
        result += token
        state = decoder.simulate(state, token)
        assert state.s != JsonState.INVALID
        print(state)
    print(result)

    # token = decoder.token_bytes[max_index]
    # print(token)







if __name__ == "__main__":
    main()
