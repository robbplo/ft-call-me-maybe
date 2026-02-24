from src.constrained_decoder import ConstrainedDecoder, S
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
    decoder = ConstrainedDecoder(model, vocabulary)

    ids = model.encode("generate a json with key 'hello' and value 'world'. JSON:")

    logits = model.get_logits_from_input_ids(ids.tolist()[0])
    mask = decoder.get_logit_mask(S.START)
    masked_logits = [a * b for a, b in zip(logits, mask)]


    print(len([a for a in masked_logits if a > 0]))
    max_index = masked_logits.index(max(masked_logits))
    print("max token", max_index)
    # token = decoder.token_bytes[max_index]
    # print(token)







if __name__ == "__main__":
    main()
