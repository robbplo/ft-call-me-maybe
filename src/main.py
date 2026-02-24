from constrained_decoder import ConstrainedDecoder
from llm_sdk import Small_LLM_Model
import json

type Vocab = dict[str, int]
type ReverseVocab = dict[int, str]

def load_vocab(vocab_path) -> tuple[Vocab, ReverseVocab]:
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    reverse_vocab = {v: k for k, v in vocab.items()}
    return vocab, reverse_vocab

def main():
    model = Small_LLM_Model()
    # ids = model.encode("Hello from ft-call-me-maybe!")
    # print("ids", ids)

    vocab, reverse_vocab = load_vocab(model.get_path_to_vocabulary_json())

    decoder = ConstrainedDecoder(model)



if __name__ == "__main__":
    main()
