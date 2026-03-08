from src.function_call_generator import FunctionCallGenerator
from src.vocabulary import Vocabulary
from src.llm_sdk import Small_LLM_Model
import json


def load_token_map(vocab_path) -> dict[str, int]:
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab


def main():
    model = Small_LLM_Model()
    token_map = load_token_map(model.get_path_to_vocabulary_json())
    vocab = Vocabulary(token_map)
    generator = FunctionCallGenerator(model, vocab)

    with open("data/input/function_calling_tests.json", "r") as f:
        questions = [q["prompt"] for q in json.load(f)]

    function_calls = [generator.generate(question) for question in questions]

    with open("data/output/function_calling_results.json", "w") as f:
        json.dump([fc.model_dump() for fc in function_calls], f, indent=2)


if __name__ == "__main__":
    main()
