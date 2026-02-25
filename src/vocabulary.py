import json
from typing import cast
from llm_sdk import Small_LLM_Model


class Vocabulary:
    def __init__(self, input: dict[str, int]):
        self.data: dict[str, int] = input
        self.reverse: dict[int, str] = {v: k for k, v in self.data.items()}
        self.size: int = len(input)

    @staticmethod
    def from_model(model: Small_LLM_Model) -> 'Vocabulary':
        path = model.get_path_to_vocabulary_json()
        with open(path, "r") as f:
            vocab = cast(dict[str, int], json.load(f))
        return Vocabulary(vocab)

    def __getitem__(self, key: str | int) -> str | int:
        if isinstance(key, str):
            return self.data[key]
        return self.reverse[key]


