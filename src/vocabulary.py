import json
from typing import cast

from src.llm_sdk import Small_LLM_Model


class Vocabulary:
    """Bidirectional mapping between token strings and their integer IDs.

    Attributes:
        data: Forward mapping from token string to token ID.
        reverse: Reverse mapping from token ID to token string.
        size: Total number of tokens in the vocabulary.
    """

    def __init__(self, token_map: dict[str, int]) -> None:
        """Initialize the vocabulary from a token-to-ID mapping.

        Args:
            token_map: Dictionary mapping token strings to integer IDs.
        """
        self.data: dict[str, int] = token_map
        self.reverse: dict[int, str] = {v: k for k, v in self.data.items()}
        self.size: int = len(token_map)

    @staticmethod
    def from_model(model: Small_LLM_Model) -> "Vocabulary":
        """Build a Vocabulary by reading the JSON file referenced by the model.

        Args:
            model: The LLM instance whose vocabulary path will be used.

        Returns:
            A new Vocabulary loaded from the model's vocabulary JSON file.
        """
        path = model.get_path_to_vocabulary_json()
        with open(path, "r") as f:
            token_map = cast(dict[str, int], json.load(f))
        return Vocabulary(token_map)

    def __getitem__(self, key: int) -> str:
        """Return the token string for a given token ID.

        Args:
            key: Integer token ID to look up.

        Returns:
            The token string corresponding to *key*.
        """
        return self.reverse[key]
