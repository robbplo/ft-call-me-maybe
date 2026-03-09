from pydantic import ValidationError

from src.models.function_definition import (
    FunctionDefinition,
    FunctionDefinitions,
)
from src.decoding import ConstrainedStringDecoder
from src.vocabulary import Vocabulary
from src.llm_sdk import Small_LLM_Model


class FunctionSelector:
    """Selects the most appropriate function for a natural-language prompt.

    Uses the LLM together with a :class:`ConstrainedStringDecoder` to perform
    greedy constrained decoding over the set of known function names, ensuring
    the model always produces a valid function name.

    Attributes:
        functions: Mapping from function name to its full definition.
        model: LLM used for logit generation.
        decoder: Constrained string decoder restricted to known function names.
    """

    def __init__(self, model: Small_LLM_Model, vocab: Vocabulary) -> None:
        """Initialize the selector by loading function definitions.

        Args:
            model: LLM instance used for constrained generation.
            vocab: Vocabulary shared with the rest of the pipeline.
        """
        self.functions: dict[str, FunctionDefinition] = self._load_functions()
        self.model: Small_LLM_Model = model
        self.decoder: ConstrainedStringDecoder = ConstrainedStringDecoder(
            model, vocab, list(self.functions.keys()))

    def select_function(self, prompt: str) -> FunctionDefinition:
        """Select the function that best matches *prompt*.

        Runs constrained greedy decoding until exactly one function name
        remains as a valid completion of the generated prefix.

        Args:
            prompt: Natural-language question or instruction.

        Returns:
            The :class:`FunctionDefinition` whose name was generated.
        """
        functions = self.decoder.allowed_strings
        full_prompt = f"""
I know these functions:
- {'\n- '.join(functions)}
QUESTION: {prompt}
The function which applies best to the question is: """
        result = ""
        for _ in range(20):
            ids = self.model.encode(full_prompt + result)
            logits = self.model.get_logits_from_input_ids(ids.tolist()[0])
            mask = self.decoder.get_logit_mask(result)
            masked_logits = [a + b for a, b in zip(logits, mask)]

            max_index = masked_logits.index(max(masked_logits))
            token = self.model.decode([max_index])
            result += token

            substrings = [f for f in functions if f.startswith(result)]
            if len(substrings) == 1:
                result = substrings[0]
                break
        if result not in self.functions:
            raise ValueError(
                f"Decoding did not converge to a valid function name; "
                f"got {result!r}. Known functions: "
                f"{list(self.functions.keys())}"
            )
        return self.functions[result]

    def _load_functions(self) -> dict[str, FunctionDefinition]:
        """Load function definitions from the default input file.

        Returns:
            A dictionary mapping each function name to its definition.
        """
        path = "data/input/functions_definition.json"
        try:
            defs = FunctionDefinitions.from_file(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Function definitions file not found: {path}"
            )
        except (OSError, ValidationError) as e:
            raise RuntimeError(
                f"Failed to load function definitions from {path}: {e}"
            ) from e
        return {d.fn_name: d for d in defs.root}
