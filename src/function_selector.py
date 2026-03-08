from src.models.function_definition import FunctionDefinition, FunctionDefinitions
from src.constrained_string_decoder import ConstrainedStringDecoder
from src.vocabulary import Vocabulary
from src.llm_sdk import Small_LLM_Model

class FunctionSelector:
    def __init__(self, model: Small_LLM_Model, vocab: Vocabulary):
        self.functions: dict[str, FunctionDefinition] = self._load_functions()
        self.model: Small_LLM_Model = model
        self.decoder: ConstrainedStringDecoder = ConstrainedStringDecoder(
            model, vocab, list(self.functions.keys()))

    def select_function(self, prompt: str) -> FunctionDefinition:
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
        return self.functions[result]


    def _load_functions(self) -> dict[str, FunctionDefinition]:
        defs = FunctionDefinitions.from_file("data/input/functions_definition.json")
        return {d.fn_name: d for d in defs.root}

