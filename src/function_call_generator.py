from src.models.function_definition import FunctionDefinition
from src.models.function_call import FunctionCall
from src.function_selector import FunctionSelector
from src.constrained_decoder import ConstrainedJSONDecoder, JsonState, State
from src.vocabulary import Vocabulary
from src.llm_sdk import Small_LLM_Model


class FunctionCallGenerator:
    def __init__(self, model: Small_LLM_Model, vocab: Vocabulary):
        self.model = model
        self.vocab = vocab
        self.decoder = ConstrainedJSONDecoder(model, vocab)
        self.function_selector = FunctionSelector(model, vocab)

    def generate(self, question: str) -> FunctionCall:
        function = self.function_selector.select_function(question)
        prompt, result = self._build_prompt(question, function)

        print(question)
        print(result, end="")

        result = self._generate_json(prompt, result, function)
        return FunctionCall.model_validate_json(result)

    def _build_prompt(self, question: str, function: FunctionDefinition) -> tuple[str, str]:
        prompt = f"""PROMPT: {question}
    ARGUMENTS:
    - {'- \n'.join(function.args_names)}
    FUNCTION CALL: """

        initial_result = '{' + \
            f'"prompt": "{question}",' + \
            f'"fn_name": "{function.fn_name}",' + \
            '"args": {'
        return prompt, initial_result

    def _generate_json(self, prompt: str, result: str, function: FunctionDefinition) -> str:
        state = State(JsonState.START, depth=0, allowed_keys=function.args_names,
                      keys=[""], current_key="")
        next_s, next_depth = self.decoder._simulate_structure(state.s, result, state.depth)
        state.s, state.depth = next_s, next_depth

        for _ in range(100):
            ids = self.model.encode(prompt + result)
            logits = self.model.get_logits_from_input_ids(ids.tolist()[0])

            mask = self.decoder.get_logit_mask(state)
            masked_logits = [a + b for a, b in zip(logits, mask)]
            max_index = masked_logits.index(max(masked_logits))

            token = self.model.decode([max_index])
            result += token

            _, keys, current_key = self.decoder._simulate_schema(state, token, prints=True)
            state.keys = keys
            state.current_key = current_key

            next_s, next_depth = self.decoder._simulate_structure(state.s, token, state.depth)
            state.s = next_s
            state.depth = next_depth

            print(token, end="", flush=True)

            if state.s == JsonState.END:
                break
            assert state.s != JsonState.INVALID

        return result
