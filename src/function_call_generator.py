from src.models.function_definition import FunctionDefinition
from src.models.function_call import FunctionCall
from src.function_selector import FunctionSelector
from src.constrained_decoder import ConstrainedJSONDecoder, JsonState, State
from src.vocabulary import Vocabulary
from src.llm_sdk import Small_LLM_Model


class FunctionCallGenerator:
    """Orchestrates function selection and constrained JSON generation.

    Given a natural-language question this class selects the most appropriate
    function using the LLM, constructs a structured prompt, and then drives
    token-by-token constrained decoding to produce a valid JSON
    :class:`~src.models.function_call.FunctionCall`.

    Attributes:
        model: LLM used for encoding, logit generation, and decoding.
        vocab: Vocabulary shared across all decoders.
        decoder: Constrained JSON decoder enforcing structural and schema validity.
        function_selector: Component responsible for function name selection.
    """

    def __init__(self, model: Small_LLM_Model, vocab: Vocabulary) -> None:
        """Initialize the generator with a model and vocabulary.

        Args:
            model: LLM instance to use for all generation steps.
            vocab: Vocabulary built from the model's vocabulary JSON file.
        """
        self.model: Small_LLM_Model = model
        self.vocab: Vocabulary = vocab
        self.decoder: ConstrainedJSONDecoder = ConstrainedJSONDecoder(model, vocab)
        self.function_selector: FunctionSelector = FunctionSelector(model, vocab)

    def generate(self, question: str) -> FunctionCall:
        """Translate a natural-language question into a structured function call.

        Args:
            question: The natural-language prompt to process.

        Returns:
            A validated :class:`~src.models.function_call.FunctionCall` instance.
        """
        function = self.function_selector.select_function(question)
        prompt, result = self._build_prompt(question, function)

        print(question)
        print(result, end="")

        result = self._generate_json(prompt, result, function)
        return FunctionCall.model_validate_json(result)

    def _build_prompt(self, question: str, function: FunctionDefinition) -> tuple[str, str]:
        """Build the LLM prompt and the pre-filled JSON prefix.

        Args:
            question: The original natural-language question.
            function: The function definition selected for this question.

        Returns:
            A two-tuple ``(prompt, initial_result)`` where *prompt* is the full
            text fed to the LLM and *initial_result* is the partial JSON string
            that the decoder will continue from.
        """
        prompt = f"""PROMPT: {question}
    ARGUMENTS:
    - {'- \n'.join(function.args_names)}
    FUNCTION CALL: """

        initial_result = (
            '{'
            + f'"prompt": "{question}",'
            + f'"fn_name": "{function.fn_name}",'
            + '"args": {'
        )
        return prompt, initial_result

    def _generate_json(self, prompt: str, result: str, function: FunctionDefinition) -> str:
        """Run constrained greedy decoding to complete the JSON string.

        Continues token-by-token generation from *result* until the JSON FSM
        reaches :attr:`~src.state.JsonState.END` or the iteration limit is hit.

        Args:
            prompt: Full LLM prompt (used as context for logit generation).
            result: Partial JSON string to continue from.
            function: Function definition whose argument names constrain the schema.

        Returns:
            The completed JSON string.
        """
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
