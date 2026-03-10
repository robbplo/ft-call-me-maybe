*This project has been created as part of the 42 curriculum by robbin*

# Call Me Maybe

## Description

`ft-call-me-maybe` is a small function-calling pipeline built around the idea that an LLM should not
answer a request directly, but should instead translate the request into a structured function call.

In this repository, the model receives a natural-language prompt, selects the best matching function
from `data/input/functions_definition.json`, and generates a JSON object with:

- the original prompt
- the selected function name
- the extracted arguments

The main objective of the project is reliability. Instead of hoping that the model "just writes
valid JSON", the code constrains decoding token by token so that invalid structural continuations
are never selected.

## Installation and Execution

### Requirements

- Python 3.12 or later
- `uv`
- enough RAM to load `Qwen/Qwen3-0.6B`
- internet access on the first run if the model is not cached yet

### Install dependencies

```bash
make install
```

or:

```bash
uv sync
```

### Run the project

```bash
make run
```

or:

```bash
uv run python -m src
```

### Run with explicit paths

```bash
uv run python -m src \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Other useful commands

- `make debug` runs the program with `pdb`
- `make lint` runs `flake8` and `mypy`
- `make clean` removes Python cache directories

## Input and Output

### Prompt input

The CLI reads a JSON file containing prompt objects. In this repository the default input file is:

```text
data/input/function_calling_tests.json
```

Its current shape is:

```json
[
  { "prompt": "What is the square root of 16?" },
  { "prompt": "Reverse the string 'hello'" }
]
```

### Function definitions

Function definitions are currently loaded from:

```text
data/input/functions_definition.json
```

This path is fixed in `src/function_selector.py`.

Its shape is:

```json
[
  {
    "fn_name": "fn_add_numbers",
    "args_names": ["a", "b"],
    "args_types": {
      "a": "float",
      "b": "float"
    },
    "return_type": "float"
  }
]
```

### Output

The generated file is written by default to:

```text
data/output/function_calling_results.json
```

Example output:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {
      "a": 2,
      "b": 3
    }
  }
]
```

## Algorithm Explanation

The pipeline is split into two decoding stages.

### 1. Function selection

`src/function_selector.py` asks the model to choose one function name from the known list. It does
not allow arbitrary free-form text. Instead, `ConstrainedStringDecoder` only keeps tokens that
remain a valid prefix of at least one allowed function name.

That means:

- if the valid functions are `fn_add_numbers` and `fn_multiply_numbers`
- and the current prefix is `fn_`
- only tokens that can continue one of those names are allowed

This removes invalid function names from the search space and guarantees that the result is one of
the declared functions.

### 2. Partial JSON prefix

After the function is selected, `src/function_call_generator.py` builds a partial result:

```json
{"prompt": "...", "fn_name": "...", "args": {
```

The decoder does not regenerate fields that are already known. It only needs to complete the `args`
object.

### 3. Constrained JSON decoding

`src/decoding/constrained_json_decoder.py` is the core of the project.

It combines two filters:

1. A structural filter based on a JSON finite-state machine.
2. A schema filter based on the allowed argument keys and argument types for the selected function.

For every generation step:

1. the model returns logits for the next token
2. every token in the vocabulary is decoded to its text form
3. the decoder simulates what would happen if that token were appended
4. tokens that break JSON syntax or schema rules get `-inf`
5. greedy decoding chooses the highest remaining logit

The structural state machine tracks states such as:

- start of object
- expecting a key
- inside a key
- expecting a colon
- expecting a value
- inside a string
- inside a number, fraction, or exponent
- inside a boolean literal prefix
- expecting comma or end of object

The schema layer adds a second guarantee at depth 2, inside `args`:

- only expected keys may appear
- keys are matched character by character
- unknown keys are rejected immediately
- once a key is complete, the decoder remembers which argument the next value belongs to
- values are constrained to the declared primitive type for that argument
- the object cannot close before all required keys have been emitted

### 4. Validation

The final JSON string is validated with Pydantic models from `src/models/`.

- `FunctionDefinition` validates the schema file
- `FunctionCall` validates the generated output

This means the project has two safety nets:

- constrained decoding before a token is chosen
- Pydantic validation after decoding is complete

## Design Decisions

- The pipeline separates function selection from argument generation. This keeps each decoding
  problem smaller and easier to constrain.
- The output prefix is pre-filled for `prompt` and `fn_name`. Only the uncertain part, `args`, is
  generated by the decoder.
- The JSON decoder works at token level but simulates candidates character by character. This
  matters because one tokenizer token may contain multiple characters.
- Argument types from `functions_definition.json` are threaded into the JSON decoder so the schema
  mask can constrain both keys and values.
- Structural masks are precomputed for each JSON state and nesting depth, which reduces repeated
  work during generation.
- Pydantic is used at the boundaries so malformed schema files and malformed outputs fail clearly.
- The implementation stays close to the subject's core idea: the LLM still performs the selection
  and extraction, but constrained decoding prevents invalid output structure.

## Performance Analysis

On the target hardware used for evaluation, a complete run takes about **3 minutes 30 seconds**.

For the bundled sample set in `data/input/function_calling_tests.json`, the program generates a
valid JSON array with 14 results. The strongest part of the system is structural reliability: the
output remains parseable and schema-shaped because invalid continuations are masked out before
selection.

Semantic accuracy is still limited by the small model. Straightforward prompts such as addition,
multiplication, square root, greeting, and simple string reversal work well, but more open-ended
extraction tasks are weaker. In the current sample run, the regex substitution prompt for vowel
replacement produced a semantically questionable argument set, which shows that constrained decoding
guarantees format, not perfect reasoning.

## Challenges Faced

- The tokenizer can emit multi-character tokens, so checking single characters is not enough. The
  decoder must simulate the full token string through the FSM.
- Function selection must be done by the model, not by heuristics. Constraining generation to valid
  function-name prefixes solves this without hardcoding keyword rules.
- JSON validity and schema validity are different problems. A string can be valid JSON while still
  containing the wrong keys, so both checks are needed.
- Type-aware masking has to follow token prefixes across key/value boundaries, because a single
  token can contain the end of a key, the colon, and the start of a typed value.

### Current limitations

- String decoding does not handle escaped quotes.
- Type-aware constraints are centered on primitive values inside the `args` object only.
- The supported primitive argument types are `str`, `int`, `float`, and `bool`.
- Function definitions are loaded from a fixed path instead of a dedicated CLI option.

## Testing Strategy

The repository is currently tested in four layers:

- CLI smoke test with `uv run python -m src --help`
- schema validation by loading `data/input/functions_definition.json` through the Pydantic model
- decoder sanity checks with dummy tokenizers for both `ConstrainedStringDecoder` and `ConstrainedJSONDecoder`
- end-to-end execution on the bundled prompt file, producing a valid `function_calling_results.json`

The most useful checks during development were the decoder-level tests in `test/`, because they make
it easy to verify that invalid prefixes, invalid keys, type-mismatched values, and early object
termination are rejected before the model can choose them.

Run the unit suite with:

```bash
uv run pytest -q test
```

## Example Usage

### Default run

```bash
uv run python -m src
```

### Custom output path

```bash
uv run python -m src \
  --input data/input/function_calling_tests.json \
  --output /tmp/function_calling_results.json
```

### Example result

```json
{
  "prompt": "Greet john",
  "fn_name": "fn_greet",
  "args": {
    "name": "john"
  }
}
```

## Resources

- Python documentation for `argparse`, `json`, and exception handling
- Pydantic documentation for model validation
- Hugging Face documentation for tokenizers and causal language models
- Qwen model card for `Qwen/Qwen3-0.6B`
- RFC 8259 for the JSON grammar

## AI Usage

AI was used as a support tool for documentation drafting, discussing constrained-decoding edge
cases, and reviewing how to explain the architecture clearly. The implementation details,
repository-specific behavior, and output examples were checked directly against the source code and
local runs instead of being copied blindly from generated text.

