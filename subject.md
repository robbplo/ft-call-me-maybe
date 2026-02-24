# Call Me Maybe — Introduction to Function Calling in LLMs

*Version 1.1 | Made in collaboration with @ldevelle, @pcamaren, @crfernan*

---

## I. Foreword

Humans have always built structures to make information reliable, shareable, and usable — from Roman stone tablets tracking grain shipments, to sailors logging ocean currents in structured tables, to NASA's laminated flowcharts, to barcode scanners, to standardized beekeeper forms. The goal today is to make AI speak the language of computers.

---

## II. AI Instructions

### Context
AI can assist with many tasks during your learning journey. Always approach AI tools with caution and critically assess results — you can never be completely sure your question was well-formed or that the generated content is accurate. Peers are a valuable resource to help avoid mistakes and blind spots.

### Main Message
- Use AI to reduce repetitive or tedious tasks
- Develop prompting skills (coding and non-coding) for your future career
- Learn how AI systems work to anticipate and avoid risks, biases, and ethical issues
- Continue building technical and power skills by working with peers
- Only use AI-generated content you fully understand and can take responsibility for

### Learner Rules
- Explore AI tools and understand how they work so you can use them ethically
- Reflect on your problem before prompting — write clearer, more detailed, more relevant prompts
- Develop the habit of systematically checking, reviewing, questioning, and testing AI-generated content
- Always seek peer review — don't rely solely on your own validation

### Phase Outcomes
- Develop both general-purpose and domain-specific prompting skills
- Boost productivity with effective use of AI tools
- Strengthen computational thinking, problem-solving, adaptability, and collaboration

### Comments and Examples

**Good practice:** Ask AI "How do I test a sorting function?" → Get ideas → Try them out → Review with a peer → Refine together.

**Bad practice:** Ask AI to write a whole function, copy-paste it into your project. During peer-evaluation, you can't explain what it does or why → you fail.

**Good practice:** Use AI to help design a parser → Walk through logic with a peer → Catch bugs and rewrite together, better and fully understood.

**Bad practice:** Let Copilot generate code for a key part of your project. It compiles, but you can't explain how it handles pipes → you fail the evaluation.

---

## III. Introduction

### III.1 What is Function Calling?

LLMs are powerful at understanding and generating human language, but they don't naturally produce
structured, machine-executable output. Function calling bridges this gap by translating natural
language requests into precise function calls with typed arguments.

**Example:**

| Approach | Output |
|---|---|
| User prompt | "What is the sum of 40 and 2?" |
| Traditional LLM | "The sum of 40 and 2 is 42." |
| Function Calling System | `{"function": "add_numbers", "arguments": {"a": 40, "b": 2}}` |

The function calling system provides the tools to solve it — the right function name and the correct
arguments with proper types — rather than answering directly.

### III.2 Why is This Important?

Function calling enables LLMs to:
- **Interact with external systems**: Call APIs, query databases, control devices
- **Execute code**: Perform calculations, data transformations, file operations
- **Chain operations**: Break complex tasks into executable steps
- **Provide structured output**: Generate JSON, XML, or other machine-readable formats
- **Extract structured data from unstructured text**: e.g., given a large book, extract fields such as `{protagonist name, protagonist sex, protagonist age}`

This technology powers modern AI assistants, code generation tools, and autonomous agents.

### III.3 The Challenge

Small language models (like the 0.6B parameter model used here) are notoriously unreliable at
generating structured output — they might produce valid JSON only 30% of the time from prompting
alone. Yet production systems achieve 99%+ reliability with these same small models.

**How?** The answer is **constrained decoding** — a technique that guides the model's output
token-by-token to guarantee valid structure, without relying on prompting alone.

---

## IV. Common Instructions

### IV.1 General Rules
- Written in **Python 3.10 or later**
- Must adhere to the **flake8** coding standard
- Handle exceptions gracefully using `try-except`; prefer context managers for resources
- All resources must be properly managed to prevent leaks
- Code must include **type hints** for parameters, return types, and variables (using `typing` module); all functions must pass **mypy** without errors
- Include **docstrings** following PEP 257 (Google or NumPy style)

### IV.2 Makefile

Must include the following rules:

| Rule | Description |
|---|---|
| `install` | Install project dependencies (pip, uv, pipx, etc.) |
| `run` | Execute the main script |
| `debug` | Run in debug mode using Python's built-in debugger (pdb) |
| `clean` | Remove temporary files/caches (`__pycache__`, `.mypy_cache`) |
| `lint` | Run `flake8 .` and `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` |
| `lint-strict` *(optional)* | Run `flake8 .` and `mypy . --strict` |

### IV.3 Additional Guidelines
- Create test programs using `pytest` or `unittest` (not submitted or graded)
- Include a `.gitignore` file
- Recommended: use virtual environments (`venv` or `conda`)

### IV.3.1 Additional Requirements
- All classes must use **pydantic** for validation
- Allowed packages: `numpy`, `json`
- **Forbidden**: `dspy`, `pytorch`, `huggingface`, `transformers`, `outlines`, or any similar packages
- Required model: **Qwen/Qwen3-0.6B** (default); other models allowed as long as project works with this one
- Function selection must be done **using the LLM** — no heuristics or hardcoded logic
- **Forbidden**: using any private methods or attributes from the `llm_sdk` package
- Create a virtual environment and install `numpy` and `pydantic` using `uv`; copy `llm_sdk` into the same directory as `src`
- Reviewer and moulinette will only run `uv sync`
- All errors must be handled gracefully with clear messages; program must never crash unexpectedly

### IV.3.2 Usage

```bash
uv run python -m src [--input <input_file>] [--output <output_file>]
```

Default: reads from `data/input/`, writes to `data/output/`.

Example:
```bash
uv run python -m src --input data/input/example.json --output data/output/function_calling_results.json
```

---

## V. Mandatory Part

### V.1 Summary

Build a function calling tool that translates natural language prompts into structured function calls. Given "What is the sum of 40 and 2?", the solution must output:
- Function name: `fn_add_numbers`
- Arguments: `{"a": 40, "b": 2}`

Implementation **must use constrained decoding** to guarantee 100% valid JSON output.

### V.2 Input Files

Located in `data/input/`:

**`function_calling_tests.json`** — array of natural language prompts:
```json
[
  "What is the sum of 2 and 3?",
  "Reverse the string 'hello'",
  "Calculate the factorial of 5"
]
```

**`function_definitions.json`** — available functions with name, argument names/types, return type, and description:
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string",
    "parameters": {
      "s": {"type": "string"}
    },
    "returns": {"type": "string"}
  }
]
```

> ⚠️ Input files may contain invalid JSON or be missing entirely — implement proper error handling. Your solution will be tested with **different** prompts and function sets than the examples.

### V.3 LLM Interaction

#### V.3.1 The LLM SDK

The `llm_sdk` package provides a `Small_LLM_Model` wrapper class with the following public methods:

| Method | Signature | Description |
|---|---|---|
| `get_logits_from_input_ids` | `(input_ids: Tensor) -> Tensor` | Returns raw logits from the LLM |
| `get_path_to_vocabulary_json` | `() -> str` | Returns path to vocab JSON (maps input_ids to tokens) |
| `encode` | `(text: str) -> List[int]` | Encodes text to token IDs |
| `decode` *(optional)* | `(token_ids: List[int]) -> str` | Decodes token IDs back to text |

#### V.3.2 The Generation Pipeline

1. **Prompt**: Natural language question
2. **Tokenization**: Text broken into subword tokens (BPE or SentencePiece); leading spaces preserved (e.g., `"Ġthe"`)
3. **Input IDs**: Tokens converted to numerical IDs
4. **LLM Processing**: Neural network processes the IDs
5. **Logits**: Probability scores for each possible next token
6. **Token Selection**: Next token chosen (usually highest score) — constrained decoding applied here

This process repeats token-by-token until the complete response is generated:

```
Prompt -> Tokenization -> Input IDs -> LLM -> Logits -> Next Token Selection
```

#### V.3.3 Understanding Constrained Decoding

At each generation step:
1. Model produces logits for all possible tokens
2. Identify which tokens maintain both valid JSON structure **and** schema compliance
3. Set logits for invalid tokens to **negative infinity**
4. Sample only from remaining valid tokens

This enforces both **syntactic** validity (valid JSON) and **semantic** validity (schema conformance — e.g., a field with predefined allowed values only generates those values).

> ⚠️ Your solution must **NOT** rely on the model spontaneously producing correct JSON from a prompt. Prompting alone is not reliable and is not the skill being tested.

> 💡 Use the vocabulary JSON file to map between tokens and their string representations — this is crucial for determining which tokens are valid at each generation step.

### V.4 Output File Format

Output: `output/function_calling_results.json` — a JSON array where each object contains:

| Key | Type | Description |
|---|---|---|
| `prompt` | string | Original natural-language request |
| `fn_name` | string | Name of the function to call |
| `args` | object | All required arguments with correct types |

**Example output:**
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "fn_name": "fn_reverse_string",
    "args": {"s": "hello"}
  }
]
```

#### V.4.2 Validation Rules
- Must be valid JSON (no trailing commas, no comments)
- Keys and types must match `function_definitions.json` exactly
- No extra keys or prose anywhere in output
- All required arguments must be present
- Argument types must match function definitions (`number`, `string`, `boolean`, etc.)

> ⚠️ Input files may change during peer review — do not hardcode solutions.

### V.5 Performance and Reliability

| Target | Requirement |
|---|---|
| Accuracy | 95%+ correct function selection and argument extraction |
| JSON validity | 100% parseable and schema-compliant |
| Speed | All prompts processed in under 5 minutes |
| Error handling | Graceful handling of malformed inputs, missing files, edge cases |

### V.6 Testing Your Implementation

1. Ensure input files are in `input/`
2. Run: `uv run python -m src`
3. Check that `output/function_calling_results.json` is created
4. Validate JSON structure and content
5. Verify function names and argument types match definitions

> ⚠️ Test edge cases: empty strings, large numbers, special characters, ambiguous prompts, functions with multiple parameters.

---

## VI. Readme Requirements

The `README.md` at the repository root must include:

- **First line** (italicized): *This project has been created as part of the 42 curriculum by \<login1\>[, \<login2\>[, \<login3\>[...]]]*
- **Description**: Project goal and brief overview
- **Instructions**: Compilation, installation, and/or execution information
- **Resources**: Classic references (docs, articles, tutorials) + description of AI usage (which tasks, which parts)

Additionally for this project:
- **Algorithm explanation**: Describe your constrained decoding approach in detail
- **Design decisions**: Explain key choices in your implementation
- **Performance analysis**: Discuss accuracy, speed, and reliability
- **Challenges faced**: Document difficulties and how you solved them
- **Testing strategy**: Describe how you validated your implementation
- **Example usage**: Provide clear examples of running your program

> README must be written in **English**.

---

## VII. Submission and Peer Review

Submit via Git repository. Repository must contain:

| Item | Description |
|---|---|
| `src/` | Implementation directory |
| `pyproject.toml` & `uv.lock` | Dependency management |
| `llm_sdk/` | Copied from provided package |
| `data/input/` | Test files (for demonstration) |
| `README.md` | Comprehensive documentation |

> ⚠️ Do **not** include the `output/` directory — it will be generated during peer review.

During evaluation, a brief **modification of the project** may be requested (minor behavior change, a few lines of code, or an easy feature addition) to verify actual understanding. This should be feasible within a few minutes in any development environment.
