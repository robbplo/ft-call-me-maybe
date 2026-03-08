import argparse
import json
import sys
from pathlib import Path
from typing import cast

from src.function_call_generator import FunctionCallGenerator
from src.vocabulary import Vocabulary
from src.llm_sdk import Small_LLM_Model

DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with ``input`` and ``output`` path attributes.
    """
    parser = argparse.ArgumentParser(
        description="Function calling tool using constrained decoding.")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Path to input JSON file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Path to output JSON file")
    return parser.parse_args()


def main() -> None:
    """Entry point: load inputs, run generation, and write results.

    Reads natural-language prompts from the input JSON file, runs the
    function-calling pipeline for each, and writes the resulting
    :class:`~src.models.function_call.FunctionCall` objects to the output file.

    Exits with status code 1 and a message on stderr for any
    unrecoverable error.
    """
    args = parse_args()

    try:
        with open(args.input, "r") as f:
            data = json.load(f)
        questions: list[str] = [q["prompt"] for q in data]
    except FileNotFoundError:
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: malformed input file: {e}", file=sys.stderr)
        sys.exit(1)

    model = Small_LLM_Model()
    try:
        with open(model.get_path_to_vocabulary_json(), "r") as f:
            token_map = cast(dict[str, int], json.load(f))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: failed to load vocabulary: {e}", file=sys.stderr)
        sys.exit(1)

    vocab = Vocabulary(token_map)
    generator = FunctionCallGenerator(model, vocab)

    function_calls = []
    for question in questions:
        try:
            function_calls.append(generator.generate(question))
        except Exception as e:
            msg = f"\nError processing prompt '{question}': {e}"
            print(msg, file=sys.stderr)
            sys.exit(1)

    try:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([fc.model_dump() for fc in function_calls], f, indent=2)
    except OSError as e:
        print(f"Error: failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
