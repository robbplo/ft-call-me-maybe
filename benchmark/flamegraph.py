"""Flamegraph benchmark for the constrained decoding pipeline.

Exercises the hot paths without loading the real LLM:
  - ConstrainedJSONDecoder.__init__  (structural mask precomputation)
  - get_logit_mask                   (schema mask, called per token step)
  - advance_state                    (state copy + FSM simulation per token)

Usage
-----
# cProfile → SVG flamegraph (no extra deps beyond flameprof):
    uv run python -m benchmark.flamegraph

# pyinstrument → interactive HTML flamegraph:
    uv run pyinstrument -r html -o benchmark/profile.html \\
        -m benchmark.flamegraph

# py-spy (external, zero overhead, works on the real pipeline too):
    uv run py-spy record -o benchmark/flamegraph.svg -- \\
        python -m src

Output
------
benchmark/profile.prof   raw cProfile data (open with snakeviz, pstats, …)
benchmark/flamegraph.svg SVG flamegraph rendered by flameprof
"""

import cProfile
import pstats
from pathlib import Path

import flameprof

from src.decoding import ConstrainedJSONDecoder
from src.decoding.state import JsonState, State
from src.vocabulary import Vocabulary

# ---------------------------------------------------------------------------
# Synthetic vocabulary
# ---------------------------------------------------------------------------
# Qwen3-0.6B has ~151 936 tokens.  We reproduce that size so mask-building
# loops run for the same number of iterations as in production.
#
# Token content is chosen to give the FSM realistic work:
#   - single characters covering every ASCII printable (high token-mask hit)
#   - multi-character tokens that mix JSON structure chars with word pieces

VOCAB_SIZE = 151_936


def _build_synthetic_vocab() -> dict[str, int]:
    tokens: dict[str, int] = {}
    idx = 0

    # JSON structure singles
    for ch in '{}[]:,\\"0123456789truefalsnull \t\n\r-+.eE':
        if ch not in tokens:
            tokens[ch] = idx
            idx += 1

    # Multi-char tokens that stress the FSM (JSON-like fragments)
    json_fragments = [
        '{"', '"}', '":', ',"', ': ', ', ', '{"key":', '": "', '": ',
        "true", "false", "null", "0.", ".0", "1.0", "-1", "1e", "1E",
        '"}', '}}', "{}", '[]',
    ]
    for frag in json_fragments:
        if frag not in tokens:
            tokens[frag] = idx
            idx += 1

    # Fill the rest with synthetic word-piece tokens (never valid JSON alone)
    prefix_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    i = 0
    while idx < VOCAB_SIZE:
        length = (i % 6) + 1
        chars = []
        tmp = i
        for _ in range(length):
            chars.append(prefix_chars[tmp % len(prefix_chars)])
            tmp //= len(prefix_chars)
        token = "".join(chars)
        if token not in tokens:
            tokens[token] = idx
            idx += 1
        i += 1

    return tokens


class _SyntheticTokenizer:
    """Minimal tokenizer stub: decode(ids) → concatenated token strings."""

    def __init__(self, reverse: dict[int, str]) -> None:
        self._reverse = reverse

    def decode(self, ids: list[int]) -> str:
        return "".join(self._reverse.get(i, "") for i in ids)


# ---------------------------------------------------------------------------
# States to benchmark (cover the most common decoding positions)
# ---------------------------------------------------------------------------

def _make_states(allowed_keys: list[str], allowed_types: dict) -> list[State]:
    """Return a representative set of States that exercise different paths."""
    return [
        # depth-1: structural-only mask (fast path)
        State(JsonState.EXPECT_KEY, depth=1,
              allowed_keys=allowed_keys, allowed_types=allowed_types),
        State(JsonState.IN_KEY, depth=1,
              allowed_keys=allowed_keys, allowed_types=allowed_types),
        State(JsonState.EXPECT_COLON, depth=1,
              allowed_keys=allowed_keys, allowed_types=allowed_types),
        State(JsonState.EXPECT_VALUE, depth=1,
              allowed_keys=allowed_keys, allowed_types=allowed_types),
        State(JsonState.EXPECT_COMMA_OR_END, depth=1,
              allowed_keys=allowed_keys, allowed_types=allowed_types),
        # depth-2: schema mask (slow path — advance_state called per token)
        State(JsonState.EXPECT_KEY, depth=2,
              allowed_keys=allowed_keys, allowed_types=allowed_types),
        State(JsonState.IN_KEY, depth=2,
              allowed_keys=allowed_keys, allowed_types=allowed_types,
              current_key=""),
        State(JsonState.EXPECT_COLON, depth=2,
              allowed_keys=allowed_keys, allowed_types=allowed_types,
              current_value_key="source_string"),
        State(JsonState.EXPECT_VALUE, depth=2,
              allowed_keys=allowed_keys, allowed_types=allowed_types,
              current_value_key="source_string"),
        State(JsonState.IN_STRING, depth=2,
              allowed_keys=allowed_keys, allowed_types=allowed_types,
              current_value_key="source_string",
              current_value_buffer='"hel'),
        State(JsonState.EXPECT_COMMA_OR_END, depth=2,
              allowed_keys=allowed_keys, allowed_types=allowed_types,
              keys=["source_string"],
              current_value_key=None),
        State(JsonState.EXPECT_VALUE, depth=2,
              allowed_keys=allowed_keys, allowed_types=allowed_types,
              current_value_key="n"),
        State(JsonState.IN_NUMBER, depth=2,
              allowed_keys=allowed_keys, allowed_types=allowed_types,
              current_value_key="n",
              current_value_buffer="42"),
    ]


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

def run_benchmark(decoder: ConstrainedJSONDecoder, states: list[State],
                  iterations: int) -> None:
    """Call get_logit_mask for every state, repeated *iterations* times."""
    for _ in range(iterations):
        for state in states:
            decoder.get_logit_mask(state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path("benchmark")
    out_dir.mkdir(exist_ok=True)

    print("Building synthetic vocabulary …")
    token_map = _build_synthetic_vocab()
    vocab = Vocabulary(token_map)
    reverse = {v: k for k, v in token_map.items()}
    tokenizer = _SyntheticTokenizer(reverse)

    print(f"  vocab size : {vocab.size:,}")

    # Use the most argument-heavy function to stress schema validation
    allowed_keys = ["source_string", "regex", "replacement"]
    allowed_types: dict = {
        "source_string": "str",
        "regex": "str",
        "replacement": "str",
    }
    states = _make_states(allowed_keys, allowed_types)

    # --- profile decoder construction (structural mask precomputation) -------
    print("Profiling ConstrainedJSONDecoder.__init__ …")
    profiler = cProfile.Profile()
    profiler.enable()
    decoder = ConstrainedJSONDecoder(tokenizer, vocab)
    profiler.disable()
    _save(profiler, out_dir, "init")

    # --- profile get_logit_mask (hot path during generation) ----------------
    iterations = 20  # simulates ~20 token generation steps × 13 states
    print(f"Profiling get_logit_mask ({iterations} iterations × "
          f"{len(states)} states) …")
    profiler = cProfile.Profile()
    profiler.enable()
    run_benchmark(decoder, states, iterations)
    profiler.disable()
    _save(profiler, out_dir, "get_logit_mask")

    # --- combined profile (realistic end-to-end decoding loop) --------------
    print("Profiling combined (init + 50 mask calls) …")
    profiler = cProfile.Profile()
    profiler.enable()
    decoder2 = ConstrainedJSONDecoder(tokenizer, vocab)
    run_benchmark(decoder2, states, 50)
    profiler.disable()
    _save(profiler, out_dir, "combined")

    print()
    print("Done. Output files:")
    for svg in sorted(out_dir.glob("*.svg")):
        print(f"  {svg}")
    for prof in sorted(out_dir.glob("*.prof")):
        print(f"  {prof}  (open with: uv run python -m pstats {prof})")
    print()
    print("Top 20 cumulative time (combined profile):")
    ps = pstats.Stats(str(out_dir / "combined.prof"))
    ps.sort_stats("cumulative")
    ps.print_stats(20)


def _save(profiler: cProfile.Profile, out_dir: Path, label: str) -> None:
    prof_path = out_dir / f"{label}.prof"
    svg_path = out_dir / f"{label}.svg"

    profiler.dump_stats(str(prof_path))

    # Render SVG flamegraph via flameprof.render(stats_dict, out, fmt='svg')
    # flameprof expects the raw stats dict from pstats.Stats.stats
    stats = pstats.Stats(profiler)
    with open(svg_path, "w") as f:
        flameprof.render(stats.stats, f)

    print(f"  saved {prof_path} and {svg_path}")


if __name__ == "__main__":
    main()
