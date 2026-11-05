#!/usr/bin/env python3
"""
Turing state-machine text generator driven by a combinatoric mechanism
======================================================================

Run it:   python discrete_lock_turing.py        (no arguments; it asks you)
Settings: edit CONFIG near the bottom of this file.

Every generated token is elected by the combinatoric mechanism
(combinatoric_mechanism.py): the generator's algorithms are split into atoms
(n-gram probability, adiabatic ramp, fragment alignment/consistency, ledger,
Turing-state bonus, harmonization, association affinity, continuity delta,
corpus 4-grams, the state filters, the tau/epsilon gates) and a few hundred
subsets of those atoms each vote for a token at every step.  The votes are the
token distribution.

The I-II-III-O loop, Turing states, ledger and fragments are unchanged:

  I Identity -> II Correspondence -> III Verification -> O Recycling
  M_{t+1} = O(III(II(I(M_t))))        I_{t+1} = O_t

API:  load_corpus, TextEngine, generate_text, build_dataset
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import contextlib
import io
import json
import sys
import tempfile
import threading

import gradio as gr
import pandas as pd
"""
Gradio GUI for the Turing state-machine text generator.

Put this file in the same folder as:
    discrete_lock_turing.py        (the generator you posted)
    combinatoric_mechanism.py      (required by the generator)

Install:  python -m pip install gradio numpy
Run:      python gradio_app.py
"""



sys.path.insert(0, str(Path(__file__).resolve().parent))


_LOCK = threading.Lock()  # stdout capture + shared CONFIG are not thread-safe

SEGMENT_COLUMNS = [
    "state", "end_state", "steps", "text", "natural_rate",
    "novelty", "fallbacks", "relaxations", "closed", "mechanism_agreement",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paths(files: Any) -> List[str]:
    """Normalise gr.File output across Gradio versions."""
    if not files:
        return []
    if not isinstance(files, (list, tuple)):
        files = [files]
    out = []
    for f in files:
        out.append(str(getattr(f, "name", f)))
    return out


def _segments_df(segments: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in segments:
        row = {k: s.get(k) for k in SEGMENT_COLUMNS}
        for k in ("natural_rate", "novelty", "mechanism_agreement"):
            if isinstance(row[k], float):
                row[k] = round(row[k], 3)
        rows.append(row)
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def _gen_options(tokens, n_masks, beam, tau, epsilon, temperature, levels,
                 ledger_weight, keep_learning, initial_state,
                 template_associates) -> Dict[str, Any]:
    return dict(
        tokens=int(tokens),
        n_masks=int(n_masks),
        beam=int(beam),
        tau=float(tau),
        epsilon=float(epsilon),
        temperature=float(temperature),
        levels=int(levels),
        ledger_weight=float(ledger_weight),
        keep_learning=bool(keep_learning),
        initial_state=str(initial_state),
        template_associates=int(template_associates),
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def load_engine(files, paths_text, pasted, order, alpha, warm_limit,
                hf_max, hf_stream):
    sources = _paths(files)
    sources += [p.strip() for p in (paths_text or "").split("|") if p.strip()]

    try:
        with _LOCK:
            dlt.CONFIG["hf_max_samples"] = int(hf_max)
            dlt.CONFIG["hf_streaming"] = bool(hf_stream)

            texts: List[str] = []
            used: List[str] = []
            if sources:
                text, used = dlt.load_corpus(sources)
                texts.append(text)
            if (pasted or "").strip():
                texts.append(pasted.strip())
                used.append("<pasted text>")
            if not texts:
                text, used = dlt.load_corpus(None)  # built-in corpus
                texts.append(text)

            engine = dlt.TextEngine(
                corpus_text="\n\n".join(texts),
                order=int(order),
                alpha=float(alpha),
                warm_limit=int(warm_limit),
            )
            engine.sources = used
    except Exception as exc:  # noqa: BLE001
        return None, f"❌ Could not load corpus: {exc}"

    status = (
        f"✅ Loaded **{len(engine.corpus_tokens):,}** tokens from "
        f"**{len(used)}** source(s) · vocab {len(engine.model.vocab):,} · "
        f"{len(engine.sentences):,} sentences\n\n"
        + "\n".join(f"- `{u}`" for u in used[:20])
        + ("\n- …" if len(used) > 20 else "")
    )
    return engine, status


def run_generate(engine, prompt, seed, trace, *opts):
    if engine is None:
        raise gr.Error("Load a corpus first (Corpus tab).")
    options = _gen_options(*opts)
    prompt = (prompt or "").strip() or dlt.CONFIG["default_prompt"]

    buf = io.StringIO()
    try:
        with _LOCK, contextlib.redirect_stdout(buf):
            result = engine.generate_text(
                prompt, seed=int(seed), trace=bool(trace), **options)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Generation failed: {exc}")

    summary = {
        k: result[k] for k in (
            "prompt", "seed", "n_tokens", "n_segments", "mean_natural_rate",
            "mean_novelty", "mean_agreement", "relaxations", "fallbacks",
            "ledger")
    }
    return (
        result["text"],
        json.dumps(summary, indent=2, default=float),
        _segments_df(result["segments"]),
        buf.getvalue() or "(trace off)",
    )


def run_dataset(engine, custom_prompts, n_prompts, n_seeds, base_seed, fmt,
                include_segments, *opts):
    if engine is None:
        raise gr.Error("Load a corpus first (Corpus tab).")
    options = _gen_options(*opts)

    prompts = [p.strip() for p in (custom_prompts or "").splitlines() if p.strip()]
    if not prompts:
        prompts = engine.auto_prompts(int(n_prompts), int(base_seed))
    seeds = [int(base_seed) + i for i in range(int(n_seeds))]

    out_dir = Path(tempfile.mkdtemp(prefix="dataset_"))
    out_path = out_dir / f"generated_dataset.{fmt}"

    buf = io.StringIO()
    try:
        with _LOCK, contextlib.redirect_stdout(buf):
            rows = dlt.build_dataset(
                engine, prompts, seeds=seeds, out_path=str(out_path),
                include_segments=bool(include_segments), progress=True,
                **options)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Dataset build failed: {exc}")

    preview = pd.DataFrame([
        {
            "id": r["id"], "prompt": r["prompt"], "seed": r["seed"],
            "n_tokens": r["n_tokens"], "n_segments": r["n_segments"],
            "mean_novelty": round(r["mean_novelty"], 3),
            "mean_agreement": (None if r["mean_agreement"] is None
                               else round(r["mean_agreement"], 3)),
            "text": r["text"],
        }
        for r in rows
    ])
    return str(out_path), preview, buf.getvalue()


def run_matrix():
    try:
        names, gram = dlt.function_inner_product_matrix(Path(dlt.__file__))
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Could not build matrix: {exc}")
    out = Path(tempfile.mkdtemp(prefix="matrix_")) / "function_inner_product.csv"
    dlt.save_function_inner_product_matrix(out, names, gram)
    df = pd.DataFrame(gram, columns=names)
    df.insert(0, "function", names)
    return str(out), df, f"{len(names)} × {len(names)} Gram matrix"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    C = dlt.CONFIG
    states = [s for s in dlt.TuringMachine.STATES if s != "HALT"]

    with gr.Blocks(title="Turing Text Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Turing state-machine text generator\n"
            "Combinatoric-mechanism voting · I-II-III-O loop")
        engine_state = gr.State(None)

        # Generation settings are shared between the Generate and Dataset tabs.
        with gr.Accordion("Generation settings", open=False):
            with gr.Row():
                tokens = gr.Slider(10, 2000, value=C["tokens"], step=10,
                                   label="Tokens")
                n_masks = gr.Number(value=C["n_masks"], precision=0,
                                    minimum=1, maximum=32767,
                                    label="Atom subsets voting (n_masks, max 32767)")
                beam = gr.Slider(1, 64, value=C["beam"], step=1, label="Beam")
            with gr.Row():
                tau = gr.Slider(0, 1, value=C["tau"], step=0.01, label="tau")
                epsilon = gr.Slider(0, 1, value=C["epsilon"], step=0.01,
                                    label="epsilon")
                temperature = gr.Slider(0, 2, value=C["temperature"], step=0.01,
                                        label="Temperature")
            with gr.Row():
                levels = gr.Slider(2, 12, value=C["levels"], step=1,
                                   label="Quantization levels")
                ledger_weight = gr.Slider(0, 3, value=C["ledger_weight"],
                                          step=0.05, label="Ledger weight")
                template_associates = gr.Slider(
                    1, 12, value=C["template_associates"], step=1,
                    label="Template associates")
            with gr.Row():
                initial_state = gr.Dropdown(states, value=C["initial_state"],
                                            label="Initial Turing state")
                keep_learning = gr.Checkbox(value=C["keep_learning"],
                                            label="Keep learning (don't lock ledger)")
        gen_inputs = [tokens, n_masks, beam, tau, epsilon, temperature, levels,
                      ledger_weight, keep_learning, initial_state,
                      template_associates]

        with gr.Tabs():
            # ------------------------------------------------------------ Corpus
            with gr.Tab("1 · Corpus"):
                gr.Markdown("Leave everything empty to use the built-in corpus.")
                files = gr.File(label="Upload corpus files "
                                      "(.txt .md .rst .json .jsonl .csv)",
                                file_count="multiple")
                paths_text = gr.Textbox(
                    label="Paths / Hugging Face specs (separate with |)",
                    placeholder="my_folder | hf:wikitext:wikitext-2-raw-v1:train:text")
                pasted = gr.Textbox(label="Or paste text", lines=5)
                with gr.Row():
                    order = gr.Slider(1, 8, value=C["order"], step=1,
                                      label="n-gram order")
                    alpha = gr.Slider(0.01, 2, value=C["alpha"], step=0.01,
                                      label="Smoothing alpha")
                    warm_limit = gr.Number(value=C["warm_limit"], precision=0,
                                           label="Ledger warm-up tokens")
                with gr.Row():
                    hf_max = gr.Number(value=C["hf_max_samples"], precision=0,
                                       label="HF max rows")
                    hf_stream = gr.Checkbox(value=C["hf_streaming"],
                                            label="HF streaming")
                load_btn = gr.Button("Load corpus / train model", variant="primary")
                status = gr.Markdown("_No corpus loaded yet._")
                load_btn.click(
                    load_engine,
                    [files, paths_text, pasted, order, alpha, warm_limit,
                     hf_max, hf_stream],
                    [engine_state, status])

            # ---------------------------------------------------------- Generate
            with gr.Tab("2 · Generate"):
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", value=C["default_prompt"],
                                        scale=4)
                    seed = gr.Number(value=C["seed"], precision=0, label="Seed",
                                     scale=1)
                    trace = gr.Checkbox(value=False, label="Per-token trace",
                                        scale=1)
                gen_btn = gr.Button("Generate", variant="primary")
                out_text = gr.Textbox(label="Generated text", lines=12,
                                      show_copy_button=True)
                with gr.Row():
                    out_summary = gr.Code(label="Run summary", language="json")
                    out_trace = gr.Textbox(label="Trace", lines=12)
                out_segments = gr.Dataframe(label="Segments", wrap=True)
                gen_btn.click(
                    run_generate,
                    [engine_state, prompt, seed, trace, *gen_inputs],
                    [out_text, out_summary, out_segments, out_trace])
                prompt.submit(
                    run_generate,
                    [engine_state, prompt, seed, trace, *gen_inputs],
                    [out_text, out_summary, out_segments, out_trace])

            # ----------------------------------------------------------- Dataset
            with gr.Tab("3 · Dataset"):
                custom_prompts = gr.Textbox(
                    label="Prompts (one per line; leave blank to draw them "
                          "from the corpus)", lines=5)
                with gr.Row():
                    n_prompts = gr.Slider(1, 500, value=C["dataset_prompts"],
                                          step=1, label="Auto prompts")
                    n_seeds = gr.Slider(1, 20, value=C["dataset_seeds"], step=1,
                                        label="Samples per prompt")
                    base_seed = gr.Number(value=C["seed"], precision=0,
                                          label="Base seed")
                with gr.Row():
                    fmt = gr.Radio(["jsonl", "csv"], value="jsonl",
                                   label="Format")
                    include_segments = gr.Checkbox(
                        value=True, label="Include segments (JSONL only)")
                ds_btn = gr.Button("Build dataset", variant="primary")
                ds_file = gr.File(label="Download dataset")
                ds_preview = gr.Dataframe(label="Preview", wrap=True)
                ds_log = gr.Textbox(label="Progress log", lines=6)
                ds_btn.click(
                    run_dataset,
                    [engine_state, custom_prompts, n_prompts, n_seeds,
                     base_seed, fmt, include_segments, *gen_inputs],
                    [ds_file, ds_preview, ds_log])

            # ------------------------------------------------------------ Matrix
            with gr.Tab("4 · Function Gram matrix"):
                gr.Markdown("Identity-aware inner-product matrix of every "
                            "function/method in the generator source.")
                mx_btn = gr.Button("Compute matrix")
                mx_info = gr.Markdown()
                mx_file = gr.File(label="Download CSV")
                mx_df = gr.Dataframe(label="Matrix", wrap=False)
                mx_btn.click(run_matrix, None, [mx_file, mx_df, mx_info])

    return demo

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "NumPy is required. Install it with: python -m pip install numpy"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combinatoric_mechanism import MechanismCombinator, StepContext  # noqa: E402


TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]",
    re.UNICODE,
)

PUNCT = set(".,!?;:")
END = {".", "!", "?"}
SPECIAL = {"", "<s>", "</s>"}
PROTOCOL_SYMBOLS = {"II", "III"}  # stage names of the I-II-III-O protocol

BUILTIN_CORPUS = (
    "The process begins with a simple observation of local flows. "
    "As patterns emerge, structural constraints shape the boundary conditions. "
    "Non-isomorphic transformations map states across mismatched dimensions. "
    "Causal ascent drives the system toward higher informational complexity. "
    "Through adiabatic continuation, the generator balances exploration and selective focus. "
    "A state machine reads a symbol, writes a symbol, moves its head, and changes state. "
    "The tape preserves the history of every emitted token. "
    "Transitions are explicit, inspectable, and reproducible. "
    "Observation becomes description, description becomes relation, and relation becomes abstraction. "
    "Synthesis gathers prior evidence, while reflection consolidates the generated sequence."
)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: Sequence[str]) -> str:
    out = ""
    no_space_before = set(".,!?;:%)]}»”’")
    no_space_after = set("([{«“‘")

    for token in tokens:
        if not out:
            out = token
        elif token in no_space_before or out[-1:] in no_space_after:
            out += token
        else:
            out += " " + token

    return out


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    temp = max(1e-6, float(temperature))
    arr = np.asarray(values, dtype=float) / temp
    arr -= np.max(arr)
    exp = np.exp(np.clip(arr, -60.0, 60.0))
    return exp / max(float(exp.sum()), 1e-12)


def adiabatic_lambda(step: int, total_steps: int) -> float:
    u = min(1.0, max(0.0, step / max(1, total_steps - 1)))
    return 3.0 * u * u - 2.0 * u * u * u


def natural_rate(model: "NGramModel", tokens: Sequence[str]) -> float:
    pairs = list(zip(tokens, tokens[1:]))

    if not pairs or len(model.counts) < 2:
        return 1.0

    seen = sum(
        1
        for previous, current in pairs
        if model.counts[1].get((previous,), {}).get(current)
    )

    return seen / len(pairs)


def novelty(
    ngrams: Dict[int, Set[Tuple[str, ...]]],
    tokens: Sequence[str],
    length: int = 4,
) -> float:
    grams = ngrams.get(length, set())
    n = len(tokens)

    if n < length or not grams:
        return 1.0

    covered: Set[int] = set()

    for i in range(n - length + 1):
        if tuple(tokens[i:i + length]) in grams:
            covered.update(range(i, i + length))

    return 1.0 - len(covered) / n


@dataclass
class Candidate:
    token: str
    base_logp: float
    score: float
    consistency: float
    alignment: float
    probability: float = 0.0
    delta: float = 0.0  # continuity deviation Delta(M_t, M_{t+1})


class NGramModel:
    """Smoothed n-gram language model with longest-context backoff."""

    def __init__(self, order: int = 3, alpha: float = 0.25) -> None:
        self.order = max(1, int(order))
        self.alpha = max(1e-9, float(alpha))
        self.counts: List[Dict[Tuple[str, ...], Counter]] = [
            defaultdict(Counter) for _ in range(self.order)
        ]
        self.vocab: Counter = Counter()
        self.total_tokens = 0

    def train(self, text: str) -> None:
        tokens = [""] * (self.order - 1) + tokenize(text) + [""]

        for i in range(self.order - 1, len(tokens)):
            target = tokens[i]
            self.vocab[target] += 1
            self.total_tokens += 1

            for context_length in range(self.order):
                context = (
                    tuple(tokens[i - context_length:i])
                    if context_length
                    else ()
                )
                self.counts[context_length][context][target] += 1

    def candidates(
        self,
        context: Sequence[str],
        limit: int = 80,
    ) -> List[Tuple[str, float]]:
        vocab_size = max(1, len(self.vocab))
        context = list(context)
        counter: Counter = Counter()

        for context_length in range(
            min(self.order - 1, len(context)),
            -1,
            -1,
        ):
            key = (
                tuple(context[-context_length:])
                if context_length
                else ()
            )
            found = self.counts[context_length].get(key)

            if not found:
                continue

            weight = 1.0 + 0.7 * context_length

            for token, count in found.items():
                counter[token] += weight * count

            if context_length >= 1:
                break

        if not counter:
            counter.update(self.vocab)

        total = sum(counter.values())
        ranked: List[Tuple[str, float]] = []

        for token, count in counter.items():
            probability = (
                count + self.alpha
            ) / (total + self.alpha * vocab_size)
            ranked.append(
                (token, math.log(max(probability, 1e-12)))
            )

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:max(1, limit)]


@dataclass(frozen=True)
class TuringTransition:
    """
    One Turing-machine transition rule.

    state:     control state in which this rule is active.
    read:      "*" matches any symbol class; otherwise END, PUNCT, LONG,
               MEDIUM, SHORT or BLANK.
    write:     literal token to append, or None to append the sampled candidate.
    move:      head movement: -1, 0, or +1.
    min_steps / max_steps: bounds on steps already spent IN `state`.
    """

    state: str
    read: str
    write: Optional[str]
    move: int
    next_state: str
    min_steps: int
    max_steps: int
    max_copy: Optional[int]
    priority: int


class TuringMachine:
    """Turing-style finite control. The generated token history is the tape."""

    STATES = (
        "OBSERVE",
        "DESCRIBE",
        "RELATE",
        "ABSTRACT",
        "SYNTHESIZE",
        "REFLECT",
        "HALT",
    )

    # (state, next_state, min_steps, max_steps, max_copy)
    _CYCLE = (
        ("OBSERVE", "DESCRIBE", 3, 20, None),
        ("DESCRIBE", "RELATE", 4, 20, 3),
        ("RELATE", "ABSTRACT", 5, 22, 3),
        ("ABSTRACT", "SYNTHESIZE", 4, 24, 2),
        ("SYNTHESIZE", "REFLECT", 6, 24, 2),
        ("REFLECT", "HALT", 5, 24, 3),
    )

    def __init__(self, initial_state: str = "OBSERVE") -> None:
        if initial_state not in self.STATES:
            raise ValueError(f"Unknown initial state: {initial_state}")

        self.state = initial_state
        self.head = 0
        self.steps = 0          # total steps since reset
        self.state_steps = 0    # steps spent in the current state

        rules: List[TuringTransition] = []
        for state, nxt, lo, hi, copy in self._CYCLE:
            # A sentence end after `lo` steps advances the cycle.
            rules.append(TuringTransition(
                state, "END", None, 0 if nxt == "HALT" else +1,
                nxt, lo, hi, copy, 20))
            # Anything else keeps the machine in the same state.
            rules.append(TuringTransition(
                state, "*", None, +1, state, 0, hi, copy, 0))
        self.transitions: Tuple[TuringTransition, ...] = tuple(rules)

    @staticmethod
    def classify(token: str) -> str:
        if token in END:
            return "END"
        if token in PUNCT:
            return "PUNCT"
        if not token:
            return "BLANK"
        if len(token) >= 7:
            return "LONG"
        if len(token) <= 3:
            return "SHORT"
        return "MEDIUM"

    def reset(self, initial_state: str = "OBSERVE") -> None:
        if initial_state not in self.STATES:
            raise ValueError(f"Unknown initial state: {initial_state}")

        self.state = initial_state
        self.head = 0
        self.steps = 0
        self.state_steps = 0

    def read(self, tape: Sequence[str]) -> str:
        if not tape:
            return ""

        index = min(max(self.head, 0), len(tape) - 1)
        return tape[index]

    def matching_rules(self, symbol_token: str) -> List[TuringTransition]:
        symbol = self.classify(symbol_token)

        return [
            rule
            for rule in self.transitions
            if rule.state == self.state
            and rule.read in ("*", symbol)
            and rule.min_steps <= self.state_steps <= rule.max_steps
        ]

    def step(self, tape: List[str], emitted_token: str) -> str:
        """Apply the highest-priority matching rule and return the next state."""
        applicable = self.matching_rules(emitted_token)

        self.steps += 1
        self.state_steps += 1

        if not applicable:
            self.state = "HALT"
            return self.state

        rule = max(applicable, key=lambda item: item.priority)

        if rule.write is not None:
            tape.append(rule.write)

        self.head = max(0, min(len(tape) - 1, self.head + rule.move)) if tape else 0

        if rule.next_state != self.state:
            self.state_steps = 0
        self.state = rule.next_state
        return self.state

    def is_halted(self) -> bool:
        return self.state == "HALT"


class TransitionLedger:
    """Counts transitions between quantized fragment states."""

    def __init__(self) -> None:
        self.counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
        self.locked = False

    def record(
        self,
        previous: Optional[Tuple[int, ...]],
        current: Tuple[int, ...],
    ) -> None:
        if self.locked or previous is None:
            return

        self.counts[previous][current] += 1

    def bonus(
        self,
        previous: Optional[Tuple[int, ...]],
        current: Tuple[int, ...],
    ) -> float:
        if previous is None:
            return 0.0

        return math.log1p(
            self.counts.get(previous, {}).get(current, 0)
        )

    def lock(self) -> "TransitionLedger":
        self.locked = True
        return self

    def stats(self) -> Dict[str, object]:
        return {
            "states": len(self.counts),
            "transitions": sum(
                len(targets) for targets in self.counts.values()
            ),
            "locked": self.locked,
        }


class LockedFragments:
    """Projects context, token, and transition features into quantized cells."""

    def __init__(
        self,
        seed: int = 17,
        ledger: Optional[TransitionLedger] = None,
        levels: int = 5,
        ledger_weight: float = 0.3,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.ledger = ledger if ledger is not None else TransitionLedger()
        self.levels = max(2, int(levels))
        self.ledger_weight = float(ledger_weight)
        self.previous: Optional[Tuple[int, ...]] = None

        self.w_context = self.rng.normal(0.0, 0.7, size=(5, 3))
        self.w_word = self.rng.normal(0.0, 0.7, size=(5, 4))
        self.w_transition = self.rng.normal(0.0, 0.7, size=(5, 2))
        self._cache: Dict[str, Tuple[int, ...]] = {}

    @staticmethod
    def _hash_features(token: str, width: int) -> np.ndarray:
        raw = token.lower().encode("utf-8", errors="ignore")
        values = np.zeros(width, dtype=float)

        if not raw:
            return values

        for i, byte in enumerate(raw):
            values[i % width] += ((byte % 31) - 15) / 15.0
            values[(i * 3 + 1) % width] += ((byte % 17) - 8) / 8.0

        values /= max(1, len(raw))
        values[0] += min(len(raw), 20) / 20.0
        values[1] += (
            sum(chr(byte).isalpha() for byte in raw)
            / max(1, len(raw))
        )

        return np.tanh(values)

    def features(
        self,
        context: Sequence[str],
        token: str,
        logp: float,
        rank: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        recent = list(context[-4:])
        context_vector = np.zeros(5, dtype=float)

        if recent:
            for i, word in enumerate(recent):
                context_vector += self._hash_features(word, 5) * (
                    0.65 ** (len(recent) - i - 1)
                )

            context_vector /= max(1, len(recent))

        word_vector = self._hash_features(token, 5)

        transition_vector = np.array(
            [
                np.tanh(logp),
                1.0 / (1.0 + rank),
                1.0 if token[:1].isupper() else 0.0,
                1.0 if token in ".,!?;:" else 0.0,
                min(len(token), 24) / 24.0,
            ],
            dtype=float,
        )

        g1 = np.tanh(context_vector @ self.w_context)
        g2 = np.tanh(word_vector @ self.w_word)
        g3 = np.tanh(transition_vector @ self.w_transition)

        return g1, g2, g3

    @staticmethod
    def quantize(
        vector: np.ndarray,
        levels: int,
    ) -> Tuple[int, ...]:
        indexes = np.clip(
            np.floor(
                (np.asarray(vector, dtype=float) + 1.0)
                / 2.0
                * levels
            ),
            0,
            levels - 1,
        )
        return tuple(int(index) for index in indexes)

    def locked_state(
        self,
        context: Sequence[str],
        token: str,
        logp: float,
        rank: int,
    ) -> Tuple[int, ...]:
        g1, g2, g3 = self.features(context, token, logp, rank)
        state = self.quantize(
            np.concatenate([g1, g2, g3]),
            self.levels,
        )
        self._cache[token] = state
        return state

    def score(
        self,
        context: Sequence[str],
        token: str,
        logp: float,
        rank: int,
        lam: float,
    ) -> Tuple[float, float]:
        g1, g2, g3 = self.features(context, token, logp, rank)

        summaries = np.array(
            [
                np.linalg.norm(g1),
                np.linalg.norm(g2),
                np.linalg.norm(g3),
            ]
        )

        mean = float(np.mean(summaries))
        spread = float(np.std(summaries) / (mean + 1e-9))

        consistency = float(
            np.clip(
                0.55 * spread + 0.45 * (1.0 - mean),
                0.0,
                1.0,
            )
        )

        reference = np.array(
            [
                np.mean(g1),
                np.mean(g2),
                np.mean(g3),
                np.tanh(logp),
            ]
        )
        reference /= max(float(np.linalg.norm(reference)), 1e-12)

        candidate = np.array(
            [
                g1[0],
                g2[0],
                g3[0],
                np.tanh(logp + 0.1 * rank),
            ]
        )
        candidate /= max(float(np.linalg.norm(candidate)), 1e-12)

        alignment = (
            float(np.dot(reference, candidate)) + 1.0
        ) / 2.0

        fragment_score = 1.6 * alignment - 1.15 * consistency

        total = (
            (1.0 - lam)
            * (0.75 * logp + 0.25 * fragment_score)
            + lam
            * (1.25 * logp + 0.85 * fragment_score)
        )

        state = self._cache.get(token)

        if state is not None:
            total += self.ledger_weight * self.ledger.bonus(
                self.previous,
                state,
            )

        return total, consistency

    def commit(self, token: str) -> Tuple[int, ...]:
        state = self._cache[token]
        self.ledger.record(self.previous, state)
        self.previous = state
        self._cache.clear()
        return state


def content_word(token: str) -> bool:
    return len(token) > 3 and token not in PUNCT


def harmonization_score(
    context: Sequence[str],
    candidate: str,
    strength: float = 0.75,
) -> float:
    """
    Small context-sensitive score adjustment for equation-like text.

    If a recent context looks like an equation and contains the letter ``i``
    without the digit ``1``, prefer ``1`` over ``i``. It is a token-selection
    heuristic only; it does not rewrite the existing tape.

    Exemption: when the context contains the I-II-III-O protocol symbols
    (II / III), "I" is the stage name Identity, not a mistyped 1.
    """
    recent = list(context[-16:])
    if not recent:
        return 0.0

    if any(token in PROTOCOL_SYMBOLS for token in recent):
        return 0.0

    equation_markers = {"=", "+", "-", "*", "/", "^", "(", ")"}
    looks_like_equation = any(token in equation_markers for token in recent)
    if not looks_like_equation:
        return 0.0

    has_i = any(token.lower() == "i" for token in recent)
    has_one = "1" in recent
    if not has_i or has_one:
        return 0.0

    if candidate == "1":
        return float(strength)
    if candidate.lower() == "i":
        return -float(strength)
    return 0.0


def build_corpus_ngrams(
    sentences: Sequence[Sequence[str]],
) -> Dict[int, Set[Tuple[str, ...]]]:
    ngrams: Dict[int, Set[Tuple[str, ...]]] = {}

    for length in range(2, 9):
        grams: Set[Tuple[str, ...]] = set()

        for tokens in sentences:
            for i in range(len(tokens) - length + 1):
                grams.add(tuple(tokens[i:i + length]))

        ngrams[length] = grams

    return ngrams


# ---------------------------------------------------------------------------
# Recursive memory state M_t (stage III of the I-II-III-O loop)
# ---------------------------------------------------------------------------

MEMORY_WIDTH = 8
MEMORY_DECAY = 0.7
_PHI_CACHE: Dict[str, np.ndarray] = {}


def _phi(token: str) -> np.ndarray:
    vec = _PHI_CACHE.get(token)
    if vec is None:
        vec = LockedFragments._hash_features(token, MEMORY_WIDTH)
        _PHI_CACHE[token] = vec
    return vec


def _fold(raw: np.ndarray, token: str) -> np.ndarray:
    """Recycle one token into the unnormalised memory: raw' = rho*raw + phi(token)."""
    return MEMORY_DECAY * raw + _phi(token)


def _fold_all(tokens: Sequence[str]) -> np.ndarray:
    raw = np.zeros(MEMORY_WIDTH, dtype=float)
    for token in tokens:
        raw = _fold(raw, token)
    return raw


def _unit_state(raw: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(raw))
    return raw / n if n > 1e-12 else raw


def continuity_delta(m_t: np.ndarray, m_next: np.ndarray) -> float:
    """Delta(M_t, M_{t+1}) in [0, 1]: half the distance between unit states."""
    return float(np.clip(0.5 * np.linalg.norm(m_t - m_next), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Function inner-product (Gram) matrix
# ---------------------------------------------------------------------------

def _identity_ast(node: ast.AST, parameter_map: Optional[Dict[str, str]] = None) -> str:
    """Canonicalize an expression for conservative functional-identity matching.

    Parameter names are alpha-renamed (x/y -> ARG0/ARG1). Commutative operands
    of +, *, &, |, ==, and != are sorted, so x+y and y+x share a fingerprint.
    This is a syntactic identity heuristic, not a theorem prover.
    """
    parameter_map = parameter_map or {}
    if isinstance(node, ast.Name):
        return parameter_map.get(node.id, f"NAME:{node.id}")
    if isinstance(node, ast.Constant):
        return f"CONST:{type(node.value).__name__}:{node.value!r}"
    if isinstance(node, ast.Attribute):
        return f"ATTR({_identity_ast(node.value, parameter_map)}.{node.attr})"
    if isinstance(node, ast.Call):
        fn = _identity_ast(node.func, parameter_map)
        args = [_identity_ast(arg, parameter_map) for arg in node.args]
        kwargs = sorted((kw.arg or "**", _identity_ast(kw.value, parameter_map)) for kw in node.keywords)
        return f"CALL({fn};{','.join(args)};{kwargs!r})"
    if isinstance(node, ast.BinOp):
        left = _identity_ast(node.left, parameter_map)
        right = _identity_ast(node.right, parameter_map)
        op = type(node.op).__name__
        if op in {"Add", "Mult", "BitAnd", "BitOr"}:
            left, right = sorted((left, right))
        return f"BIN:{op}({left},{right})"
    if isinstance(node, ast.BoolOp):
        op = type(node.op).__name__
        values = [_identity_ast(v, parameter_map) for v in node.values]
        if op in {"And", "Or"}:
            values.sort()
        return f"BOOL:{op}({','.join(values)})"
    if isinstance(node, ast.Compare):
        return f"CMP({_identity_ast(node.left, parameter_map)};{','.join(type(op).__name__ for op in node.ops)};{','.join(_identity_ast(v, parameter_map) for v in node.comparators)})"
    if isinstance(node, ast.UnaryOp):
        return f"UNARY:{type(node.op).__name__}({_identity_ast(node.operand, parameter_map)})"
    if isinstance(node, ast.IfExp):
        return f"IF({_identity_ast(node.test, parameter_map)}?{_identity_ast(node.body, parameter_map)}:{_identity_ast(node.orelse, parameter_map)})"
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return f"{type(node).__name__}({','.join(_identity_ast(v, parameter_map) for v in node.elts)})"
    if isinstance(node, ast.Subscript):
        return f"SUB({_identity_ast(node.value, parameter_map)}[{_identity_ast(node.slice, parameter_map)}])"
    if isinstance(node, ast.Slice):
        return f"SLICE({_identity_ast(node.lower, parameter_map) if node.lower else '_'}:{_identity_ast(node.upper, parameter_map) if node.upper else '_'}:{_identity_ast(node.step, parameter_map) if node.step else '_'})"
    # Fallback retains the node type and recursively represented children.
    return f"{type(node).__name__}({','.join(_identity_ast(c, parameter_map) for c in ast.iter_child_nodes(node))})"


def _return_identity_features(fn: ast.AST) -> List[str]:
    """Extract canonical return-expression identities from a function."""
    args = getattr(fn, "args", None)
    arg_names: List[str] = []
    if args is not None:
        arg_names = [a.arg for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)]
        if args.vararg:
            arg_names.append(args.vararg.arg)
        if args.kwarg:
            arg_names.append(args.kwarg.arg)
    parameter_map = {name: f"ARG{i}" for i, name in enumerate(arg_names)}
    identities: List[str] = []
    stack = list(reversed(getattr(fn, "body", [])))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Return) and node.value is not None:
            identities.append(_identity_ast(node.value, parameter_map))
        # Do not walk into nested function definitions.
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return identities


def _function_nodes_without_nested(function_node: ast.AST) -> List[ast.AST]:
    """Return a function's AST nodes without counting nested function bodies."""
    nodes: List[ast.AST] = []
    stack = list(ast.iter_child_nodes(function_node))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        nodes.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return nodes


def function_inner_product_matrix(source_path: Optional[Path] = None) -> Tuple[List[str], np.ndarray]:
    """Build an identity-aware Gram matrix for every function/method in source.

    Each function vector combines AST structure, normalized function-call names,
    and canonical return-expression identities. G = X @ X.T measures shared
    features and recognized syntactic identities; it does not prove arbitrary
    functions equal.
    """
    path = source_path or Path(__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    definitions: List[Tuple[str, ast.AST]] = []

    def visit(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                definitions.append((qualified, child))
                visit(child, qualified)
            elif isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                visit(child, qualified)
            else:
                visit(child, prefix)

    visit(tree)
    if not definitions:
        return [], np.zeros((0, 0), dtype=float)

    feature_sets: List[Dict[str, float]] = []
    for _, fn in definitions:
        features: Dict[str, float] = {}
        for node in _function_nodes_without_nested(fn):
            key = f"AST:{type(node).__name__}"
            features[key] = features.get(key, 0.0) + 1.0
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                else:
                    call_name = type(node.func).__name__
                key = f"CALL:{call_name}"
                features[key] = features.get(key, 0.0) + 1.0
        for identity in _return_identity_features(fn):
            key = f"IDENTITY:{identity}"
            features[key] = features.get(key, 0.0) + 2.0
        feature_sets.append(features)

    vocabulary = sorted({key for features in feature_sets for key in features})
    index = {key: i for i, key in enumerate(vocabulary)}
    vectors = np.zeros((len(definitions), len(vocabulary)), dtype=float)
    for row, features in enumerate(feature_sets):
        for key, value in features.items():
            vectors[row, index[key]] = value

    return [name for name, _ in definitions], vectors @ vectors.T


def save_function_inner_product_matrix(path: Path, names: Sequence[str], matrix: np.ndarray) -> None:
    """Save the labelled function Gram matrix as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["function", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *[f"{value:.6g}" for value in row]])


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    model: NGramModel,
    prompt: str,
    *,
    token_count: int = 500,
    beam_size: int = 12,
    tau: float = 0.95,
    temperature: float = 0.85,
    seed: int = 7,
    show_trace: bool = False,
    levels: int = 5,
    ledger_weight: float = 0.3,
    warm_tokens: Optional[Sequence[str]] = None,
    lock_ledger: bool = True,
    initial_state: str = "OBSERVE",
    template_associates: int = 4,
    epsilon: float = 0.6,
    corpus_ngrams: Optional[Dict[int, Set[Tuple[str, ...]]]] = None,
    combinator: Optional[MechanismCombinator] = None,
    base_associations: Optional[Dict[str, Counter]] = None,
) -> Tuple[str, List[Dict[str, object]], TransitionLedger]:
    """
    Generate text from an association template, continued by a Turing machine
    through the I-II-III-O loop. Each token is elected by the combinatoric
    mechanism (a vote over sampled subsets of the algorithm atoms).
    """
    return _generate(
        model, prompt, token_count, beam_size, tau, temperature, seed,
        show_trace, levels, ledger_weight, warm_tokens, lock_ledger,
        initial_state, template_associates, epsilon, corpus_ngrams,
        combinator, base_associations,
    )


def build_associations(model: NGramModel) -> Dict[str, Counter]:
    """Symmetric association table from corpus bigrams (no punctuation/padding)."""
    associations: Dict[str, Counter] = defaultdict(Counter)

    for context, targets in model.counts[1].items():
        if not context:
            continue

        source = context[0]

        for target, count in targets.items():
            if (
                source and target
                and source not in SPECIAL and target not in SPECIAL
                and source not in PUNCT and target not in PUNCT
            ):
                associations[source][target] += count
                associations[target][source] += count

    return associations


def _generate(
    model, prompt, token_count, beam_size, tau, temperature, seed,
    show_trace, levels, ledger_weight, warm_tokens, lock_ledger,
    initial_state, template_associates, epsilon, corpus_ngrams,
    combinator, base_associations,
):
    rng = random.Random(seed)
    ledger = TransitionLedger()
    fragments = LockedFragments(seed, ledger, levels, ledger_weight)

    # -------------------------------------------------------------
    # Prompt tools.
    # -------------------------------------------------------------
    prompt_tokens = tokenize(prompt) or ["The"]
    prompt_content = [t for t in prompt_tokens if content_word(t)]

    # -------------------------------------------------------------
    # Association table (shared, read-only).
    # -------------------------------------------------------------
    associations = (
        base_associations if base_associations is not None
        else build_associations(model)
    )

    def top_associates(word: str, k: int) -> List[str]:
        for variant in (word, word.lower(), word.capitalize()):
            counts = associations.get(variant)
            if counts:
                return [t for t, _ in counts.most_common(k) if t not in SPECIAL]
        return []

    # -------------------------------------------------------------
    # Ad-hoc template:  A process is <ASSOC> and <ASSOC>
    # -------------------------------------------------------------
    tape: List[str] = list(prompt_tokens)
    filled_slots = 0

    if prompt_content:
        slots: List[str] = []
        for word in prompt_content[:2]:
            choices = top_associates(word, max(1, template_associates))
            if choices:
                slots.append(rng.choice(choices))

        if slots:
            tape.append("is")
            for index, filled in enumerate(slots):
                if index:
                    tape.append("and")
                tape.append(filled)
                filled_slots += 1

    template_text = detokenize(tape)

    if show_trace:
        print(f"[template] {template_text}  ({filled_slots} slot(s) filled)")

    # -------------------------------------------------------------
    # Optional ledger warm-up.
    # -------------------------------------------------------------
    if warm_tokens:
        previous: Optional[Tuple[int, ...]] = None
        context = [""] * (model.order - 1)

        for token in list(warm_tokens)[:20000]:
            ranked = model.candidates(context, limit=40)

            rank, logp = next(
                (
                    (rank, value)
                    for rank, (candidate, value) in enumerate(ranked)
                    if candidate == token
                ),
                (40, -8.0),
            )

            state = fragments.locked_state(context, token, logp, rank)
            ledger.record(previous, state)
            previous = state
            context.append(token)

        fragments._cache.clear()

        if lock_ledger:
            ledger.lock()

    # -------------------------------------------------------------
    # Turing-controlled continuation: the I-II-III-O loop.
    # -------------------------------------------------------------
    machine = TuringMachine(initial_state)

    segments: List[Dict[str, object]] = []
    produced = len(tape)
    raw = _fold_all(tape)
    total_relaxations = 0
    n_masks = len(combinator.masks) if combinator is not None else 1

    while produced < token_count:
        emitted: List[str] = []
        fallbacks = 0
        relaxations = 0
        closed = False
        start_step = produced
        segment_state = machine.state
        agreement: List[float] = []

        for local in range(max(1, token_count - produced)):
            lam = adiabatic_lambda(start_step + local, max(1, token_count))
            state = machine.state
            m_t = _unit_state(raw)

            # ---- Stage I: Identity (recognise state, propose candidates) --
            ranked = [
                (token, logp)
                for token, logp in model.candidates(
                    tape, limit=max(40, beam_size * 8)
                )
                if token not in SPECIAL
            ]

            if machine.state_steps < 3:
                nonterminal = [(t, lp) for t, lp in ranked if t not in END]
                if nonterminal:
                    ranked = nonterminal

            # State-specific hard constraints.
            if state == "DESCRIBE":
                recent = {t for t in tape[-12:] if content_word(t)}
                filtered = [(t, lp) for t, lp in ranked if t not in recent]
                if filtered:
                    ranked = filtered

            elif state == "ABSTRACT":
                filtered = [
                    (t, lp) for t, lp in ranked
                    if t in PUNCT or len(t) <= 3 or len(t) >= 7
                ]
                if filtered:
                    ranked = filtered

            elif state == "RELATE":
                local_tail = set(tape[-8:])
                filtered = [(t, lp) for t, lp in ranked if t not in local_tail]
                if filtered:
                    ranked = filtered

            if not ranked:
                fallbacks += 1
                ranked = [(".", -8.0)]

            # ---- Stage II: Correspondence (context <-> candidate) ---------
            scored: List[Candidate] = []

            for rank, (token, logp) in enumerate(ranked):
                fragments.locked_state(tape, token, logp, rank)

                score, consistency = fragments.score(
                    tape, token, logp, rank, lam
                )
                score += harmonization_score(tape, token, strength=0.75)

                if state == "ABSTRACT" and len(token) >= 7 and token not in PUNCT:
                    score += 1.0
                elif state == "DESCRIBE" and content_word(token):
                    score += 0.8
                elif state == "RELATE" and content_word(token):
                    score += 0.4
                elif state == "REFLECT":
                    score += 0.2

                alignment = float(
                    np.clip((score - logp) / 2.5 + 0.5, 0.0, 1.0)
                )

                if consistency <= tau:
                    delta = continuity_delta(
                        m_t, _unit_state(_fold(raw, token))
                    )
                    scored.append(
                        Candidate(token, logp, score, consistency,
                                  alignment, delta=delta)
                    )

            if not scored:
                token, logp = ranked[0]
                fragments.locked_state(tape, token, logp, 0)
                score, consistency = fragments.score(
                    tape, token, logp, 0, lam
                )
                delta = continuity_delta(m_t, _unit_state(_fold(raw, token)))
                scored = [
                    Candidate(token, logp, score, consistency, 0.5,
                              delta=delta)
                ]

            # ---- Combinatoric mechanism: every atom subset votes -----------
            combo_votes: Counter = Counter()

            if combinator is not None:
                ctx = StepContext(
                    tape=tape, ranked=ranked, state=state, lam=lam,
                    fragments=fragments,
                    delta_fn=lambda tok, _m=m_t, _r=raw: continuity_delta(
                        _m, _unit_state(_fold(_r, tok))),
                    harm_fn=lambda tok: harmonization_score(tape, tok, 0.75),
                    associations=associations, anchors=prompt_content,
                    corpus_ngrams=corpus_ngrams, tau=tau, epsilon=epsilon,
                )
                combo_votes, _winners = combinator.vote(ctx)
                if combo_votes:
                    agreement.append(max(combo_votes.values()) / n_masks)

            # ---- Stage III: Verification (Delta <= epsilon) ---------------
            eps_eff = max(1e-6, epsilon)
            verified = [c for c in scored if c.delta <= eps_eff]
            while not verified and eps_eff < 1.0:
                eps_eff = min(1.0, eps_eff * 1.5)
                relaxations += 1
                verified = [c for c in scored if c.delta <= eps_eff]
            if not verified:
                verified = [min(scored, key=lambda c: c.delta)]

            # The vote is the distribution: tokens elected by more mechanisms
            # are proportionally more likely. The classic score only breaks ties.
            if combo_votes:
                voted = [c for c in verified if combo_votes.get(c.token, 0) > 0]
                if voted:
                    verified = voted
                for c in verified:
                    c.score = (math.log1p(combo_votes.get(c.token, 0))
                               + 0.01 * c.score)

            verified.sort(key=lambda c: c.score, reverse=True)
            beam = verified[:max(1, beam_size)]
            probabilities = softmax([c.score for c in beam], temperature)

            for candidate, probability in zip(beam, probabilities):
                candidate.probability = float(probability)

            if temperature <= 0.01:
                chosen = beam[0]
            else:
                chosen = rng.choices(beam, weights=probabilities, k=1)[0]

            # ---- Stage O: Recycling (I_{t+1} = O_t) ------------------------
            fragments.commit(chosen.token)
            tape.append(chosen.token)
            emitted.append(chosen.token)
            produced += 1
            raw = _fold(raw, chosen.token)

            if show_trace:
                print(
                    f"[{start_step + local:03d}] "
                    f"{state:<10} "
                    f"{chosen.token!r:<16} "
                    f"p={math.exp(chosen.base_logp):.2f} "
                    f"score={chosen.score:+.3f} "
                    f"D={chosen.consistency:.3f} "
                    f"delta={chosen.delta:.3f} "
                    f"votes={combo_votes.get(chosen.token, 0)}/{n_masks}"
                )

            next_state = machine.step(tape, chosen.token)

            if next_state == "HALT":
                closed = chosen.token in END
                break

        # Close an unfinished segment (tape edited too, so they agree).
        if emitted and emitted[-1] not in END:
            if emitted[-1] in PUNCT:
                emitted[-1] = "."
                tape[-1] = "."
            else:
                emitted.append(".")
                tape.append(".")
                produced += 1
            closed = True
            raw = _fold_all(tape)

        total_relaxations += relaxations

        segments.append(
            {
                "template": template_text,
                "state": segment_state,
                "end_state": machine.state,
                "steps": machine.steps,
                "tokens": emitted,
                "text": detokenize(emitted),
                "natural_rate": natural_rate(model, emitted),
                "novelty": (
                    novelty(corpus_ngrams, emitted)
                    if corpus_ngrams is not None
                    else 1.0
                ),
                "fallbacks": fallbacks,
                "relaxations": relaxations,
                "closed": closed,
                "mechanism_agreement": (
                    float(np.mean(agreement)) if agreement else None
                ),
            }
        )

        if machine.is_halted():
            machine.reset("OBSERVE")

    if show_trace:
        print(f"[continuity] epsilon relaxations: {total_relaxations}")

    return detokenize(tape), segments, ledger


# ---------------------------------------------------------------------------
# Dataset layer: corpus loading -> engine -> generated-sample dataset
# ---------------------------------------------------------------------------

TEXT_SUFFIXES = {".txt", ".md", ".text", ".rst"}
TABLE_SUFFIXES = {".json", ".jsonl", ".csv"}
TEXT_FIELDS = ("text", "content", "body", "sentence", "document")


def _texts_from_json(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for field in TEXT_FIELDS:
            if isinstance(obj.get(field), str):
                yield obj[field]
                return
        for value in obj.values():
            yield from _texts_from_json(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _texts_from_json(item)


def _read_one(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return [path.read_text(encoding="utf-8-sig")]
    if suffix == ".jsonl":
        out: List[str] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line:
                out.extend(_texts_from_json(json.loads(line)))
        return out
    if suffix == ".json":
        return list(_texts_from_json(json.loads(path.read_text(encoding="utf-8-sig"))))
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            return []
        header = [h.strip().lower() for h in rows[0]]
        col = next((header.index(f) for f in TEXT_FIELDS if f in header), None)
        body = rows[1:] if col is not None else rows
        return [r[col if col is not None else 0] for r in body
                if r and len(r) > (col or 0)]
    raise ValueError(f"Unsupported corpus file type: {path}")


HF_PREFIX = "hf:"


def _hf_rows(spec: str) -> List[str]:
    """Read text from a Hugging Face dataset.

    spec:  hf:<name>[:<config>[:<split>[:<field>]]]   (empty parts allowed)
      hf:imdb
      hf:wikitext:wikitext-2-raw-v1:train:text
      hf:roneneldan/TinyStories::train
      hf:/path/to/dataset_saved_with_save_to_disk

    At most CONFIG["hf_max_samples"] rows are read, streamed when
    CONFIG["hf_streaming"] is true, so large datasets are not downloaded whole.
    The field defaults to the first text-like (else first string) column.
    """
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "Hugging Face support needs the `datasets` package: "
            "python -m pip install datasets") from exc

    parts = spec[len(HF_PREFIX):].split(":")
    name = parts[0].strip()
    config = (parts[1].strip() or None) if len(parts) > 1 else None
    split = (parts[2].strip() or "train") if len(parts) > 2 else "train"
    field = (parts[3].strip() or None) if len(parts) > 3 else None
    limit = int(CONFIG["hf_max_samples"])

    if not name:
        raise ValueError("Empty Hugging Face dataset name (use hf:<name>).")

    if Path(name).exists():                       # dataset saved to disk
        dataset = load_from_disk(name)
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split] if split in dataset else next(iter(dataset.values()))
    else:
        dataset = load_dataset(
            name, config, split=split, streaming=bool(CONFIG["hf_streaming"]))

    def pick_field(row: Dict[str, Any]) -> str:
        if field:
            if field not in row:
                raise ValueError(
                    f"Field {field!r} not in {name}; columns: {list(row)}")
            return field
        for candidate in TEXT_FIELDS:
            if isinstance(row.get(candidate), (str, list)):
                return candidate
        for key, value in row.items():
            if isinstance(value, str):
                return key
        raise ValueError(f"No text column in {name}; columns: {list(row)}")

    texts: List[str] = []
    column: Optional[str] = None
    for row in dataset:
        if column is None:
            column = pick_field(row)
        value = row.get(column)
        if isinstance(value, list):
            value = " ".join(v for v in value if isinstance(v, str))
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
        if len(texts) >= limit:
            break

    if not texts:
        raise ValueError(f"No text rows read from Hugging Face dataset {name!r}.")
    return texts


def load_corpus(sources: Optional[Sequence[str]] = None) -> Tuple[str, List[str]]:
    """Read files/directories/Hugging Face datasets into one training text.
    None -> built-in corpus.

    Sources starting with "hf:" are Hugging Face datasets (see _hf_rows).
    Accepts .txt/.md/.rst, .json/.jsonl (a text/content/body field, or any
    strings) and .csv (a text-like column, else the first column).
    """
    if not sources:
        return BUILTIN_CORPUS, ["<builtin>"]

    texts: List[str] = []
    used: List[str] = []
    for source in sources:
        if str(source).startswith(HF_PREFIX):
            chunk = _hf_rows(str(source))
            texts.extend(chunk)
            used.append(str(source))
            continue
        path = Path(source)
        if path.is_dir():
            files = sorted(p for p in path.rglob("*")
                           if p.suffix.lower() in TEXT_SUFFIXES | TABLE_SUFFIXES)
        elif path.exists():
            files = [path]
        else:
            raise FileNotFoundError(f"Corpus source not found: {source}")
        for file in files:
            try:
                chunk = [t.strip() for t in _read_one(file) if t and t.strip()]
            except UnicodeDecodeError as exc:
                raise ValueError(f"Could not decode {file} as UTF-8: {exc}") from exc
            if chunk:
                texts.extend(chunk)
                used.append(str(file))

    if not texts:
        raise ValueError("No usable text found in the corpus sources.")
    return "\n\n".join(texts), used


# ---------------------------------------------------------------------------
# Settings: edit here (no command-line arguments)
# ---------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    "order": 5,                 # n-gram order
    "alpha": 0.25,              # n-gram smoothing
    "warm_limit": 20000,        # corpus tokens used to warm the ledger
    "tokens": 200,              # tokens per generation
    "seed": 7,
    "n_masks": 512,             # atom subsets voting at every step (max 32,767)
    "beam": 12,
    "tau": 0.95,
    "epsilon": 0.6,
    "temperature": 0.85,
    "levels": 5,
    "ledger_weight": 0.9,
    "keep_learning": False,
    "initial_state": "OBSERVE",
    "template_associates": 4,
    "trace": False,
    "default_prompt": "A process",
    "dataset_out": "generated_dataset.jsonl",   # .jsonl or .csv
    "dataset_prompts": 25,      # prompts drawn from the corpus
    "dataset_seeds": 2,         # samples per prompt
    "hf_max_samples": 2000,     # rows read from a Hugging Face dataset
    "hf_streaming": True,       # stream instead of downloading it whole
}

GEN_KEYS = ("tokens", "n_masks", "beam", "tau", "epsilon", "temperature",
            "levels", "ledger_weight", "keep_learning", "initial_state",
            "template_associates")


class TextEngine:
    """Trained model + corpus statistics, reusable across many generations."""

    def __init__(
        self,
        corpus: Optional[Sequence[str]] = None,
        *,
        corpus_text: Optional[str] = None,
        order: int = 5,
        alpha: float = 0.25,
        warm_limit: int = 20000,
    ) -> None:
        if corpus_text is not None:
            self.corpus_text, self.sources = corpus_text, ["<text>"]
        else:
            self.corpus_text, self.sources = load_corpus(corpus)

        self.corpus_tokens = tokenize(self.corpus_text)
        if not self.corpus_tokens:
            raise ValueError("Corpus contains no recognizable tokens.")

        self.model = NGramModel(order=order, alpha=alpha)
        self.model.train(self.corpus_text)
        self.associations = build_associations(self.model)
        self.warm_tokens = self.corpus_tokens[:max(0, warm_limit)] or None

        self.sentences = [
            tokenize(s)
            for s in re.split(r"(?<=[.!?])\s+", self.corpus_text.strip())
            if s.strip()
        ]
        self.corpus_ngrams = build_corpus_ngrams(self.sentences)

    def generate_text(
        self,
        prompt: str = "A process",
        *,
        tokens: int = 200,
        seed: int = 7,
        trace: bool = False,
        n_masks: int = 512,
        beam: int = 12,
        tau: float = 0.95,
        epsilon: float = 0.6,
        temperature: float = 0.85,
        levels: int = 5,
        ledger_weight: float = 0.9,
        keep_learning: bool = False,
        initial_state: str = "OBSERVE",
        template_associates: int = 4,
    ) -> Dict[str, Any]:
        combinator = MechanismCombinator(n_masks, seed=seed)
        text, segments, ledger = generate(
            self.model,
            prompt,
            token_count=max(1, tokens),
            beam_size=max(1, beam),
            tau=float(np.clip(tau, 0.0, 1.0)),
            temperature=max(0.0, temperature),
            seed=seed,
            show_trace=trace,
            levels=levels,
            ledger_weight=ledger_weight,
            warm_tokens=self.warm_tokens,
            lock_ledger=not keep_learning,
            initial_state=initial_state,
            template_associates=max(1, template_associates),
            epsilon=float(np.clip(epsilon, 0.0, 1.0)),
            corpus_ngrams=self.corpus_ngrams,
            combinator=combinator,
            base_associations=self.associations,
        )
        agree = [s["mechanism_agreement"] for s in segments
                 if s["mechanism_agreement"] is not None]
        return {
            "prompt": prompt,
            "seed": seed,
            "text": text,
            "segments": segments,
            "n_tokens": len(tokenize(text)),
            "n_segments": len(segments),
            "mean_natural_rate": float(np.mean([s["natural_rate"] for s in segments])),
            "mean_novelty": float(np.mean([s["novelty"] for s in segments])),
            "mean_agreement": float(np.mean(agree)) if agree else None,
            "relaxations": int(sum(s["relaxations"] for s in segments)),
            "fallbacks": int(sum(s["fallbacks"] for s in segments)),
            "ledger": ledger.stats(),
        }

    def auto_prompts(self, n: int, seed: int = 7, max_words: int = 3) -> List[str]:
        """Prompts taken from the starts of randomly chosen corpus sentences."""
        rng = random.Random(seed)
        pool = [s for s in self.sentences if any(content_word(t) for t in s)]
        rng.shuffle(pool)
        out: List[str] = []
        for sentence in pool:
            words = [t for t in sentence if t not in PUNCT][:rng.randint(1, max_words)]
            prompt = " ".join(words)
            if prompt and prompt not in out:
                out.append(prompt)
            if len(out) >= n:
                break
        return out or ["A process"]


_ENGINES: Dict[Tuple, TextEngine] = {}


def generate_text(
    prompt: str = "A process",
    corpus: Optional[Sequence[str]] = None,
    *,
    engine: Optional[TextEngine] = None,
    **options: Any,
) -> Dict[str, Any]:
    """One-call API. Engines are cached per corpus."""
    if engine is None:
        key = tuple(corpus or ())
        engine = _ENGINES.get(key)
        if engine is None:
            engine = _ENGINES[key] = TextEngine(
                corpus, order=CONFIG["order"], alpha=CONFIG["alpha"],
                warm_limit=CONFIG["warm_limit"])
    return engine.generate_text(prompt, **options)


CSV_COLUMNS = (
    "id", "prompt", "seed", "text", "n_tokens", "n_segments",
    "mean_natural_rate", "mean_novelty", "mean_agreement",
    "relaxations", "fallbacks",
)


def build_dataset(
    engine: TextEngine,
    prompts: Sequence[str],
    *,
    seeds: Sequence[int] = (7,),
    out_path: Optional[str] = None,
    include_segments: bool = True,
    progress: bool = True,
    **options: Any,
) -> List[Dict[str, Any]]:
    """Generate prompts x seeds samples; optionally write JSONL or CSV."""
    rows: List[Dict[str, Any]] = []
    total = len(prompts) * len(seeds)

    for prompt in prompts:
        for seed in seeds:
            result = engine.generate_text(prompt, seed=seed, **options)
            row = {"id": len(rows), **result}
            row.pop("ledger", None)
            if not include_segments:
                row.pop("segments", None)
            rows.append(row)
            if progress:
                print(f"[dataset] {len(rows)}/{total} prompt={prompt!r} seed={seed}")

    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".csv":
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        else:
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if progress:
            print(f"[dataset] wrote {len(rows)} rows to {path}")

    return rows


# ---------------------------------------------------------------------------
# Interactive main (no arguments)
# ---------------------------------------------------------------------------

HELP = (
    "Type a prompt to generate.  Commands:  :dataset  build a dataset with "
    "CONFIG settings,  :matrix  save the function Gram matrix,  :trace  toggle "
    "per-token trace,  :quit"
)



if __name__ == "__main__":
    build_ui().queue().launch()

