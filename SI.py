#!/usr/bin/env python3
"""
Turing state-machine text generator + combinatoric mechanism + CLI
=========================================================================

Everything lives in this one file.

Run it:     python turing_generator_optical_preprocessed.py --help
Install:    python -m pip install numpy pillow  (+ datasets for hf: corpora)
Settings:   edit CONFIG below, or use the command-line options.

Paper-derived pre-dataset image path:
  image -> grayscale -> resize to (rows*16, cols*16) -> parallel 16x16
  mean pooling -> confidence-gated binary cell / erasure tokens -> corpus.
  Text sources remain unchanged. Pillow is required only for image sources.

Every generated token is elected by the combinatoric mechanism: the generator's
algorithms are split into 15 atoms (n-gram probability, adiabatic ramp,
fragment alignment, fragment consistency, ledger, Turing-state bonus,
harmonization, association affinity, continuity delta, corpus 4-grams, state
filter, tau gate, epsilon gate, rank prior, repetition penalty) and a few
hundred subsets of those atoms each vote for a token at every step.  The votes
are the token distribution.

If a separate `combinatoric_mechanism.py` sits next to this file it is used;
otherwise the built-in implementation below is used.

  I Identity -> II Correspondence -> III Verification -> O Recycling
  M_{t+1} = O(III(II(I(M_t))))        I_{t+1} = O_t

API:  load_corpus, TextEngine, generate_text, build_dataset,
      save_engine, load_engine
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import csv
import gzip
import io
import json
import math
import os
import pickle
import random
import re
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "NumPy is required. Install it with: python -m pip install numpy"
    ) from exc

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


def content_word(token: str) -> bool:
    return len(token) > 3 and token not in PUNCT


# ---------------------------------------------------------------------------
# Combinatoric mechanism (built in; an external combinatoric_mechanism.py
# next to this file takes precedence if present)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from combinatoric_mechanism import MechanismCombinator, StepContext  # type: ignore
    MECHANISM_SOURCE = "external combinatoric_mechanism.py"
except ImportError:
    MECHANISM_SOURCE = "built-in"

    ATOM_NAMES = (
        "ngram", "ramp", "alignment", "consistency", "ledger", "turing_state",
        "harmonization", "association", "continuity", "corpus_4gram",
        "state_filter", "tau_gate", "epsilon_gate", "rank_prior", "repetition",
    )

    @dataclass
    class StepContext:  # type: ignore[no-redef]
        tape: List[str]
        ranked: List[Tuple[str, float]]
        state: str
        lam: float
        fragments: Any
        delta_fn: Callable[[str], float]
        harm_fn: Callable[[str], float]
        associations: Dict[str, Counter]
        anchors: List[str]
        corpus_ngrams: Optional[Dict[int, Set[Tuple[str, ...]]]]
        tau: float
        epsilon: float

    class MechanismCombinator:  # type: ignore[no-redef]
        """Every non-empty subset of atoms (up to n_masks of the 2^15-1) votes."""

        def __init__(self, n_masks: int = 512, seed: int = 7) -> None:
            n_atoms = len(ATOM_NAMES)
            space = 2 ** n_atoms - 1
            n = max(1, min(int(n_masks), space))
            rng = random.Random(seed)
            codes = (list(range(1, space + 1)) if n == space
                     else rng.sample(range(1, space + 1), n))
            self.masks: List[int] = codes
            self.matrix = np.array(
                [[(c >> b) & 1 for b in range(n_atoms)] for c in codes],
                dtype=float)

        # -- atom values for every candidate: shape (n_candidates, n_atoms) --
        @staticmethod
        def _association(ctx: StepContext, token: str) -> float:
            if not ctx.anchors:
                return 0.0
            total = 0.0
            for anchor in ctx.anchors:
                for variant in (anchor, anchor.lower(), anchor.capitalize()):
                    counts = ctx.associations.get(variant)
                    if counts:
                        total += counts.get(token, 0) or counts.get(token.lower(), 0)
                        break
            return math.log1p(total)

        @staticmethod
        def _state_filter(ctx: StepContext, token: str) -> float:
            if ctx.state == "DESCRIBE":
                recent = {t for t in ctx.tape[-12:] if content_word(t)}
                return -1.0 if token in recent else 0.0
            if ctx.state == "ABSTRACT":
                ok = token in PUNCT or len(token) <= 3 or len(token) >= 7
                return 0.0 if ok else -1.0
            if ctx.state == "RELATE":
                return -1.0 if token in set(ctx.tape[-8:]) else 0.0
            return 0.0

        @staticmethod
        def _state_bonus(state: str, token: str) -> float:
            if state == "ABSTRACT" and len(token) >= 7 and token not in PUNCT:
                return 1.0
            if state == "DESCRIBE" and content_word(token):
                return 0.8
            if state == "RELATE" and content_word(token):
                return 0.4
            if state == "REFLECT":
                return 0.2
            return 0.0

        def _atoms(self, ctx: StepContext, tokens: List[str],
                   logps: Dict[str, Tuple[int, float]]) -> np.ndarray:
            frag = ctx.fragments
            tape = ctx.tape
            tail3 = tuple(tape[-3:])
            grams4 = (ctx.corpus_ngrams or {}).get(4, set())
            recent6 = set(tape[-6:])
            rows = []

            for token in tokens:
                rank, logp = logps[token]
                total, consistency = frag.score(tape, token, logp, rank, ctx.lam)
                alignment = float(np.clip((total - logp) / 2.5 + 0.5, 0.0, 1.0))
                delta = ctx.delta_fn(token)
                cached = frag._cache.get(token)
                ledger = (frag.ledger.bonus(frag.previous, cached)
                          if cached is not None else 0.0)
                four = 1.0 if (len(tail3) == 3 and tail3 + (token,) in grams4) else 0.0
                rep = -1.0 if (content_word(token) and token in recent6) else 0.0

                rows.append([
                    logp,                                        # ngram
                    ctx.lam * logp,                              # ramp
                    alignment,                                   # alignment
                    -consistency,                                # consistency
                    ledger,                                      # ledger
                    self._state_bonus(ctx.state, token),         # turing_state
                    ctx.harm_fn(token),                          # harmonization
                    self._association(ctx, token),               # association
                    -delta,                                      # continuity
                    four,                                        # corpus_4gram
                    self._state_filter(ctx, token),              # state_filter
                    -1.0 if consistency > ctx.tau else 0.0,      # tau_gate
                    -1.0 if delta > ctx.epsilon else 0.0,        # epsilon_gate
                    1.0 / (1.0 + rank),                          # rank_prior
                    rep,                                         # repetition
                ])
            return np.asarray(rows, dtype=float)

        def vote(self, ctx: StepContext) -> Tuple[Counter, List[str]]:
            tokens: List[str] = []
            logps: Dict[str, Tuple[int, float]] = {}
            for rank, (token, logp) in enumerate(ctx.ranked):
                if token not in logps:
                    logps[token] = (rank, logp)
                    tokens.append(token)
            if not tokens:
                return Counter(), []

            atoms = self._atoms(ctx, tokens, logps)
            mean = atoms.mean(axis=0)
            std = atoms.std(axis=0)
            safe = np.where(std > 1e-9, std, 1.0)
            z = np.where(std > 1e-9, (atoms - mean) / safe, 0.0)

            scores = z @ self.matrix.T                  # (candidates, masks)
            winner_idx = scores.argmax(axis=0)          # ties -> best n-gram rank
            winners = [tokens[i] for i in winner_idx]
            return Counter(winners), winners


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

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
# Stage-aware synthesis: real composition of prior Turing-stage evidence
# ---------------------------------------------------------------------------

STAGE_KEYS = (
    "OBSERVE", "DESCRIBE", "RELATE", "ABSTRACT", "SYNTHESIZE", "REFLECT",
)

# Relative contribution of each completed stage to the synthesized process state.
SYNTHESIS_WEIGHTS = {
    "OBSERVE": 0.20,
    "DESCRIBE": 0.25,
    "RELATE": 0.25,
    "ABSTRACT": 0.30,
}


def new_stage_memory() -> Dict[str, np.ndarray]:
    """Create one decaying memory vector per Turing process stage."""
    return {key: np.zeros(MEMORY_WIDTH, dtype=float) for key in STAGE_KEYS}


def update_stage_memory(
    stage_memory: Dict[str, np.ndarray],
    state: str,
    token: str,
    decay: float = 0.7,
) -> None:
    """Fold a token into the memory belonging to the stage that emitted it."""
    if state not in stage_memory:
        return
    stage_memory[state] = decay * stage_memory[state] + _phi(token)


def synthesize_stage_memory(
    stage_memory: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compose OBSERVE/DESCRIBE/RELATE/ABSTRACT evidence into SYNTHESIZE."""
    combined = np.zeros(MEMORY_WIDTH, dtype=float)

    for stage, weight in SYNTHESIS_WEIGHTS.items():
        vector = _unit_state(stage_memory.get(stage, np.zeros(MEMORY_WIDTH)))
        combined += weight * vector

    return _unit_state(combined)


def synthesis_affinity(
    token: str,
    synthesis_vector: np.ndarray,
) -> float:
    """Alignment between a candidate token and the synthesized process state."""
    if float(np.linalg.norm(synthesis_vector)) < 1e-9:
        return 0.0
    return float(np.dot(_phi(token), synthesis_vector))


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

    # Stage-aware processual control: each Turing stage retains evidence from
    # the tokens it generated; SYNTHESIZE composes those stage memories.
    stage_memory = new_stage_memory()
    previous_machine_state = machine.state
    synthesis_vector = np.zeros(MEMORY_WIDTH, dtype=float)

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

                # Processual synthesis: in SYNTHESIZE, prefer candidates that
                # continue the process assembled from prior stages. In REFLECT,
                # prefer candidates that reduce divergence between the
                # synthesized process state and the actual generated tape.
                if state == "SYNTHESIZE":
                    score += 0.8 * synthesis_affinity(
                        token, synthesis_vector
                    )
                elif state == "REFLECT":
                    process_state = _unit_state(
                        stage_memory.get("SYNTHESIZE", synthesis_vector)
                    )
                    global_state = _unit_state(raw)
                    current_gap = continuity_delta(process_state, global_state)
                    next_global_state = _unit_state(_fold(raw, token))
                    next_gap = continuity_delta(
                        process_state, next_global_state
                    )
                    score += 1.0 * (current_gap - next_gap)

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

            # Record the emitted token in the memory of the stage that chose it.
            update_stage_memory(stage_memory, state, chosen.token)

            # When the process enters SYNTHESIZE, compose the evidence left by
            # OBSERVE, DESCRIBE, RELATE, and ABSTRACT into a real synthesis state.
            if next_state == "SYNTHESIZE" and previous_machine_state != "SYNTHESIZE":
                synthesis_vector = synthesize_stage_memory(stage_memory)
                stage_memory["SYNTHESIZE"] = synthesis_vector

            previous_machine_state = next_state

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
OPTICAL_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TEXT_FIELDS = ("text", "content", "body", "sentence", "document")


def optical_preprocess_image(path: Path) -> Tuple[str, Dict[str, Any]]:
    """Apply the paper's 16x16 spatial pooling and confidence gate to one image.

    The image is treated as one observed frame. It is converted to grayscale,
    resized to rows*16 by cols*16, reshaped into (rows,16,cols,16), and reduced
    with one vectorised mean over the two within-cell axes. Cells farther than
    confidence from the threshold become BIT0/BIT1; cells in the threshold band
    become ERASURE. This does not simulate repeated exposures or camera noise.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Optical image preprocessing needs Pillow: python -m pip install pillow"
        ) from exc

    rows = max(1, int(CONFIG.get("optical_rows", 8)))
    cols = max(1, int(CONFIG.get("optical_cols", 8)))
    cell_size = 16  # fixed by the paper
    threshold = float(CONFIG.get("optical_threshold", 127.0))
    confidence = max(0.0, float(CONFIG.get("optical_confidence", 25.0)))

    with Image.open(path) as source:
        gray = source.convert("L").resize(
            (cols * cell_size, rows * cell_size), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(gray, dtype=np.float32)

    # One parallel pooling operation: P[i,j] is the mean of its 256 pixels.
    blocks = pixels.reshape(rows, cell_size, cols, cell_size)
    pooled = blocks.mean(axis=(1, 3))

    low = threshold - confidence
    high = threshold + confidence
    symbols = np.full((rows, cols), "ERASURE", dtype=object)
    symbols[pooled <= low] = "BIT0"
    symbols[pooled >= high] = "BIT1"

    lines = [f"OPTICAL_IMAGE ROWS {rows} COLS {cols}"]
    for i in range(rows):
        lines.append(f"ROW{i} " + " ".join(symbols[i].tolist()))
    text = " ".join(lines)
    stats = {
        "source": str(path),
        "rows": rows,
        "cols": cols,
        "cells": rows * cols,
        "bit0": int(np.count_nonzero(symbols == "BIT0")),
        "bit1": int(np.count_nonzero(symbols == "BIT1")),
        "erasures": int(np.count_nonzero(symbols == "ERASURE")),
        "threshold": threshold,
        "confidence": confidence,
        "pooling": f"{cell_size}x{cell_size}",
    }
    return text, stats


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
    if suffix in OPTICAL_IMAGE_SUFFIXES:
        optical_text, stats = optical_preprocess_image(path)
        print(
            "[optical-preprocess] "
            f"{path.name}: {stats['cells']} cells · "
            f"BIT0={stats['bit0']} BIT1={stats['bit1']} "
            f"ERASURE={stats['erasures']} · {stats['pooling']} pooling"
        )
        return [optical_text]
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
    return [path.read_text(encoding="utf-8-sig")]


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
                           if p.suffix.lower() in TEXT_SUFFIXES | TABLE_SUFFIXES | OPTICAL_IMAGE_SUFFIXES)
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
# Settings: edit here, or use the GUI controls
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
    # Paper-derived optical pre-dataset preprocessing (for image corpus files).
    "optical_rows": 8,
    "optical_cols": 8,
    "optical_threshold": 127.0,
    "optical_confidence": 25.0,
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

        self.warm_limit = int(warm_limit)
        self.model = NGramModel(order=order, alpha=alpha)
        self.corpus_tokens = tokenize(self.corpus_text)
        if not self.corpus_tokens:
            raise ValueError("Corpus contains no recognizable tokens.")

        self.model.train(self.corpus_text)
        self._derive()

    def _derive(self) -> None:
        """Rebuild everything that follows from the trained model + corpus text."""
        self.corpus_tokens = tokenize(self.corpus_text)
        self.associations = build_associations(self.model)
        self.warm_tokens = self.corpus_tokens[:max(0, self.warm_limit)] or None
        self.sentences = [
            tokenize(s)
            for s in re.split(r"(?<=[.!?])\s+", self.corpus_text.strip())
            if s.strip()
        ]
        self.corpus_ngrams = build_corpus_ngrams(self.sentences)

    # -- persistence ---------------------------------------------------------
    def to_state(self) -> Dict[str, Any]:
        """Plain-data snapshot (dicts/lists/tuples/strings/numbers only)."""
        return {
            "format": MODEL_FORMAT,
            "version": MODEL_VERSION,
            "order": self.model.order,
            "alpha": self.model.alpha,
            "total_tokens": self.model.total_tokens,
            "vocab": dict(self.model.vocab),
            "counts": [
                {ctx: dict(targets) for ctx, targets in level.items()}
                for level in self.model.counts
            ],
            "corpus_text": self.corpus_text,
            "sources": list(self.sources),
            "warm_limit": self.warm_limit,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TextEngine":
        if not isinstance(state, dict) or state.get("format") != MODEL_FORMAT:
            raise ValueError("Not a Turing text-engine model file.")
        if state.get("version") != MODEL_VERSION:
            raise ValueError(
                f"Unsupported model version {state.get('version')!r} "
                f"(this build reads version {MODEL_VERSION}).")
        order = int(state["order"])
        counts = state["counts"]
        if len(counts) != order:
            raise ValueError("Corrupt model file: n-gram order mismatch.")

        engine = cls.__new__(cls)
        engine.corpus_text = str(state["corpus_text"])
        engine.sources = [str(x) for x in state.get("sources", [])]
        engine.warm_limit = int(state.get("warm_limit", 20000))

        model = NGramModel(order=order, alpha=float(state["alpha"]))
        model.counts = [
            defaultdict(Counter, {tuple(ctx): Counter(targets)
                                  for ctx, targets in level.items()})
            for level in counts
        ]
        model.vocab = Counter(state["vocab"])
        model.total_tokens = int(state["total_tokens"])
        engine.model = model
        engine._derive()
        if not engine.corpus_tokens:
            raise ValueError("Corrupt model file: empty corpus.")
        return engine

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


MODEL_FORMAT = "turing-text-engine"
MODEL_VERSION = 1
MODEL_SUFFIX = ".tengine"


class _SafeUnpickler(pickle.Unpickler):
    """Only plain data may be loaded: no classes or callables except set types."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" and name in {"set", "frozenset"}:
            return getattr(builtins, name)
        raise pickle.UnpicklingError(
            f"Blocked unsafe object in model file: {module}.{name}")


def save_engine(engine: TextEngine, path: Any) -> Path:
    """Save a trained engine (gzip-compressed). Writes atomically."""
    target = Path(path)
    if not target.suffix:
        target = target.with_suffix(MODEL_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as handle:
        pickle.dump(engine.to_state(), handle, protocol=4)
    os.replace(tmp, target)
    return target


def load_engine(path: Any) -> TextEngine:
    """Load an engine saved by save_engine (restricted, data-only unpickling)."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Model file not found: {source}")
    try:
        with gzip.open(source, "rb") as handle:
            state = _SafeUnpickler(handle).load()
    except (OSError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Could not read model file: {exc}") from exc
    return TextEngine.from_state(state)


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
# Command-line interface (Gradio-free)
# ---------------------------------------------------------------------------

import argparse


def _add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", nargs="+", default=None,
                        help="Text files, directories, CSV/JSON files, images, or hf:dataset sources.")
    parser.add_argument("--model-in", default=None,
                        help="Load an existing .tengine model instead of training from corpus.")
    parser.add_argument("--order", type=int, default=CONFIG["order"])
    parser.add_argument("--alpha", type=float, default=CONFIG["alpha"])
    parser.add_argument("--warm-limit", type=int, default=CONFIG["warm_limit"])
    parser.add_argument("--hf-max-samples", type=int, default=CONFIG["hf_max_samples"])
    parser.add_argument("--no-hf-streaming", action="store_true",
                        help="Download/read the Hugging Face dataset without streaming.")
    parser.add_argument("--optical-rows", type=int, default=CONFIG["optical_rows"])
    parser.add_argument("--optical-cols", type=int, default=CONFIG["optical_cols"])
    parser.add_argument("--optical-threshold", type=float, default=CONFIG["optical_threshold"])
    parser.add_argument("--optical-confidence", type=float, default=CONFIG["optical_confidence"])


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tokens", type=int, default=CONFIG["tokens"])
    parser.add_argument("--n-masks", type=int, default=CONFIG["n_masks"])
    parser.add_argument("--beam", type=int, default=CONFIG["beam"])
    parser.add_argument("--tau", type=float, default=CONFIG["tau"])
    parser.add_argument("--epsilon", type=float, default=CONFIG["epsilon"])
    parser.add_argument("--temperature", type=float, default=CONFIG["temperature"])
    parser.add_argument("--levels", type=int, default=CONFIG["levels"])
    parser.add_argument("--ledger-weight", type=float, default=CONFIG["ledger_weight"])
    parser.add_argument("--keep-learning", action="store_true", default=CONFIG["keep_learning"])
    parser.add_argument("--initial-state", choices=[s for s in TuringMachine.STATES if s != "HALT"],
                        default=CONFIG["initial_state"])
    parser.add_argument("--template-associates", type=int, default=CONFIG["template_associates"])


def _prepare_config(args: argparse.Namespace) -> None:
    CONFIG["hf_max_samples"] = args.hf_max_samples
    CONFIG["hf_streaming"] = not args.no_hf_streaming
    CONFIG["optical_rows"] = args.optical_rows
    CONFIG["optical_cols"] = args.optical_cols
    CONFIG["optical_threshold"] = args.optical_threshold
    CONFIG["optical_confidence"] = args.optical_confidence


def _get_engine(args: argparse.Namespace) -> TextEngine:
    _prepare_config(args)
    if args.model_in:
        engine = load_engine(args.model_in)
        print(f"Loaded model: {args.model_in}")
    else:
        corpus_text, sources = load_corpus(args.corpus)
        engine = TextEngine(corpus_text=corpus_text, order=args.order,
                            alpha=args.alpha, warm_limit=args.warm_limit)
        engine.sources = sources
        print("Training corpus sources:")
        for source in sources:
            print(f"  - {source}")
    print(f"Corpus tokens: {len(engine.corpus_tokens):,} | vocabulary: {len(engine.model.vocab):,} | n-gram order: {engine.model.order}")
    return engine


def _generation_options(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "tokens": args.tokens,
        "n_masks": args.n_masks,
        "beam": args.beam,
        "tau": args.tau,
        "epsilon": args.epsilon,
        "temperature": args.temperature,
        "levels": args.levels,
        "ledger_weight": args.ledger_weight,
        "keep_learning": args.keep_learning,
        "initial_state": args.initial_state,
        "template_associates": args.template_associates,
    }


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turing state-machine text generator with combinatoric token election and paper-derived optical image preprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate text from a prompt.",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_engine_arguments(gen)
    _add_generation_arguments(gen)
    gen.add_argument("--prompt", default=CONFIG["default_prompt"])
    gen.add_argument("--seed", type=int, default=CONFIG["seed"])
    gen.add_argument("--trace", action="store_true")
    gen.add_argument("--model-out", default=None, help="Optionally save the trained/updated model.")
    gen.add_argument("--output", default=None, help="Optionally write generated text to this file.")

    data = sub.add_parser("dataset", help="Generate a prompt-by-seed dataset as JSONL or CSV.",
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_engine_arguments(data)
    _add_generation_arguments(data)
    data.add_argument("--prompts-file", default=None, help="UTF-8 file containing one prompt per line.")
    data.add_argument("--prompts", nargs="*", default=None, help="Explicit prompts.")
    data.add_argument("--n-prompts", type=int, default=CONFIG["dataset_prompts"])
    data.add_argument("--n-seeds", type=int, default=CONFIG["dataset_seeds"])
    data.add_argument("--seed", type=int, default=CONFIG["seed"])
    data.add_argument("--output", default=CONFIG["dataset_out"])
    data.add_argument("--no-segments", action="store_true")

    train = sub.add_parser("train", help="Train from corpus and save a reusable .tengine model.",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_engine_arguments(train)
    train.add_argument("--model-out", default="model.tengine")

    matrix = sub.add_parser("matrix", help="Compute the source function inner-product matrix as CSV.")
    matrix.add_argument("--output", default="function_inner_product.csv")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "matrix":
        names, gram = function_inner_product_matrix(Path(__file__))
        save_function_inner_product_matrix(Path(args.output), names, gram)
        print(f"Saved {len(names)} function vectors and {gram.shape[0]}x{gram.shape[1]} matrix to {args.output}")
        return 0

    if args.command == "train":
        if args.model_in:
            parser.error("train does not accept --model-in; omit it to train a new model")
        engine = _get_engine(args)
        path = save_engine(engine, args.model_out)
        print(f"Saved model: {path}")
        return 0

    engine = _get_engine(args)
    options = _generation_options(args)

    if args.command == "generate":
        result = engine.generate_text(args.prompt, seed=args.seed, trace=args.trace, **options)
        print("\n" + result["text"] + "\n")
        summary = {key: value for key, value in result.items() if key not in {"text", "segments", "ledger"}}
        summary["ledger"] = result["ledger"]
        print("Generation summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.trace:
            print("\nTrace:")
            for index, segment in enumerate(result["segments"], 1):
                print(f"[{index}] {json.dumps(segment, ensure_ascii=False, default=str)}")
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result["text"], encoding="utf-8")
            print(f"Wrote generated text to {output_path}")
        if args.model_out:
            print(f"Saved model: {save_engine(engine, args.model_out)}")
        return 0

    if args.command == "dataset":
        if args.prompts_file:
            prompts = [line.strip() for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        elif args.prompts:
            prompts = [prompt.strip() for prompt in args.prompts if prompt.strip()]
        else:
            prompts = engine.auto_prompts(max(1, args.n_prompts), seed=args.seed)
        if not prompts:
            raise ValueError("No dataset prompts were supplied or generated.")
        seeds = list(range(args.seed, args.seed + max(1, args.n_seeds)))
        rows = build_dataset(engine, prompts, seeds=seeds, out_path=args.output,
                             include_segments=not args.no_segments, progress=True, **options)
        print(f"Dataset complete: {len(rows)} rows -> {args.output}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
