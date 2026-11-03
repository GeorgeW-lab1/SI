#!/usr/bin/env python3
"""
Turing state-machine text generator with an ad-hoc association template
=======================================================================

The generate() function:

1. Tokenizes the prompt.
2. Extracts prompt content words.
3. Builds a symmetric association table directly from corpus bigrams.
4. Constructs an ad-hoc template such as:

       A process is <ASSOC> and <ASSOC>

5. Fills each <ASSOC> slot from the strongest corpus associates.
6. Uses the filled template as the initial Turing-machine tape.
7. Continues generation using explicit Turing states, one token at a time,
   through the recursive I-II-III-O memory loop.

The recursive memory loop (the "I-II-III-O Protocol")
-----------------------------------------------------
  I    Identity        recognise the current state: rank next-token candidates
                       (the Turing state's hard constraints apply here)
  II   Correspondence  signal between context and candidate (fragment score,
                       ledger bonus, Turing-state bonuses)
  III  Verification    continuity check  Delta(M_t, M_{t+1}) <= epsilon
  O    Recycling       reintegrate the chosen token: write it to the tape,
                       fold it into memory, advance the Turing machine

      M_{t+1} = O(III(II(I(M_t))))        I_{t+1} = O_t

M_t is a unit vector: the exponentially decayed sum of hashed token features
of everything on the tape. Delta is half the Euclidean distance between
consecutive unit states (0..1). Over-strict enforcement would classify
ordinary variation as error, so epsilon is relaxed geometrically whenever no
candidate passes, rather than rejecting everything.

Usage:
  python discrete_lock_turing.py corpus.txt --prompt "once upon a" --tokens 200 --trace
  python discrete_lock_turing.py --prompt "The machine" --tokens 120 --epsilon 0.4
  python discrete_lock_turing.py --interactive
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

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

    state:
      The control state in which this rule is active. (Previously missing, so
      every rule fired in every state and the machine never advanced.)

    read:
      "*" matches any symbol class; otherwise END, PUNCT, LONG, MEDIUM,
      SHORT or BLANK.

    write:
      A literal token to append, or None to append the sampled candidate.

    move:
      Head movement: -1, 0, or +1.

    min_steps / max_steps:
      Bounds on the number of steps already spent IN `state`.
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
    Apply a small context-sensitive score adjustment to equation-like text.

    If a recent context looks like an equation and contains the letter ``i``
    without the digit ``1``, prefer ``1`` over ``i`` as the next token. This is
    only a heuristic for token selection; it does not prove or solve algebra,
    and it deliberately does not rewrite the existing tape.

    Exemption: when the context contains the I-II-III-O protocol symbols
    (II / III), "I" is the stage name Identity, not a mistyped 1, so no
    adjustment is made.
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
    and canonical return-expression identities. The identity component recognizes
    limited forms such as ``x + y`` versus ``y + x`` and parameter renaming.
    G = X @ X.T is an inner-product matrix. It measures shared features and
    recognized syntactic identities; it does not prove arbitrary functions equal.
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
) -> Tuple[str, List[Dict[str, object]], TransitionLedger]:
    """
    Generate text using an ad-hoc template constructed from prompt
    associations, continued by a Turing machine through the I-II-III-O loop.
    """

    rng = random.Random(seed)
    ledger = TransitionLedger()
    fragments = LockedFragments(seed, ledger, levels, ledger_weight)

    # -------------------------------------------------------------
    # Prompt tools.
    # -------------------------------------------------------------
    prompt_tokens = tokenize(prompt) or ["The"]
    prompt_content = [t for t in prompt_tokens if content_word(t)]

    # -------------------------------------------------------------
    # Build a symmetric association table from corpus bigrams.
    # Punctuation and padding are never associates.
    # -------------------------------------------------------------
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

    def top_associates(word: str, k: int) -> List[str]:
        # Try the word as written, then lowercase / capitalised variants.
        for variant in (word, word.lower(), word.capitalize()):
            counts = associations.get(variant)
            if counts:
                return [t for t, _ in counts.most_common(k) if t not in SPECIAL]
        return []

    # -------------------------------------------------------------
    # Construct and fill the ad-hoc template:
    #   A process is <ASSOC> and <ASSOC>
    # A slot with no associates is dropped rather than echoing the prompt.
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
    # I_0 is the filled template, folded into the recursive memory.
    # -------------------------------------------------------------
    machine = TuringMachine(initial_state)

    segments: List[Dict[str, object]] = []
    produced = len(tape)
    raw = _fold_all(tape)
    total_relaxations = 0

    while produced < token_count:
        emitted: List[str] = []
        fallbacks = 0
        relaxations = 0
        closed = False
        start_step = produced
        segment_state = machine.state

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

            # ---- Stage III: Verification (Delta <= epsilon) ---------------
            # Relax epsilon instead of rejecting everything, so ordinary
            # variation is not misclassified as an error.
            eps_eff = max(1e-6, epsilon)
            verified = [c for c in scored if c.delta <= eps_eff]
            while not verified and eps_eff < 1.0:
                eps_eff = min(1.0, eps_eff * 1.5)
                relaxations += 1
                verified = [c for c in scored if c.delta <= eps_eff]
            if not verified:
                verified = [min(scored, key=lambda c: c.delta)]

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
                    f"delta={chosen.delta:.3f}"
                )

            next_state = machine.step(tape, chosen.token)

            if next_state == "HALT":
                closed = chosen.token in END
                break

        # Close an unfinished segment. The tape is edited too, so segment
        # text and tape always agree.
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
            }
        )

        if machine.is_halted():
            machine.reset("OBSERVE")

    if show_trace:
        print(f"[continuity] epsilon relaxations: {total_relaxations}")

    return detokenize(tape), segments, ledger


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Turing state-machine text generator using an ad-hoc "
            "association template and the I-II-III-O memory loop."
        )
    )

    parser.add_argument(
        "corpus",
        nargs="?",
        help="UTF-8 corpus file; omit to use the built-in corpus.",
    )
    parser.add_argument("--prompt", default="A process")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Read prompts from stdin repeatedly (blank line or EOF quits).",
    )
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--beam", type=int, default=12)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.6,
        help="Max continuity deviation Delta(M_t, M_t+1) per token (0..1); "
             "relaxed automatically if nothing passes.",
    )
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--initial-state",
        default="OBSERVE",
        choices=TuringMachine.STATES,
    )
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--ledger-weight", type=float, default=0.9)
    parser.add_argument("--warm-limit", type=int, default=20000)
    parser.add_argument(
        "--keep-learning",
        action="store_true",
        help="Keep the transition ledger mutable during generation.",
    )
    parser.add_argument(
        "--template-associates",
        type=int,
        default=4,
        help="Number of associates considered for each template slot.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print the template and per-token generation trace.",
    )
    parser.add_argument("--save", help="Write the generated text to this file.")
    parser.add_argument("--segments-json", help="Write segment statistics to this JSON file.")
    parser.add_argument(
        "--function-matrix",
        nargs="?",
        const="function_inner_product.csv",
        help="Write the inner-product (Gram) matrix of all functions/methods to CSV; optionally provide a path.",
    )
    parser.add_argument(
        "--show-function-matrix",
        action="store_true",
        help="Print the labelled function inner-product matrix before generation.",
    )

    args = parser.parse_args()

    corpus_path = Path(input("Filename: "))

 

    try:
        corpus_text = corpus_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        parser.error(f"Could not decode corpus as UTF-8: {exc}")
    

    corpus_tokens = tokenize(corpus_text)

    if not corpus_tokens:
        parser.error("Corpus contains no recognizable tokens.")

    model = NGramModel(order=args.order, alpha=args.alpha)
    model.train(corpus_text)

    # Corpus n-grams for the novelty statistic (built once, from the corpus
    # itself rather than from the generated tape).
    corpus_sentences = [
        tokenize(s)
        for s in re.split(r"(?<=[.!?])\s+", corpus_text.strip())
        if s.strip()
    ]
    corpus_ngrams = build_corpus_ngrams(corpus_sentences)

    if args.function_matrix or args.show_function_matrix:
        try:
            function_names, gram_matrix = function_inner_product_matrix(Path(__file__))
        except (OSError, SyntaxError) as exc:
            parser.error(f"Could not construct function inner-product matrix: {exc}")

        if args.function_matrix:
            matrix_path = Path(args.function_matrix)
            save_function_inner_product_matrix(matrix_path, function_names, gram_matrix)
            print(f"[function-matrix] saved {len(function_names)} x {len(function_names)} matrix to {matrix_path}")

        if args.show_function_matrix:
            print("\n--- IDENTITY-AWARE FUNCTION INNER-PRODUCT MATRIX ---")
            print("Functions:", " | ".join(function_names))
            np.set_printoptions(precision=1, suppress=True, linewidth=180)
            print(gram_matrix)

    def run(prompt: str) -> None:
        text, segments, _ledger = generate(
            model,
            input("Prompt> "),
            token_count=max(1, args.tokens),
            beam_size=max(1, args.beam),
            tau=float(np.clip(args.tau, 0.0, 1.0)),
            temperature=max(0.0, args.temperature),
            seed=args.seed,
            show_trace=args.trace,
            levels=args.levels,
            ledger_weight=args.ledger_weight,
            warm_tokens=(
                corpus_tokens[:max(0, args.warm_limit)]
                if args.warm_limit > 0
                else None
            ),
            lock_ledger=not args.keep_learning,
            initial_state=args.initial_state,
            template_associates=max(1, args.template_associates),
            epsilon=float(np.clip(args.epsilon, 0.0, 1.0)),
            corpus_ngrams=corpus_ngrams,
        )

        print("\n--- GENERATED TEXT ---\n")
        print(text)

        if args.save:
            Path(args.save).write_text(text, encoding="utf-8")
            print(f"\n[Saved output to {args.save}]")

        if args.segments_json:
            Path(args.segments_json).write_text(
                json.dumps(segments, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[Saved segments to {args.segments_json}]")

   
    while True:
        run(args.prompt)


if __name__ == "__main__":
    main()
