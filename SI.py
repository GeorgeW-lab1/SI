
#!/usr/bin/env python3
"""
Turing state-machine text generator
===================================

A corpus-based text generator controlled by an explicit Turing-style state
machine. The generated token sequence is the machine's tape; named states
control how the next token is proposed, filtered, scored, and accepted.

States:
  OBSERVE     continue local corpus-like flow
  DESCRIBE    introduce new content while avoiding recent repetition
  RELATE      bridge toward less-local vocabulary
  ABSTRACT    prefer long, content-bearing vocabulary
  SYNTHESIZE  blend local and broader corpus evidence
  REFLECT     recall and consolidate the generated history
  HALT        accepting state; ends the current segment

The previous module pipeline has been removed. There are no Module,
ModuleResult, blackboard, or dataset-phrase substitution components.

Usage:
  python discrete_lock_turing.py corpus.txt --prompt "once upon a" --tokens 120
  python discrete_lock_turing.py --prompt "The machine" --tokens 80
  python discrete_lock_turing.py corpus.txt --order 3 --trace
"""

from __future__ import annotations

import argparse
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
    raise SystemExit("NumPy is required. Install it with: python -m pip install numpy") from exc


TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)
PUNCT = set(".,!?;:")
END = {".", "!", "?"}
SPECIAL = {"", "<s>", "</s>"}

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


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


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
    seen = sum(1 for a, b in pairs if model.counts[1].get((a,), {}).get(b))
    return seen / len(pairs)


def novelty(ngrams: Dict[int, Set[Tuple[str, ...]]], tokens: Sequence[str], length: int = 4) -> float:
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
        if not tokens:
            return
        for i in range(self.order - 1, len(tokens)):
            target = tokens[i]
            self.vocab[target] += 1
            self.total_tokens += 1
            for ctx_len in range(self.order):
                context = tuple(tokens[i - ctx_len:i]) if ctx_len else ()
                self.counts[ctx_len][context][target] += 1

    def candidates(self, context: Sequence[str], limit: int = 80) -> List[Tuple[str, float]]:
        vocab_size = max(1, len(self.vocab))
        context = list(context)
        counter: Counter = Counter()

        for ctx_len in range(min(self.order - 1, len(context)), -1, -1):
            key = tuple(context[-ctx_len:]) if ctx_len else ()
            found = self.counts[ctx_len].get(key)
            if not found:
                continue
            weight = 1.0 + 0.7 * ctx_len
            for token, count in found.items():
                counter[token] += weight * count
            if ctx_len >= 1:
                break

        if not counter:
            counter.update(self.vocab)

        total = sum(counter.values())
        ranked = []
        for token, count in counter.items():
            probability = (count + self.alpha) / (total + self.alpha * vocab_size)
            ranked.append((token, math.log(max(probability, 1e-12))))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:max(1, limit)]


@dataclass(frozen=True)
class TuringTransition:
    """One transition rule.

    read:
      "*" matches any symbol class.
      Other valid values are END, PUNCT, LONG, MEDIUM, SHORT, WORD, BLANK.
    write:
      A literal token to append, or None to append the sampled candidate.
    move:
      Head movement: -1, 0, or +1.
    """

    read: str
    write: Optional[str]
    move: int
    next_state: str
    min_steps: int
    max_steps: int
    max_copy: Optional[int]
    priority: int


class TuringMachine:
    """Explicit finite control for generation.

    The tape is the token history. The head normally sits at the most recently
    written token. A transition reads its class, optionally writes a literal
    token, moves the head, and enters the next state.
    """

    STATES = (
        "OBSERVE",
        "DESCRIBE",
        "RELATE",
        "ABSTRACT",
        "SYNTHESIZE",
        "REFLECT",
        "HALT",
    )

    def __init__(self, initial_state: str = "OBSERVE") -> None:
        if initial_state not in self.STATES:
            raise ValueError(f"Unknown initial state: {initial_state}")
        self.state = initial_state
        self.head = 0
        self.steps = 0
        self.transitions: Tuple[TuringTransition, ...] = (
            TuringTransition("END", None, +1, "DESCRIBE", 3, 20, None, 20),
            TuringTransition("*", None, +1, "OBSERVE", 0, 20, None, 0),

            TuringTransition("END", None, +1, "RELATE", 4, 20, 3, 20),
            TuringTransition("*", None, +1, "DESCRIBE", 0, 20, 3, 0),

            TuringTransition("END", None, +1, "ABSTRACT", 5, 22, 3, 20),
            TuringTransition("*", None, +1, "RELATE", 0, 22, 3, 0),

            TuringTransition("END", None, +1, "SYNTHESIZE", 4, 24, 2, 20),
            TuringTransition("*", None, +1, "ABSTRACT", 0, 24, 2, 0),

            TuringTransition("END", None, +1, "REFLECT", 6, 24, 2, 20),
            TuringTransition("*", None, +1, "SYNTHESIZE", 0, 24, 2, 0),

            TuringTransition("END", None, 0, "HALT", 5, 24, 3, 20),
            TuringTransition("*", None, +1, "REFLECT", 0, 24, 3, 0),

        )

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

    def read(self, tape: Sequence[str]) -> str:
        if not tape:
            return ""
        index = min(max(self.head, 0), len(tape) - 1)
        return tape[index]

    def matching_rules(self, tape: Sequence[str]) -> List[TuringTransition]:
        symbol = self.classify(self.read(tape))
        return [
            rule for rule in self.transitions
            if rule.read in ("*", symbol)
            and rule.min_steps <= self.steps <= rule.max_steps
        ]

    def step(self, tape: List[str], emitted_token: str) -> str:
        """Apply the highest-priority matching rule and return the next state."""
        symbol = self.classify(emitted_token)
        applicable = [
            rule for rule in self.transitions
            if rule.read in ("*", symbol)
            and rule.min_steps <= self.steps <= rule.max_steps
        ]

        self.steps += 1

        if not applicable:
            self.state = "HALT"
            return self.state

        rule = max(applicable, key=lambda item: item.priority)

        if rule.write is not None:
            tape.append(rule.write)

        if tape:
            self.head = max(0, min(len(tape) - 1, self.head + rule.move))
        else:
            self.head = 0

        self.state = rule.next_state
        return self.state

    def is_halted(self) -> bool:
        return self.state == "HALT"

    def snapshot(self) -> Dict[str, object]:
        return {
            "state": self.state,
            "head": self.head,
            "steps": self.steps,
            "tape_length": self.head + 1,
        }


class TransitionLedger:
    """Count transitions between quantized fragment states."""

    def __init__(self) -> None:
        self.counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
        self.locked = False

    def record(self, previous: Optional[Tuple[int, ...]], current: Tuple[int, ...]) -> None:
        if self.locked or previous is None:
            return
        self.counts[previous][current] += 1

    def bonus(self, previous: Optional[Tuple[int, ...]], current: Tuple[int, ...]) -> float:
        if previous is None:
            return 0.0
        return math.log1p(self.counts.get(previous, {}).get(current, 0))

    def lock(self) -> "TransitionLedger":
        self.locked = True
        return self

    def stats(self) -> Dict[str, object]:
        return {
            "states": len(self.counts),
            "transitions": sum(len(targets) for targets in self.counts.values()),
            "locked": self.locked,
        }


class LockedFragments:
    """Projects context/token/transition features into quantized state cells."""

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
        values[1] += sum(chr(byte).isalpha() for byte in raw) / max(1, len(raw))
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
                context_vector += self._hash_features(word, 5) * (0.65 ** (len(recent) - i - 1))
            context_vector /= max(1, len(recent))

        word_vector = self._hash_features(token, 5)
        transition_vector = np.array([
            np.tanh(logp),
            1.0 / (1.0 + rank),
            1.0 if token[:1].isupper() else 0.0,
            1.0 if token in ".,!?;:" else 0.0,
            min(len(token), 24) / 24.0,
        ], dtype=float)

        g1 = np.tanh(context_vector @ self.w_context)
        g2 = np.tanh(word_vector @ self.w_word)
        g3 = np.tanh(transition_vector @ self.w_transition)
        return g1, g2, g3

    @staticmethod
    def quantize(vector: np.ndarray, levels: int) -> Tuple[int, ...]:
        indexes = np.clip(
            np.floor((np.asarray(vector, dtype=float) + 1.0) / 2.0 * levels),
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
        state = self.quantize(np.concatenate([g1, g2, g3]), self.levels)
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
        summaries = np.array([
            np.linalg.norm(g1),
            np.linalg.norm(g2),
            np.linalg.norm(g3),
        ])
        mean = float(np.mean(summaries))
        spread = float(np.std(summaries) / (mean + 1e-9))
        consistency = float(np.clip(0.55 * spread + 0.45 * (1.0 - mean), 0.0, 1.0))

        reference = np.array([
            np.mean(g1),
            np.mean(g2),
            np.mean(g3),
            np.tanh(logp),
        ])
        reference /= max(float(np.linalg.norm(reference)), 1e-12)

        candidate = np.array([
            g1[0],
            g2[0],
            g3[0],
            np.tanh(logp + 0.1 * rank),
        ])
        candidate /= max(float(np.linalg.norm(candidate)), 1e-12)

        alignment = (float(np.dot(reference, candidate)) + 1.0) / 2.0
        fragment_score = 1.6 * alignment - 1.15 * consistency
        total = (
            (1.0 - lam) * (0.75 * logp + 0.25 * fragment_score)
            + lam * (1.25 * logp + 0.85 * fragment_score)
        )

        state = self._cache.get(token)
        if state is not None:
            total += self.ledger_weight * self.ledger.bonus(self.previous, state)

        return total, consistency

    def commit(self, token: str) -> Tuple[int, ...]:
        state = self._cache[token]
        self.ledger.record(self.previous, state)
        self.previous = state
        self._cache.clear()
        return state


def content_word(token: str) -> bool:
    return len(token) > 3 and token not in PUNCT


def build_corpus_ngrams(sentences: Sequence[Sequence[str]]) -> Dict[int, Set[Tuple[str, ...]]]:
    ngrams: Dict[int, Set[Tuple[str, ...]]] = {}
    for length in range(2, 9):
        grams: Set[Tuple[str, ...]] = set()
        for tokens in sentences:
            for i in range(len(tokens) - length + 1):
                grams.add(tuple(tokens[i:i + length]))
        ngrams[length] = grams
    return ngrams


def run_turing_state(
    machine: TuringMachine,
    model: NGramModel,
    fragments: LockedFragments,
    tape: List[str],
    corpus_ngrams: Dict[int, Set[Tuple[str, ...]]],
    rng: random.Random,
    *,
    budget: int,
    step0: int,
    total_steps: int,
    beam_size: int,
    tau: float,
    temperature: float,
    show_trace: bool = False,
) -> Tuple[List[str], Dict[str, object]]:
    """Run the machine until it halts, reaches its step budget, or emits END."""

    machine.reset()
    emitted: List[str] = []
    fallbacks = 0
    forced_close = False
    closed = False

    for local in range(max(1, budget)):
        lam = adiabatic_lambda(step0 + local, max(1, total_steps))
        state = machine.state

        ranked = [
            (token, logp)
            for token, logp in model.candidates(tape, limit=max(40, beam_size * 8))
            if token not in SPECIAL
        ]

        if machine.steps < 3:
            nonterminal = [(t, lp) for t, lp in ranked if t not in END]
            if nonterminal:
                ranked = nonterminal

        # State-specific hard filters.
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
            # Prefer tokens not immediately repeated from the local tail.
            local_tail = set(tape[-8:])
            filtered = [(t, lp) for t, lp in ranked if t not in local_tail]
            if filtered:
                ranked = filtered

        elif state == "SYNTHESIZE":
            # Suppress immediate loops, but retain normal corpus candidates.
            if len(tape) >= 2:
                previous = tape[-1]
                seen_pairs = set(zip(tape[-20:], tape[-19:]))
                filtered = [
                    (t, lp) for t, lp in ranked
                    if (previous, t) not in seen_pairs
                ]
                if filtered:
                    ranked = filtered

        # Avoid long verbatim corpus runs.
        max_copy = next(
            (rule.max_copy for rule in machine.matching_rules(tape) if rule.max_copy is not None),
            None,
        )
        if max_copy is not None:
            run_length = max_copy + 1
            grams = corpus_ngrams.get(run_length, set())
            tail = tuple(tape[-(run_length - 1):]) if run_length > 1 else ()
            filtered = [
                (t, lp) for t, lp in ranked
                if len(tail) < run_length - 1 or (tail + (t,)) not in grams
            ]
            if filtered:
                ranked = filtered

        if not ranked:
            fallbacks += 1
            ranked = [(".", -8.0)]

        scored: List[Candidate] = []
        for rank, (token, logp) in enumerate(ranked):
            fragments.locked_state(tape, token, logp, rank)
            score, consistency = fragments.score(tape, token, logp, rank, lam)

            # State-specific soft preferences.
            if state == "ABSTRACT" and len(token) >= 7 and token not in PUNCT:
                score += 1.0
            elif state == "DESCRIBE" and content_word(token):
                score += 0.8
            elif state == "RELATE" and content_word(token):
                score += 0.4
            elif state == "REFLECT":
                score += 0.2

            alignment = float(np.clip((score - logp) / 2.5 + 0.5, 0.0, 1.0))
            if consistency <= tau:
                scored.append(Candidate(token, logp, score, consistency, alignment))

        if not scored:
            token, logp = ranked[0]
            fragments.locked_state(tape, token, logp, 0)
            score, consistency = fragments.score(tape, token, logp, 0, lam)
            scored = [Candidate(token, logp, score, consistency, 0.5)]

        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        beam = scored[:max(1, beam_size)]
        probabilities = softmax([candidate.score for candidate in beam], temperature)

        for candidate, probability in zip(beam, probabilities):
            candidate.probability = float(probability)

        if temperature <= 0.01:
            chosen = beam[0]
        else:
            chosen = rng.choices(beam, weights=probabilities, k=1)[0]

        fragments.commit(chosen.token)
        tape.append(chosen.token)
        emitted.append(chosen.token)

        if show_trace:
            print(
                f"[{step0 + local:03d}] {machine.state:<10} "
                f"{chosen.token!r:<16} "
                f"p={math.exp(chosen.base_logp):.2f} "
                f"score={chosen.score:+.3f} "
                f"D={chosen.consistency:.3f}"
                f"{' FALLBACK' if fallbacks and local == 0 else ''}"
            )

        next_state = machine.step(tape, chosen.token)

        if next_state == "HALT":
            closed = chosen.token in END
            break

    if emitted and emitted[-1] not in END:
        if emitted[-1] in PUNCT:
            emitted[-1] = "."
        else:
            emitted.append(".")
        closed = True
        forced_close = True

    diagnostics = {
        "state": machine.state,
        "steps": machine.steps,
        "head": machine.head,
        "tokens": list(emitted),
        "text": detokenize(emitted),
        "natural_rate": natural_rate(model, emitted),
        "novelty": novelty(corpus_ngrams, emitted),
        "fallbacks": fallbacks,
        "closed": closed,
        "forced_close": forced_close,
    }
    return emitted, diagnostics


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
) -> Tuple[str, List[Dict[str, object]], TransitionLedger]:
    rng = random.Random(seed)
    ledger = TransitionLedger()
    fragments = LockedFragments(seed, ledger, levels, ledger_weight)

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

        if lock_ledger:
            ledger.lock()

    tape = tokenize(prompt) or ["The"]
    prompt_line = detokenize(tape)
    segments: List[Dict[str, object]] = []
    machine = TuringMachine(initial_state)
    produced = 0

    while produced < token_count:
        segment, diagnostics = run_turing_state(
            machine,
            model,
            fragments,
            tape,
            build_corpus_ngrams([tape]),
            rng,
            budget=token_count - produced,
            step0=produced,
            total_steps=token_count,
            beam_size=beam_size,
            tau=float(np.clip(tau, 0.0, 1.0)),
            temperature=max(0.0, temperature),
            show_trace=show_trace,
        )

        if not segment:
            break

        produced += len(segment)
        diagnostics["segment_index"] = len(segments)
        segments.append(diagnostics)

        if machine.is_halted():
            machine.reset("OBSERVE")

    return detokenize(tape), segments, ledger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corpus-based text generator controlled by a Turing state machine."
    )
    parser.add_argument("corpus", nargs="?", help="UTF-8 corpus file; omit to use the built-in corpus.")
    parser.add_argument("--prompt", default="A process", help="Initial prompt.")
    parser.add_argument("--tokens", type=int, default=500, help="Maximum number of new tokens.")
    parser.add_argument("--order", type=int, default=4, help="N-gram order.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Additive smoothing.")
    parser.add_argument("--beam", type=int, default=12, help="Candidate beam size.")
    parser.add_argument("--tau", type=float, default=0.95, help="Maximum allowed consistency score.")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--initial-state", default="OBSERVE", choices=TuringMachine.STATES)
    parser.add_argument("--levels", type=int, default=15, help="Grid cells per feature dimension.")
    parser.add_argument("--ledger-weight", type=float, default=0.3, help="Bonus for known locked transitions.")
    parser.add_argument("--warm-limit", type=int, default=20000, help="Corpus tokens used to warm the ledger.")
    parser.add_argument("--keep-learning", action="store_true", help="Keep the transition ledger mutable.")
    parser.add_argument("--trace", action="store_true", help="Print per-token machine trace.")
    parser.add_argument("--save", help="Save generated text to this file.")
    parser.add_argument("--segments-json", help="Save segment diagnostics to this JSON file.")
    args = parser.parse_args()

    if args.corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.is_file():
            parser.error(f"Corpus file not found: {corpus_path}")
        try:
            corpus_text = corpus_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            parser.error(f"Could not decode corpus as UTF-8: {exc}")
    else:
        corpus_text = BUILTIN_CORPUS

    corpus_tokens = tokenize(corpus_text)
    if not corpus_tokens:
        parser.error("Corpus contains no recognizable tokens.")

    model = NGramModel(order=args.order, alpha=args.alpha)
    model.train(corpus_text)
    while True:
        text, segments, ledger = generate(
            model,
            input("USER: "),
            token_count=max(1, args.tokens),
            beam_size=max(1, args.beam),
            tau=args.tau,
            temperature=args.temperature,
            seed=args.seed,
            show_trace=args.trace,
            levels=args.levels,
            ledger_weight=args.ledger_weight,
            warm_tokens=corpus_tokens[:max(0, args.warm_limit)] if args.warm_limit > 0 else None,
            lock_ledger=not args.keep_learning,
            initial_state=args.initial_state,
        )

        print("\n--- GENERATED TEXT ---\n")
        print(text)


if __name__ == "__main__":
    main()
