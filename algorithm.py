"""
Symbol-set adiabatic text generator
===================================

Character-level version of the non-isomorphic adiabatic generator.
Tokens are single symbols. The alphabet is a universe U, and every
"fragment" reasons with sets over U using logical set operations
(union |, intersection &, xor ^, difference -, complement ~).

Conceptual mapping:
  F       reference next-symbol distribution from the n-gram model
  g_k     fragment-specific set-valued lifts of (context, candidate)
            g1: Follow-set space      (subsets of the alphabet)
            g2: class-signature space (bit-vectors over symbol classes)
            g3: recent-window space   (subset of the last few symbols)
  lambda  smooth adiabatic ramp from exploratory to selective scoring
  D       consistency score (lower = more consistent), built from set ops
  tau     maximum permitted D for a candidate

Set-expression gates (optional):
  --allow  "LETTER | SPACE | END"        only these symbols may be emitted
  --forbid "DIGIT ^ (DIGIT & '0')"       these symbols are removed
  Names: ALL LOWER UPPER LETTER VOWEL CONSONANT DIGIT ALNUM SPACE END PAUSE
         QUOTE BRACKET OTHER PUNCT, plus chars("abc") for a literal set.

Examples:
  python symbolset_adiabatic_textgen.py corpus.txt --prompt "Once upon a" --tokens 300
  python symbolset_adiabatic_textgen.py corpus.txt --allow "LOWER | SPACE | END"
  python symbolset_adiabatic_textgen.py corpus.txt --interactive --order 6
  python symbolset_adiabatic_textgen.py corpus.txt --sets

Only dependency: NumPy
"""

import argparse
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

BOS, EOS = "<bos>", ""


f = open(input("Filename: "), "r")
BUILTIN_CORPUS = f.read()

# --------------------------------------------------------------------------
# Symbol sets over the alphabet
# --------------------------------------------------------------------------

class SymbolSpace:
    """The universe U: an ordered alphabet, with sets stored as bit masks."""

    def __init__(self, alphabet: Iterable[str]):
        self.symbols: List[str] = sorted(set(alphabet))
        self.index: Dict[str, int] = {s: i for i, s in enumerate(self.symbols)}
        self.full_mask: int = (1 << len(self.symbols)) - 1

    def make(self, mask: int) -> "SymbolSet":
        return SymbolSet(self, mask & self.full_mask)

    def empty(self) -> "SymbolSet":
        return self.make(0)

    def universe(self) -> "SymbolSet":
        return self.make(self.full_mask)

    def chars(self, chars: Iterable[str]) -> "SymbolSet":
        mask = 0
        for c in chars:
            if c in self.index:
                mask |= 1 << self.index[c]
        return self.make(mask)

    def where(self, predicate) -> "SymbolSet":
        return self.chars(s for s in self.symbols if predicate(s))


class SymbolSet:
    """Immutable subset of the alphabet with logical set operations."""
    __slots__ = ("space", "mask")

    def __init__(self, space: SymbolSpace, mask: int):
        self.space, self.mask = space, mask

    def _check(self, other: "SymbolSet") -> None:
        if not isinstance(other, SymbolSet) or other.space is not self.space:
            raise TypeError("set operations require sets from the same alphabet")

    def __or__(self, o):  self._check(o); return self.space.make(self.mask | o.mask)   # union
    def __and__(self, o): self._check(o); return self.space.make(self.mask & o.mask)   # intersection
    def __xor__(self, o): self._check(o); return self.space.make(self.mask ^ o.mask)   # symmetric difference
    def __sub__(self, o): self._check(o); return self.space.make(self.mask & ~o.mask)  # difference
    def __invert__(self): return self.space.make(~self.mask)                           # complement

    def __contains__(self, symbol: str) -> bool:
        i = self.space.index.get(symbol)
        return i is not None and bool((self.mask >> i) & 1)

    def __len__(self) -> int:
        return bin(self.mask).count("1")

    def __bool__(self) -> bool:
        return self.mask != 0

    def members(self) -> List[str]:
        return [s for s in self.space.symbols if s in self]

    def __repr__(self) -> str:
        return "{" + "".join(repr(s)[1:-1] for s in self.members()) + "}"


def build_named_sets(space: SymbolSpace) -> Dict[str, SymbolSet]:
    """Base classes from predicates; derived classes purely from set operations."""
    named: Dict[str, SymbolSet] = {}
    named["ALL"] = space.universe()
    named["LOWER"] = space.where(str.islower)
    named["UPPER"] = space.where(str.isupper)
    named["DIGIT"] = space.where(str.isdigit)
    named["SPACE"] = space.where(str.isspace)
    named["VOWEL"] = space.where(lambda c: c.lower() in "aeiou")
    named["END"] = space.chars(".!?")
    named["PAUSE"] = space.chars(",;:")
    named["QUOTE"] = space.chars("\"'`\u2018\u2019\u201c\u201d")
    named["BRACKET"] = space.chars("()[]{}")
    # Derived with logical operations:
    named["LETTER"] = named["LOWER"] | named["UPPER"]
    named["CONSONANT"] = named["LETTER"] - named["VOWEL"]
    named["ALNUM"] = named["LETTER"] | named["DIGIT"]
    named["PUNCT"] = named["ALL"] - named["ALNUM"] - named["SPACE"]
    named["OTHER"] = named["PUNCT"] - (named["END"] | named["PAUSE"] | named["QUOTE"] | named["BRACKET"])
    return named


def eval_set_expr(expr: str, space: SymbolSpace, named: Dict[str, SymbolSet]) -> SymbolSet:
    """Evaluate e.g. "LETTER & ~VOWEL | SPACE" with | & ^ - ~ and chars('..')."""
    env = dict(named)
    env["chars"] = space.chars
    try:
        result = eval(expr, {"__builtins__": {}}, env)  # restricted namespace
    except Exception as exc:
        raise ValueError(f"Bad set expression {expr!r}: {exc}")
    if isinstance(result, str):
        result = space.chars(result)
    if not isinstance(result, SymbolSet):
        raise ValueError(f"Expression {expr!r} did not produce a symbol set")
    return result


# --------------------------------------------------------------------------
# Tokenisation: single symbols
# --------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return list(text)


def detokenize(tokens: Sequence[str]) -> str:
    return "".join(t for t in tokens if t not in (BOS, EOS))


# --------------------------------------------------------------------------
# N-gram model over symbols
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    token: str
    base_logp: float
    score: float
    consistency: float
    alignment: float
    probability: float = 0.0


class NGramModel:
    """Smoothed symbol n-gram model with backoff to shorter contexts."""

    def __init__(self, order: int = 5, alpha: float = 0.25):
        self.order = max(1, int(order))
        self.alpha = max(1e-9, float(alpha))
        self.counts = [defaultdict(Counter) for _ in range(self.order)]
        self.vocab = Counter()
        self.bigrams = defaultdict(Counter)  # for follow sets

    def train(self, text: str) -> None:
        tokens = [BOS] * (self.order - 1) + tokenize(text) + [EOS]
        for i in range(self.order - 1, len(tokens)):
            target = tokens[i]
            self.vocab[target] += 1
            if i > 0:
                self.bigrams[tokens[i - 1]][target] += 1
            for ctx_len in range(self.order):
                context = tuple(tokens[i - ctx_len:i]) if ctx_len else ()
                self.counts[ctx_len][context][target] += 1

    def candidates(self, context_tokens: Sequence[str], limit: int = 80) -> List[Tuple[str, float]]:
        vocab_size = max(1, len(self.vocab))
        context = list(context_tokens)
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
            prob = (count + self.alpha) / (total + self.alpha * vocab_size)
            ranked.append((token, math.log(max(prob, 1e-12))))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:max(1, limit)]


# --------------------------------------------------------------------------
# Non-isomorphic set fragments
# --------------------------------------------------------------------------

CLASS_ORDER = ["LOWER", "UPPER", "VOWEL", "CONSONANT", "DIGIT", "SPACE",
               "END", "PAUSE", "QUOTE", "BRACKET", "OTHER"]


class SetFragments:
    """
    Three set-valued lifts of (context, candidate), each living in a different
    space and none invertible into the others:
      g1  Follow sets            subsets of the alphabet (|U| bits)
      g2  class signatures       bit-vectors over symbol classes (11 bits)
      g3  recent-symbol window   subset of U built from the last w symbols
    """

    def __init__(self, space: SymbolSpace, named: Dict[str, SymbolSet],
                 model: NGramModel, window: int = 8, min_follow: int = 2):
        self.space, self.named, self.window = space, named, window
        self.empty = space.empty()

        # g1: Follow(s) = union of {t} for every t that followed s often enough
        self.follow: Dict[str, SymbolSet] = {}
        for s, counter in model.bigrams.items():
            strong = [t for t, c in counter.items() if c >= min_follow]
            chosen = strong if strong else list(counter)
            self.follow[s] = space.chars(chosen)

        # g2: class signature of each symbol as an int bit-vector
        self.signature: Dict[str, int] = {}
        for sym in space.symbols:
            bits = 0
            for k, name in enumerate(CLASS_ORDER):
                if sym in named[name]:
                    bits |= 1 << k
            self.signature[sym] = bits

    def _follow(self, sym: str) -> SymbolSet:
        return self.follow.get(sym, self.empty)

    def _sig(self, sym: str) -> int:
        return self.signature.get(sym, 0)

    def measures(self, context: Sequence[str], token: str) -> Tuple[float, float]:
        """Return (consistency D, alignment), both in [0, 1], via set operations."""
        last = context[-1] if context else BOS
        f_last, f_cand = self._follow(last), self._follow(token)
        union = f_last | f_cand

        # g1 fragments: membership (AND), symmetric difference (XOR) over union (OR)
        if f_last:
            m_member = 0.0 if token in f_last else 1.0
        else:
            m_member = 0.5
        m_xor = len(f_last ^ f_cand) / len(union) if union else 0.5
        jaccard = len(f_last & f_cand) / len(union) if union else 0.5

        # g2 fragment: class-signature Hamming distance (XOR on bit-vectors)
        m_class = bin(self._sig(last) ^ self._sig(token)).count("1") / len(CLASS_ORDER)

        # g3 fragment: how much of the candidate's follow set lies in the recent window
        win = self.space.chars(context[-self.window:])
        overlap = len(f_cand & win) / max(1, len(win))

        consistency = float(np.clip(0.25 * m_member + 0.45 * m_xor + 0.30 * m_class, 0.0, 1.0))
        alignment = float(np.clip(0.6 * jaccard + 0.4 * overlap, 0.0, 1.0))
        return consistency, alignment

    def score(self, context: Sequence[str], token: str, logp: float,
              lam: float) -> Tuple[float, float, float]:
        consistency, alignment = self.measures(context, token)
        fragment_score = 1.6 * alignment - 1.15 * consistency
        total = ((1.0 - lam) * (0.75 * logp + 0.25 * fragment_score)
                 + lam * (1.25 * logp + 0.85 * fragment_score))
        return total, consistency, alignment


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def adiabatic_lambda(step: int, total_steps: int) -> float:
    u = min(1.0, max(0.0, step / max(1, total_steps - 1)))
    return 3.0 * u * u - 2.0 * u * u * u


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    temp = max(1e-6, float(temperature))
    arr = np.asarray(values, dtype=float) / temp
    arr -= np.max(arr)
    exp = np.exp(np.clip(arr, -60, 60))
    return exp / max(float(exp.sum()), 1e-12)


def generate(
    model: NGramModel,
    fragments: SetFragments,
    prompt: str,
    token_count: int = 400,
    beam_size: int = 8,
    tau: float = 0.8,
    temperature: float = 0.85,
    seed: int = 7,
    allowed: SymbolSet = None,
    show_trace: bool = False,
) -> str:
    rng = random.Random(seed)
    output = tokenize(prompt) or ["T"]
    n_symbols = len(fragments.space.symbols)

    for step in range(max(0, token_count)):
        lam = adiabatic_lambda(step, max(1, token_count))
        ranked = [(t, lp) for t, lp in model.candidates(output, limit=max(n_symbols + 2, 40))
                  if t not in (BOS, EOS)]
        if allowed is not None:
            gated = [(t, lp) for t, lp in ranked if t in allowed]
            ranked = gated or ranked  # never dead-end on the gate
 

        scored: List[Candidate] = []
        for token, logp in ranked:
            score, consistency, alignment = fragments.score(output, token, logp, lam)
            if consistency <= tau:
                scored.append(Candidate(token, logp, score, consistency, alignment))

        if not scored:  # fallback: best raw candidate
            token, logp = ranked[0]
            score, consistency, alignment = fragments.score(output, token, logp, lam)
            scored = [Candidate(token, logp, score, consistency, alignment)]

        scored.sort(key=lambda c: c.score, reverse=True)
        beam = scored[:max(1, beam_size)]
        probs = softmax([c.score for c in beam], temperature)
        for cand, p in zip(beam, probs):
            cand.probability = float(p)

        chosen = beam[0] if temperature <= 0.01 else rng.choices(beam, weights=probs, k=1)[0]
        output.append(chosen.token)

        if show_trace:
            print(f"[{step:4d}] lam={lam:.2f} {chosen.token!r:>6} "
                  f"logp={chosen.base_logp:7.3f} D={chosen.consistency:.3f} "
                  f"align={chosen.alignment:.3f} p={chosen.probability:.3f}")

    return detokenize(output)


def main() -> None:
    ap = argparse.ArgumentParser(description="Character-level generator with symbol-set adiabatic scoring.")
    ap.add_argument("corpus", nargs="?", help="UTF-8 text file; built-in corpus if omitted.")
    ap.add_argument("--prompt", default="Once upon a", help="Initial text context.")
    ap.add_argument("--tokens", type=int, default=4000, help="Number of new symbols.")
    ap.add_argument("--order", type=int, default=5, help="Symbol n-gram order (default 5).")
    ap.add_argument("--alpha", type=float, default=0.25, help="Additive smoothing.")
    ap.add_argument("--beam", type=int, default=8, help="Candidates kept before sampling.")
    ap.add_argument("--tau", type=float, default=0.8, help="Max consistency score D (0..1).")
    ap.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature.")
    ap.add_argument("--window", type=int, default=8, help="Recent-symbol window size for g3.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed.")
    ap.add_argument("--allow", help="Set expression: only these symbols may be emitted.")
    ap.add_argument("--forbid", help="Set expression: these symbols are removed.")
    ap.add_argument("--sets", action="store_true", help="Print the named symbol sets and exit.")
    ap.add_argument("--interactive", action="store_true", help="Prompt repeatedly.")
    ap.add_argument("--trace", action="store_true", help="Per-symbol diagnostics.")
    ap.add_argument("--save", help="Write the last output to this file.")
    args = ap.parse_args()

    if args.corpus:
        path = Path(args.corpus)
        if not path.is_file():
            ap.error(f"Corpus file not found: {path}")
        try:
            corpus = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            ap.error(f"Could not decode corpus as UTF-8: {exc}")
    else:
        corpus = BUILTIN_CORPUS
    if not corpus:
        ap.error("Corpus is empty.")

    model = NGramModel(order=args.order, alpha=args.alpha)
    model.train(corpus)

    space = SymbolSpace(s for s in model.vocab if s not in (BOS, EOS))
    named = build_named_sets(space)

    if args.sets:
        for name, s in named.items():
            print(f"{name:10s} |{len(s):3d}|  {s!r}")
        return

    allowed = None
    if args.allow:
        allowed = eval_set_expr(args.allow, space, named)
    if args.forbid:
        banned = eval_set_expr(args.forbid, space, named)
        allowed = (allowed if allowed is not None else space.universe()) - banned

    fragments = SetFragments(space, named, model, window=args.window)
    tau = float(np.clip(args.tau, 0.0, 1.0))
    temperature = max(0.0, args.temperature)

    def run(prompt: str) -> str:
        return generate(model, fragments, prompt, args.tokens, args.beam, tau,
                        temperature, args.seed, allowed, args.trace)

    result = ""

    try:
        while True:
            prompt = input("USER: ")
            result = run(prompt)
            print("\n--- GENERATED TEXT ---\n")
            print(result, "\n")
    except (EOFError, KeyboardInterrupt):
        print()

if __name__ == "__main__":
    main()
