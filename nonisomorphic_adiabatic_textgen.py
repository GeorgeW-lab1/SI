"""
Non-isomorphic adiabatic text generator
=======================================

A runnable corpus-based text generator in which next-token candidates are
scored by an n-gram language model and filtered/reweighted by several
non-isomorphic process fragments.

Conceptual mapping:
  F       reference next-token distribution from the n-gram model
  g_k     fragment-specific transforms of candidate/context features
  I_k     local approximate inverse/lift (not a global inverse)
  lambda  smooth adiabatic continuation from exploratory to selective scoring
  D       proposed "diopathane" consistency score
  tau     maximum permitted D score for a candidate

This is an experimental algorithmic analogy, not an established physical
model. "Diopathane" is a user-defined name for the consistency constraint.

Examples:
  python nonisomorphic_adiabatic_textgen.py corpus.txt --prompt "once upon a" --tokens 80
  python nonisomorphic_adiabatic_textgen.py --prompt "The machine" --tokens 60
  python nonisomorphic_adiabatic_textgen.py corpus.txt --order 4 --beam 8 --tau 0.65

Only dependency: NumPy
Install: python -m pip install numpy
"""

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)

BUILTIN_CORPUS = (
    "The process begins with a simple observation of local flows. "
    "As patterns emerge, structural constraints shape the boundary conditions. "
    "Non-isomorphic transformations map states across mismatched dimensions. "
    "Causal ascent drives the system toward higher informational complexity. "
    "Through adiabatic continuation, the generator balances exploration and selective focus."
)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: Sequence[str]) -> str:
    """Join tokens into readable text without spaces before common punctuation."""
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


def prepare_corpus_with_causal_ascent(corpus_text: str, alpha_mix: float = 0.3) -> str:
    """
    Splits the corpus into individual sentences, applies frequency mixing 
    between global and local token distributions, and sequences them 
    to simulate causal ascent (ascending information entropy gradient).
    """
    # 1. Individualize sentences using standard punctuation boundaries
    raw_sentences = re.split(r'(?<=[.!?])\s+', corpus_text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return corpus_text

    # 2. Compute global corpus frequency distribution
    global_counter = Counter()
    sentence_tokens_list = [tokenize(s) for s in sentences]
    for tokens in sentence_tokens_list:
        global_counter.update(tokens)
    total_global = max(1, sum(global_counter.values()))

    # 3. Frequency mixing and causal ascent scoring per sentence
    scored_sentences = []
    for tokens in sentence_tokens_list:
        if not tokens:
            continue
        local_counter = Counter(tokens)
        total_local = max(1, sum(local_counter.values()))
        
        # Frequency mixing: interpolate local token probability with global background
        mixed_weights = {}
        for token in set(tokens):
            p_local = local_counter[token] / total_local
            p_global = global_counter[token] / total_global
            mixed_weights[token] = (1.0 - alpha_mix) * p_local * alpha_mix * p_global

        # Purporting causal ascent: calculate information entropy as a directional gradient metric
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in mixed_weights.values())
        scored_sentences.append((entropy, tokens))

    # 4. Sort sentences monotonically to establish causal ascent (ascending complexity/entropy)
    scored_sentences.sort(key=lambda item: item[0])

    # 5. Reconstruct the flattened corpus preserving the ascent trajectory
    flattened_tokens = []
    for _, tokens in scored_sentences:
        flattened_tokens.extend(tokens)

    return detokenize(flattened_tokens)


@dataclass
class Candidate:
    token: str
    base_logp: float
    score: float
    consistency: float
    alignment: float
    probability: float = 0.0


class NGramModel:
    """Smoothed n-gram model with backoff to shorter contexts."""

    def __init__(self, order: int = 3, alpha: float = 0.25):
        self.order = max(1, int(order))
        self.alpha = max(1e-9, float(alpha))
        self.counts = [defaultdict(Counter) for _ in range(self.order)]
        self.vocab = Counter()
        self.total_tokens = 0

    def train(self, text: str) -> None:
        tokens = ["<bos>"] * (self.order - 1) + tokenize(text) + ["<eos>"]
        if not tokens:
            return
        for i in range(self.order - 1, len(tokens)):
            target = tokens[i]
            self.vocab[target] += 1
            self.total_tokens += 1
            for ctx_len in range(self.order):
                context = tuple(tokens[i - ctx_len:i]) if ctx_len else ()
                self.counts[ctx_len][context][target] += 1

    def candidates(self, context_tokens: Sequence[str], limit: int = 80) -> List[Tuple[str, float]]:
        vocab_size = max(1, len(self.vocab))
        context = list(context_tokens)
        counter = Counter()
        # Blend backoff levels; a longer context receives more weight.
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


class NonIsomorphicFragments:
    """
    Fragment transforms candidate features into differently sized spaces.
    The transforms are intentionally not inverses of one another.
    """

    def __init__(self, seed: int = 17):
        rng = np.random.default_rng()#edited
        self.w_context = rng.normal(0, 0.7, size=(5, 3))
        self.w_word = rng.normal(0, 0.7, size=(5, 4))
        self.w_transition = rng.normal(0, 0.7, size=(5, 2))

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else np.zeros_like(v)

    @staticmethod
    def _hash_features(token: str, width: int) -> np.ndarray:
        """Stable token features; does not depend on Python's randomized hash."""
        raw = token.lower().encode("utf-8", errors="ignore")
        vals = np.zeros(width, dtype=float)
        if not raw:
            return vals
        for i, byte in enumerate(raw):
            vals[i % width] += ((byte % 31) - 15) / 15.0
            vals[(i * 3 + 1) % width] += ((byte % 17) - 8) / 8.0
        vals /= max(1, len(raw))
        vals[0] += min(len(raw), 20) / 20.0
        vals[1] += sum(chr(b).isalpha() for b in raw) / max(1, len(raw))
        return np.tanh(vals)

    def features(self, context: Sequence[str], token: str, logp: float, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        recent = list(context[-4:])
        ctx_vec = np.zeros(5, dtype=float)
        if recent:
            for i, word in enumerate(recent):
                f = self._hash_features(word, 5)
                ctx_vec += f * (0.65 ** (len(recent) - i - 1))
            ctx_vec /= max(1, len(recent))
        word_vec = self._hash_features(token, 5)
        transition_vec = np.array([
            np.tanh(logp),
            1.0 / (1.0 + rank),
            1.0 if token[:1].isupper() else 0.0,
            1.0 if token in ".,!?;:" else 0.0,
            min(len(token), 24) / 24.0
        ], dtype=float)
        
        g1 = np.tanh(ctx_vec @ self.w_context)
        g2 = np.tanh(word_vec @ self.w_word)
        g3 = np.tanh(transition_vec @ self.w_transition)
        return g1, g2, g3

    def score(self, context: Sequence[str], token: str, logp: float, rank: int, lam: float) -> Tuple[float, float]:
        g1, g2, g3 = self.features(context, token, logp, rank)
        summaries = np.array([
            np.linalg.norm(g1),
            np.linalg.norm(g2),
            np.linalg.norm(g3)
        ])
        mean = float(np.mean(summaries))
        spread = float(np.std(summaries) / (mean + 1e-9))
        consistency = float(np.clip(0.55 * spread + 0.45 * (1.0 - np.mean(summaries)), 0.0, 1.0))

        ref = self._unit(np.array([
            np.mean(g1),
            np.mean(g2),
            np.mean(g3),
            np.tanh(logp),
        ]))
        candidate = self._unit(np.array([
            g1[0],
            g2[0],
            g3[0],
            np.tanh(logp + 0.1 * rank)
        ]))
        cosine = float(np.dot(ref, candidate))
        alignment = (cosine + 1.0) / 2.0

        fragment_score = 1.6 * alignment - 1.15 * consistency
        total_score = (1.0 - lam) * (0.75 * logp + 0.25 * fragment_score) + lam * (1.25 * logp + 0.85 * fragment_score)
        return total_score, consistency


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
    prompt: str,
    token_count: int = 800,
    beam_size: int = 8,
    tau: float = 0.65,
    temperature: float = 0.85,
    seed: int = 7,
    show_trace: bool = False,
) -> str:
    rng = random.Random(seed)
    fragments = NonIsomorphicFragments(seed)
    output = tokenize(prompt)
    if not output:
        output = ["The"]

    for step in range(max(0, token_count)):
        lam = adiabatic_lambda(step, max(1, token_count))
        ranked = model.candidates(output, limit=max(40, beam_size * 8))
        scored: List[Candidate] = []

        for rank, (token, logp) in enumerate(ranked):
            score, consistency = fragments.score(output, token, logp, rank, lam)
            alignment = float(np.clip((score - logp) / 2.5 + 0.5, 0.0, 1.0))
            if consistency <= tau:
                scored.append(Candidate(token, logp, score, consistency, alignment))

        if not scored:
            token, logp = ranked[0]
            score, consistency = fragments.score(output, token, logp, 0, lam)
            scored = [Candidate(token, logp, score, consistency, 0.5)]

        scored.sort(key=lambda c: c.score, reverse=True)
        beam = scored[:max(1, beam_size)]
        probs = softmax([c.score for c in beam], temperature)
        for candidate, prob in zip(beam, probs):
            candidate.probability = float(prob)

        chosen = beam[0] if temperature <= 0.01 else rng.choices(beam, weights=probs, k=1)[0]
        output.append(chosen.token)

    return detokenize(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="N-gram text generator with non-isomorphic adiabatic fragment scoring.")
    parser.add_argument("corpus", nargs="?", help="UTF-8 text corpus file; uses a small built-in corpus if omitted.")
    parser.add_argument("--prompt", default="A process", help="Initial text context.")
    parser.add_argument("--tokens", type=int, default=800, help="Maximum number of new tokens.")
    parser.add_argument("--order", type=int, default=3, help="N-gram order (default: 3).")
    parser.add_argument("--alpha", type=float, default=0.25, help="Additive smoothing value.")
    parser.add_argument("--beam", type=int, default=18, help="Number of candidate next words retained before sampling.")
    parser.add_argument("--tau", type=float, default=0.65, help="Maximum allowed fragment consistency score (0..1).")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature; near zero selects the top candidate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable generation.")
    parser.add_argument("--trace", action="store_true", help="Print per-token scoring diagnostics.")
    parser.add_argument("--save", help="Optional output text file.")
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

    # Apply individual sentence preparation via frequency mixing and causal ascent
    corpus_text = prepare_corpus_with_causal_ascent(corpus_text, alpha_mix=0.3)

    if not tokenize(corpus_text):
        parser.error("Corpus contains no recognizable tokens.")

    model = NGramModel(order=args.order, alpha=args.alpha)
    model.train(corpus_text)

    result = generate(
        model=model,
        prompt=args.prompt,
        token_count=args.tokens,
        beam_size=args.beam,
        tau=float(np.clip(args.tau, 0.0, 1.0)),
        temperature=max(0.0, args.temperature),
        seed=args.seed,
        show_trace=args.trace,
    )

    print("\n--- GENERATED TEXT ---\n")
    print(result)
    
    if args.save:
        Path(args.save).write_text(result, encoding="utf-8")
        print(f"\n[Saved output to {args.save}]")


if __name__ == "__main__":
    main()
