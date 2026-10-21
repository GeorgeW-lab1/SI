"""
Non-isomorphic adiabatic text generator with a cubic Ch/Maj macrostep planner
=============================================================================

Layers
------
  1. NGramModel                reference next-token distribution F
  2. NonIsomorphicFragments    g_k transforms, consistency D, tau filter
  3. adiabatic_lambda          smooth exploratory -> selective continuation
  4. CubicMacroPlanner (new)   SHA-256-like compression that plans macrosteps

About the planner
-----------------
The planner is a SHA-256-shaped compression function:
  * round constants K are the fractional parts of the CUBE ROOTS of the first
    64 primes (exact integer arithmetic), as in SHA-256;
  * the initial registers are the fractional parts of the cube roots of the
    first 8 primes (SHA-256 uses square roots; cubes are used here to keep the
    whole construction cubic);
  * every round adds an explicit cubic term  cubic(a ^ e) = (a ^ e)^3 mod 2^32
    on top of the usual Ch / Maj / Sigma mixing.

Ch(x,y,z)  = (x & y) ^ (~x & z)          "choice"
Maj(x,y,z) = (x & y) ^ (x & z) ^ (y & z) "majority"

The 64 rounds are cut into N groups. After each group the registers are
snapshotted and each snapshot becomes one MACROSTEP (a plan segment: relative
length, lambda bias, tau scale, temperature scale, anchor word).

The BEGINNING macrostep is chosen by three Ch-voters whose bitwise Maj decides
the start index.

This is NOT a prediction. The plan depends only on a digest of
(prompt, seed, corpus, model order). It never reads n-gram probabilities, and
it is fixed before the first token is generated. It is a deterministic,
avalanche-mixed schedule, not a forecast of the text.

This is an experimental algorithmic analogy, not an established physical
model. "Diopathane" is a user-defined name for the consistency constraint.

Examples:
  python nonisomorphic_adiabatic_textgen.py corpus.txt --prompt "once upon a" --tokens 80
  python nonisomorphic_adiabatic_textgen.py --prompt "The machine" --tokens 60 --macrosteps 8 --plan-trace
  python nonisomorphic_adiabatic_textgen.py corpus.txt --no-plan

Only dependency: NumPy
"""

import argparse
import hashlib
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

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
    Split the corpus into sentences, mix local and global token frequencies,
    and order sentences by ascending entropy (the "causal ascent" ordering).
    """
    raw_sentences = re.split(r"(?<=[.!?])\s+", corpus_text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return corpus_text

    global_counter = Counter()
    sentence_tokens_list = [tokenize(s) for s in sentences]
    for tokens in sentence_tokens_list:
        global_counter.update(tokens)
    total_global = max(1, sum(global_counter.values()))

    scored_sentences = []
    for tokens in sentence_tokens_list:
        if not tokens:
            continue
        local_counter = Counter(tokens)
        total_local = max(1, sum(local_counter.values()))

        mixed_weights = {}
        for token in set(tokens):
            p_local = local_counter[token] / total_local
            p_global = global_counter[token] / total_global
            # Fixed: this is an interpolation (+), not a product (*).
            mixed_weights[token] = (1.0 - alpha_mix) * p_local + alpha_mix * p_global

        norm = max(sum(mixed_weights.values()), 1e-12)
        entropy = -sum((w / norm) * math.log(max(w / norm, 1e-12)) for w in mixed_weights.values())
        scored_sentences.append((entropy, tokens))

    scored_sentences.sort(key=lambda item: item[0])

    flattened_tokens: List[str] = []
    for _, tokens in scored_sentences:
        flattened_tokens.extend(tokens)
    return detokenize(flattened_tokens)


# ---------------------------------------------------------------------------
# Cubic SHA-256-style planner
# ---------------------------------------------------------------------------

M32 = 0xFFFFFFFF


def rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & M32


def ch(x: int, y: int, z: int) -> int:
    """Choice: bits of x select between y (1) and z (0)."""
    return ((x & y) ^ (~x & z)) & M32


def maj(x: int, y: int, z: int) -> int:
    """Majority: each output bit is the majority vote of the three input bits."""
    return ((x & y) ^ (x & z) ^ (y & z)) & M32


def big_sigma0(x: int) -> int:
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def big_sigma1(x: int) -> int:
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def small_sigma0(x: int) -> int:
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)


def small_sigma1(x: int) -> int:
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)


def cubic(x: int) -> int:
    """Explicit cubic nonlinearity modulo 2^32."""
    return (x * x * x) & M32


def _first_primes(n: int) -> List[int]:
    primes: List[int] = []
    cand = 2
    while len(primes) < n:
        if all(cand % p for p in primes if p * p <= cand):
            primes.append(cand)
        cand += 1
    return primes


def _icbrt(n: int) -> int:
    """Exact floor of the integer cube root."""
    lo, hi = 0, 1 << (n.bit_length() // 3 + 2)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid * mid * mid <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _cbrt_frac32(p: int) -> int:
    """First 32 fractional bits of the cube root of p (exact)."""
    return _icbrt(p << 96) & M32


_PRIMES = _first_primes(72)
K_CONST = [_cbrt_frac32(p) for p in _PRIMES[:64]]
H_INIT = [_cbrt_frac32(p) for p in _PRIMES[:8]]


@dataclass
class Macrostep:
    index: int            # position in the unrotated plan
    tokens: int           # token budget assigned to this macrostep
    lam_bias: float       # additive shift of the adiabatic lambda
    tau_scale: float      # multiplier on tau
    temp_scale: float     # multiplier on temperature
    anchor: str           # corpus word given a small score bonus early in the step
    registers: Tuple[int, ...] = ()


class CubicMacroPlanner:
    """Plans macrosteps from a cubic SHA-256-like compression of the seed material."""

    def __init__(self, macrosteps: int = 8, rounds: int = 64):
        self.n = max(1, min(int(macrosteps), 64))
        self.rounds = 64  # fixed by the K table; macrosteps partition the rounds
        del rounds

    @staticmethod
    def _message_words(material: bytes) -> List[int]:
        block = hashlib.sha256(material).digest()
        block += hashlib.sha256(block + material).digest()  # 64 bytes -> 16 words
        w = [int.from_bytes(block[i:i + 4], "big") for i in range(0, 64, 4)]
        for t in range(16, 64):
            w.append((small_sigma1(w[t - 2]) + w[t - 7] + small_sigma0(w[t - 15]) + w[t - 16]) & M32)
        return w

    def _compress(self, w: List[int]) -> Tuple[List[Tuple[int, ...]], Tuple[int, ...]]:
        a, b, c, d, e, f, g, h = H_INIT
        snapshots: List[Tuple[int, ...]] = []
        boundaries = {((i + 1) * self.rounds) // self.n for i in range(self.n)}
        for t in range(self.rounds):
            t1 = (h + big_sigma1(e) + ch(e, f, g) + K_CONST[t] + w[t] + cubic(a ^ e)) & M32
            t2 = (big_sigma0(a) + maj(a, b, c)) & M32
            h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M32, c, b, a, (t1 + t2) & M32
            if (t + 1) in boundaries:
                snapshots.append((a, b, c, d, e, f, g, h))
        final = tuple((x + y) & M32 for x, y in zip(H_INIT, (a, b, c, d, e, f, g, h)))
        return snapshots, final

    @staticmethod
    def _anchors(vocab: Sequence[str]) -> List[str]:
        words = sorted(t for t in vocab if t.isalpha() and len(t) >= 4 and not t.startswith("<"))
        return words or sorted(t for t in vocab if not t.startswith("<")) or ["the"]

    def choose_start(self, final: Tuple[int, ...]) -> int:
        """
        Three Choice voters (different register triples). The bitwise Majority
        of their 8-bit votes decides the beginning macrostep.
        """
        a, b, c, d, e, f, g, h = final
        v1 = ch(e, f, g) & 0xFF
        v2 = ch(a, b, c) & 0xFF
        v3 = ch(h, d, cubic(a ^ e)) & 0xFF
        return (maj(v1, v2, v3) & 0xFF) % self.n

    def plan(self, prompt: str, seed: int, corpus_digest: str, order: int,
             vocab: Sequence[str], total_tokens: int) -> Tuple[List[Macrostep], int]:
        material = f"{prompt}\x1f{seed}\x1f{corpus_digest}\x1f{order}".encode("utf-8")
        snapshots, final = self._compress(self._message_words(material))
        start = self.choose_start(final)
        anchors = self._anchors(vocab)

        weights, raw = [], []
        for i, (a, b, c, d, e, f, g, h) in enumerate(snapshots):
            weights.append(1.0 + (maj(a, b, c) & 0xFF) / 255.0)             # 1..2
            lam_bias = ((ch(e, f, g) & 0xFFFF) / 65535.0 - 0.5) * 0.30      # -0.15..0.15
            tau_scale = 0.88 + 0.12 * ((maj(d, e, f) & 0xFF) / 255.0)       # 0.88..1.0
            temp_scale = 0.80 + 0.40 * ((ch(a, g, h) & 0xFF) / 255.0)       # 0.8..1.2
            anchor = anchors[cubic(d ^ h) % len(anchors)]
            raw.append((lam_bias, tau_scale, temp_scale, anchor, (a, b, c, d, e, f, g, h)))

        wsum = sum(weights)
        lengths = [max(1, int(round(total_tokens * w / wsum))) for w in weights]
        lengths[-1] = max(1, lengths[-1] + total_tokens - sum(lengths))

        steps = [Macrostep(i, lengths[i], *raw[i][:4], registers=raw[i][4]) for i in range(len(raw))]
        rotated = steps[start:] + steps[:start]
        return rotated, start


def macrostep_for(plan: Sequence[Macrostep], step: int) -> Tuple[Macrostep, int, int]:
    """Return (macrostep, offset within it, its length) for a global token step."""
    cursor = 0
    for ms in plan:
        if step < cursor + ms.tokens:
            return ms, step - cursor, ms.tokens
        cursor += ms.tokens
    last = plan[-1]
    return last, last.tokens - 1, last.tokens


# ---------------------------------------------------------------------------
# Language model and fragments
# ---------------------------------------------------------------------------

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
    """Transforms candidate features into differently sized spaces (not inverses)."""

    def __init__(self, seed: int = 17):
        rng = np.random.default_rng(seed)  # now honours --seed
        self.w_context = rng.normal(0, 0.7, size=(5, 3))
        self.w_word = rng.normal(0, 0.7, size=(5, 4))
        self.w_transition = rng.normal(0, 0.7, size=(5, 2))

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else np.zeros_like(v)

    @staticmethod
    def _hash_features(token: str, width: int) -> np.ndarray:
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

    def features(self, context: Sequence[str], token: str, logp: float, rank: int):
        recent = list(context[-4:])
        ctx_vec = np.zeros(5, dtype=float)
        if recent:
            for i, word in enumerate(recent):
                ctx_vec += self._hash_features(word, 5) * (0.65 ** (len(recent) - i - 1))
            ctx_vec /= max(1, len(recent))
        word_vec = self._hash_features(token, 5)
        transition_vec = np.array([
            np.tanh(logp),
            1.0 / (1.0 + rank),
            1.0 if token[:1].isupper() else 0.0,
            1.0 if token in ".,!?;:" else 0.0,
            min(len(token), 24) / 24.0,
        ], dtype=float)
        g1 = np.tanh(ctx_vec @ self.w_context)
        g2 = np.tanh(word_vec @ self.w_word)
        g3 = np.tanh(transition_vec @ self.w_transition)
        return g1, g2, g3

    def score(self, context: Sequence[str], token: str, logp: float, rank: int, lam: float) -> Tuple[float, float]:
        g1, g2, g3 = self.features(context, token, logp, rank)
        summaries = np.array([np.linalg.norm(g1), np.linalg.norm(g2), np.linalg.norm(g3)])
        mean = float(np.mean(summaries))
        spread = float(np.std(summaries) / (mean + 1e-9))
        consistency = float(np.clip(0.55 * spread + 0.45 * (1.0 - np.mean(summaries)), 0.0, 1.0))

        ref = self._unit(np.array([np.mean(g1), np.mean(g2), np.mean(g3), np.tanh(logp)]))
        cand = self._unit(np.array([g1[0], g2[0], g3[0], np.tanh(logp + 0.1 * rank)]))
        alignment = (float(np.dot(ref, cand)) + 1.0) / 2.0

        fragment_score = 1.6 * alignment - 1.15 * consistency
        total = (1.0 - lam) * (0.75 * logp + 0.25 * fragment_score) + lam * (1.25 * logp + 0.85 * fragment_score)
        return total, consistency


def adiabatic_lambda(step: int, total_steps: int) -> float:
    u = min(1.0, max(0.0, step / max(1, total_steps - 1)))
    return 3.0 * u * u - 2.0 * u * u * u


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    temp = max(1e-6, float(temperature))
    arr = np.asarray(values, dtype=float) / temp
    arr -= np.max(arr)
    exp = np.exp(np.clip(arr, -60, 60))
    return exp / max(float(exp.sum()), 1e-12)


ANCHOR_BONUS = 0.5


def generate(
    model: NGramModel,
    prompt: str,
    token_count: int = 800,
    beam_size: int = 8,
    tau: float = 0.65,
    temperature: float = 0.85,
    seed: int = 7,
    show_trace: bool = False,
    plan: Sequence[Macrostep] = (),
) -> str:
    rng = random.Random(seed)
    fragments = NonIsomorphicFragments(seed)
    output = tokenize(prompt) or ["The"]
    last_macro = -1

    for step in range(max(0, token_count)):
        lam = adiabatic_lambda(step, max(1, token_count))
        tau_eff, temp_eff, anchor = tau, temperature, None

        if plan:
            ms, offset, length = macrostep_for(plan, step)
            lam = float(np.clip(lam + ms.lam_bias, 0.0, 1.0))
            tau_eff = float(np.clip(tau * ms.tau_scale, 0.0, 1.0))
            temp_eff = temperature * ms.temp_scale
            if offset < max(1, length // 3):
                anchor = ms.anchor
            if show_trace and ms.index != last_macro:
                print(f"[macrostep {ms.index}] tokens={length} lam_bias={ms.lam_bias:+.3f} "
                      f"tau*={ms.tau_scale:.3f} temp*={ms.temp_scale:.3f} anchor={ms.anchor!r}",
                      file=sys.stderr)
                last_macro = ms.index

        ranked = model.candidates(output, limit=max(40, beam_size * 8))
        scored: List[Candidate] = []
        for rank, (token, logp) in enumerate(ranked):
            score, consistency = fragments.score(output, token, logp, rank, lam)
            if anchor is not None and token == anchor:
                score += ANCHOR_BONUS
            alignment = float(np.clip((score - logp) / 2.5 + 0.5, 0.0, 1.0))
            if consistency <= tau_eff:
                scored.append(Candidate(token, logp, score, consistency, alignment))

        if not scored:
            token, logp = ranked[0]
            score, consistency = fragments.score(output, token, logp, 0, lam)
            scored = [Candidate(token, logp, score, consistency, 0.5)]

        scored.sort(key=lambda c: c.score, reverse=True)
        beam = scored[:max(1, beam_size)]
        probs = softmax([c.score for c in beam], temp_eff)
        for candidate, prob in zip(beam, probs):
            candidate.probability = float(prob)

        chosen = beam[0] if temp_eff <= 0.01 else rng.choices(beam, weights=probs, k=1)[0]
        output.append(chosen.token)

    return detokenize(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="N-gram generator with non-isomorphic adiabatic scoring and a cubic Ch/Maj macrostep planner.")
    parser.add_argument("corpus",default="xaa", nargs="?", help="UTF-8 text corpus file; uses a small built-in corpus if omitted.")
    parser.add_argument("--prompt", default="A process", help="Initial text context.")
    parser.add_argument("--tokens", type=int, default=800, help="Maximum number of new tokens.")
    parser.add_argument("--order", type=int, default=4, help="N-gram order.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Additive smoothing value.")
    parser.add_argument("--beam", type=int, default=18, help="Candidates retained before sampling.")
    parser.add_argument("--tau", type=float, default=0.95, help="Maximum allowed fragment consistency score (0..1).")
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature; near zero selects the top candidate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable generation.")
    parser.add_argument("--macrosteps", type=int, default=8, help="Number of macrosteps the 64 rounds are cut into (1..64).")
    parser.add_argument("--no-plan", action="store_true", help="Disable the macrostep planner.")
    parser.add_argument("--plan-trace", action="store_true", help="Print the plan and the start-macrostep vote.")
    parser.add_argument("--trace", action="store_true", help="Print macrostep transitions during generation.")
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

    corpus_digest = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()
    corpus_text = prepare_corpus_with_causal_ascent(corpus_text, alpha_mix=0.3)

    if not tokenize(corpus_text):
        parser.error("Corpus contains no recognizable tokens.")

    model = NGramModel(order=args.order, alpha=args.alpha)
    model.train(corpus_text)

    plan: List[Macrostep] = []
    if not args.no_plan:
        planner = CubicMacroPlanner(macrosteps=args.macrosteps)
        plan, start = planner.plan(args.prompt, args.seed, corpus_digest, args.order,
                                   list(model.vocab), max(1, args.tokens))
        if args.plan_trace:
            print(f"Start macrostep (Maj of three Ch voters): {start} of {planner.n}", file=sys.stderr)
            for ms in plan:
                print(f"  macrostep {ms.index}: tokens={ms.tokens:4d} lam_bias={ms.lam_bias:+.3f} "
                      f"tau*={ms.tau_scale:.3f} temp*={ms.temp_scale:.3f} anchor={ms.anchor!r}",
                      file=sys.stderr)

    result = generate(
        model=model,
        prompt=input("USER: "),
        token_count=args.tokens,
        beam_size=args.beam,
        tau=float(np.clip(args.tau, 0.0, 1.0)),
        temperature=max(0.0, args.temperature),
        seed=args.seed,
        show_trace=args.trace,
        plan=plan,
    )

    print("\n--- GENERATED TEXT ---\n")
    print(result)

    if args.save:
        Path(args.save).write_text(result, encoding="utf-8")
        print(f"\n[Saved output to {args.save}]")


if __name__ == "__main__":
    main()
