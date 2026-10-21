"""
Asymmetry map: block world vs mirror world
==========================================

Mapping only. No text is generated.

Block world   a cubic, SHA-256-shaped compression that cuts 64 rounds into macrosteps.
Mirror world  a reflected twin of it (word/bit reversal, left-handed sigmas, Ch/Maj swapped,
              constants and registers reversed, macrostep order reversed), plus two
              deliberate chiral defects: square-root constants and cubic(a + e).
              On the language side the mirror is the same corpus read right to left.

Three maps (all printed by default, or pick one with --only):

  macrosteps  per-macrostep differences for this prompt/seed: d_lam, d_tau, d_temp, d_len,
              anchor split, register Hamming distance (raw and reflected), plus the INFERRED
              defect field: handedness (smoothed over neighbours) and its gradient.
  sources     the planner asymmetry attributed one mirror operation at a time, with a control
              row (nothing switched on) that must read zero, averaged over many digests.
  language    forward vs time-reversed n-gram model: delta = log P_back(c|t) - log P_fwd(t|c),
              split into a frequency part (Bayes: log U(c) - log U(t)) and a residual.

Nothing here is a prediction or a physical model; both plans depend only on a digest of
(prompt, seed, corpus, order).

Examples:
  python asymmetry_map.py corpus.txt
  python asymmetry_map.py corpus.txt --only language --order 3 --top 12
  python asymmetry_map.py --only sources --map-seeds 64 --macrosteps 16
  python asymmetry_map.py corpus.txt --pure-mirror --only macrosteps

Only dependency: NumPy
"""

import argparse
import hashlib
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)

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
    no_space_before = set(".,!?;:%)]}\u00bb\u201d\u2019")
    no_space_after = set("([{\u00ab\u201c\u2018")
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
# Block world: cubic SHA-256-style planner
# ---------------------------------------------------------------------------

M32 = 0xFFFFFFFF


def rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & M32


def rotl(x: int, n: int) -> int:
    return ((x << n) | (x >> (32 - n))) & M32


def bitrev32(x: int) -> int:
    """Reflect a 32-bit word (the mirror operation on registers)."""
    return int(f"{x & M32:032b}"[::-1], 2)


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


# Mirror-image sigmas: every right rotation/shift becomes a left one.
def m_big_sigma0(x: int) -> int:
    return rotl(x, 2) ^ rotl(x, 13) ^ rotl(x, 22)


def m_big_sigma1(x: int) -> int:
    return rotl(x, 6) ^ rotl(x, 11) ^ rotl(x, 25)


def m_small_sigma0(x: int) -> int:
    return rotl(x, 7) ^ rotl(x, 18) ^ ((x << 3) & M32)


def m_small_sigma1(x: int) -> int:
    return rotl(x, 17) ^ rotl(x, 19) ^ ((x << 10) & M32)


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


def _sqrt_frac32(p: int) -> int:
    """First 32 fractional bits of the square root of p (exact)."""
    return math.isqrt(p << 64) & M32


_PRIMES = _first_primes(72)
K_CONST = [_cbrt_frac32(p) for p in _PRIMES[:64]]
H_INIT = [_cbrt_frac32(p) for p in _PRIMES[:8]]
K_SQRT = [_sqrt_frac32(p) for p in _PRIMES[:64]]   # chiral defect constants
H_SQRT = [_sqrt_frac32(p) for p in _PRIMES[:8]]


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
    """BLOCK WORLD. Plans macrosteps from a cubic SHA-256-like compression of the seed material."""

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

    def _boundaries(self) -> set:
        return {((i + 1) * self.rounds) // self.n for i in range(self.n)}

    def _compress(self, w: List[int]) -> Tuple[List[Tuple[int, ...]], Tuple[int, ...]]:
        a, b, c, d, e, f, g, h = H_INIT
        snapshots: List[Tuple[int, ...]] = []
        boundaries = self._boundaries()
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

    def build(self, prompt: str, seed: int, corpus_digest: str, order: int,
              vocab: Sequence[str], total_tokens: int) -> Tuple[List[Macrostep], Tuple[int, ...]]:
        """Unrotated macrosteps (in this world's own round order) and the final registers."""
        material = f"{prompt}\x1f{seed}\x1f{corpus_digest}\x1f{order}".encode("utf-8")
        snapshots, final = self._compress(self._message_words(material))
        anchors = self._anchors(vocab)

        weights, raw = [], []
        for (a, b, c, d, e, f, g, h) in snapshots:
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
        return steps, final

    def plan(self, prompt: str, seed: int, corpus_digest: str, order: int,
             vocab: Sequence[str], total_tokens: int) -> Tuple[List[Macrostep], int]:
        steps, final = self.build(prompt, seed, corpus_digest, order, vocab, total_tokens)
        start = self.choose_start(final)
        return steps[start:] + steps[:start], start


# ---------------------------------------------------------------------------
# Mirror world: reflected, deliberately chiral twin of the planner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MirrorFlags:
    """One switch per mirror operation, so each source of asymmetry can be isolated."""
    reflect_words: bool = True   # message words reversed in order and bit-reversed
    left_sigma: bool = True      # rotate/shift right -> left in every sigma
    swap_chmaj: bool = True      # Ch and Maj exchange roles in the round
    rev_consts: bool = True      # round constants run backwards
    rev_regs: bool = True        # initial registers loaded in reverse
    rev_order: bool = True       # macrostep sequence reversed
    sqrt_consts: bool = True     # CHIRAL: sqrt constants instead of cube-root
    add_cubic: bool = True       # CHIRAL: cubic(a + e) instead of cubic(a ^ e)


MIRROR_FACTORS = [f.name for f in fields(MirrorFlags)]
NO_FLAGS = MirrorFlags(**{n: False for n in MIRROR_FACTORS})   # control: identical to the block world


def only(name: str) -> MirrorFlags:
    return replace(NO_FLAGS, **{name: True})


class MirrorMacroPlanner(CubicMacroPlanner):
    """
    MIRROR WORLD. Same material, reflected machinery, each operation switchable.

    flags=None   : full mirror; chiral=True also turns on sqrt_consts and add_cubic.
    flags=NO_FLAGS reproduces the block world exactly (control).
    """

    def __init__(self, macrosteps: int = 8, chiral: bool = True, flags: Optional[MirrorFlags] = None):
        super().__init__(macrosteps)
        if flags is None:
            flags = MirrorFlags(sqrt_consts=chiral, add_cubic=chiral)
        self.flags = flags
        self.chiral = flags.sqrt_consts or flags.add_cubic
        k = list(K_SQRT if flags.sqrt_consts else K_CONST)
        h = list(H_SQRT if flags.sqrt_consts else H_INIT)
        self._k = k[::-1] if flags.rev_consts else k
        self._h = h[::-1] if flags.rev_regs else h

    def _message_words(self, material: bytes) -> List[int]:
        f = self.flags
        block = hashlib.sha256(material).digest()
        block += hashlib.sha256(block + material).digest()
        w = [int.from_bytes(block[i:i + 4], "big") for i in range(0, 64, 4)]
        if f.reflect_words:
            w = [bitrev32(x) for x in reversed(w)]
        s0 = m_small_sigma0 if f.left_sigma else small_sigma0
        s1 = m_small_sigma1 if f.left_sigma else small_sigma1
        for t in range(16, 64):
            w.append((s1(w[t - 2]) + w[t - 7] + s0(w[t - 15]) + w[t - 16]) & M32)
        return w

    def _compress(self, w: List[int]) -> Tuple[List[Tuple[int, ...]], Tuple[int, ...]]:
        f = self.flags
        S0 = m_big_sigma0 if f.left_sigma else big_sigma0
        S1 = m_big_sigma1 if f.left_sigma else big_sigma1
        f_e = maj if f.swap_chmaj else ch     # function applied to (e, f, g)
        f_a = ch if f.swap_chmaj else maj     # function applied to (a, b, c)
        regs = list(self._h)
        a, b, c, d, e, ff, g, h = regs
        snapshots: List[Tuple[int, ...]] = []
        boundaries = self._boundaries()
        for t in range(self.rounds):
            mix = (a + e) & M32 if f.add_cubic else (a ^ e)
            t1 = (h + S1(e) + f_e(e, ff, g) + self._k[t] + w[t] + cubic(mix)) & M32
            t2 = (S0(a) + f_a(a, b, c)) & M32
            h, g, ff, e, d, c, b, a = g, ff, e, (d + t1) & M32, c, b, a, (t1 + t2) & M32
            if (t + 1) in boundaries:
                snapshots.append((a, b, c, d, e, ff, g, h))
        final = tuple((x + y) & M32 for x, y in zip(regs, (a, b, c, d, e, ff, g, h)))
        return snapshots, final

    def mirror_plan(self, prompt: str, seed: int, corpus_digest: str, order: int,
                    vocab: Sequence[str], total_tokens: int, start: int) -> List[Macrostep]:
        """Mirror macrosteps, rotated by the SAME start as the block world."""
        steps, _ = self.build(prompt, seed, corpus_digest, order, vocab, total_tokens)
        if self.flags.rev_order:
            steps = steps[::-1]
        steps = [replace(ms, index=i) for i, ms in enumerate(steps)]
        return steps[start:] + steps[:start]


@dataclass
class AsymmetryCell:
    index: int
    d_lam: float          # block - mirror
    d_tau: float
    d_temp: float
    d_len: int
    anchor_split: bool    # the worlds disagree on the anchor word
    mirror_anchor: str
    hamming: float        # fraction of differing bits, block vs bit-REFLECTED mirror registers
    hamming_raw: float = 0.0  # same, without reflecting (0 for an identical world)
    handedness: float = 0.0   # inferred defect, smoothed over neighbouring macrosteps, in [-1, 1]
    gradient: float = 0.0     # inferred slope of the defect field


def map_asymmetry(block: Sequence[Macrostep], mirror: Sequence[Macrostep]) -> List[AsymmetryCell]:
    """Map the per-macrostep differences, then infer a smoothed defect field from them."""
    cells: List[AsymmetryCell] = []
    for b, m in zip(block, mirror):
        bits = sum(bin(x ^ bitrev32(y)).count("1") for x, y in zip(b.registers, m.registers))
        bits_raw = sum(bin(x ^ y).count("1") for x, y in zip(b.registers, m.registers))
        cells.append(AsymmetryCell(
            index=b.index,
            d_lam=b.lam_bias - m.lam_bias,
            d_tau=b.tau_scale - m.tau_scale,
            d_temp=b.temp_scale - m.temp_scale,
            d_len=b.tokens - m.tokens,
            anchor_split=(b.anchor != m.anchor),
            mirror_anchor=m.anchor,
            hamming=bits / (32.0 * len(b.registers)),
            hamming_raw=bits_raw / (32.0 * len(b.registers)),
        ))
    raw = [c.d_lam / 0.30 for c in cells]
    n = len(raw)
    for i, c in enumerate(cells):
        lo, hi = raw[max(0, i - 1)], raw[min(n - 1, i + 1)]
        c.handedness = float(np.clip(0.25 * lo + 0.5 * raw[i] + 0.25 * hi, -1.0, 1.0))
        c.gradient = float((hi - lo) / 2.0) if n > 1 else 0.0
    return cells


def attribute_planner_asymmetry(prompt: str, corpus_digest: str, order: int, vocab: Sequence[str],
                                macrosteps: int, total_tokens: int, seeds: int = 16) -> None:
    """
    Planner-side map, one source at a time. Each row switches ON a single mirror operation
    (all others off) and measures its difference from the block world, averaged over
    `seeds` different digests. Control must read zero.
    """
    rows = [("control (none)", NO_FLAGS)]
    rows += [(name, only(name)) for name in MIRROR_FACTORS]
    rows += [("ALL, pure mirror", MirrorFlags(sqrt_consts=False, add_cubic=False)),
             ("ALL, chiral", MirrorFlags())]
    print("PLANNER asymmetry by source (single operation on, vs block world)")
    print(f"  averaged over {seeds} digests, {macrosteps} macrosteps\n")
    print(f"  {'source':<18} {'ham.raw':>8} {'ham.refl':>8} {'|d_lam|':>8} {'|d_tau|':>8} {'|d_temp|':>9} {'anchor split':>13}")
    base = CubicMacroPlanner(macrosteps=macrosteps)
    for label, flags in rows:
        mp = MirrorMacroPlanner(macrosteps=macrosteps, flags=flags)
        ham, hraw, dl, dt, dtemp, split, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
        for sd in range(seeds):
            plan, start = base.plan(prompt, sd, corpus_digest, order, vocab, total_tokens)
            mir = mp.mirror_plan(prompt, sd, corpus_digest, order, vocab, total_tokens, start)
            for c in map_asymmetry(plan, mir):
                ham += c.hamming
                hraw += c.hamming_raw
                dl += abs(c.d_lam)
                dt += abs(c.d_tau)
                dtemp += abs(c.d_temp)
                split += int(c.anchor_split)
                n += 1
        print(f"  {label:<18} {hraw / n:8.3f} {ham / n:8.3f} {dl / n:8.3f} {dt / n:8.3f} {dtemp / n:9.3f} {split / n:12.0%}")
    print("\n  ham.raw: 0 = identical registers, ~0.5 = unrelated. ham.refl compares against the bit-reflected")
    print("  mirror, so even the control reads ~0.5: the function is not reflection-symmetric to begin with.")
    print("  The compression avalanches, so any single operation saturates the difference; sources are")
    print("  separated by which switch is on, not by how large the effect is.")


def map_language_asymmetry(fwd: "NGramModel", bwd: "NGramModel", top: int = 8) -> None:
    """
    Language-side map, separate from the planner. For every observed transition c -> t:
        delta    = log P_back(c | t) - log P_fwd(t | c)
        freq     = log U(c) - log U(t)        (Bayes: the part explained by token frequency alone)
        residual = delta - freq               (the genuinely directional part)
    """
    if fwd.order < 2 or not fwd.counts[1]:
        print("LANGUAGE asymmetry needs --order >= 2")
        return
    total = max(1, fwd.total_tokens)
    vsize = max(1, len(fwd.vocab))

    def logu(tok: str) -> float:
        return math.log((fwd.vocab.get(tok, 0) + fwd.alpha) / (total + fwd.alpha * vsize))

    rows = []
    for (c,), targets in fwd.counts[1].items():
        if c == "<bos>":
            continue
        for t in targets:
            delta = bwd.logprob([t], c) - fwd.logprob([c], t)
            freq = logu(c) - logu(t)
            rows.append((c, t, delta, freq, delta - freq))
    if not rows:
        print("LANGUAGE asymmetry: no transitions found")
        return
    d = np.array([r[2] for r in rows])
    fr = np.array([r[3] for r in rows])
    res = np.array([r[4] for r in rows])
    if np.std(d) > 1e-12 and np.std(fr) > 1e-12:
        corr = float(np.corrcoef(d, fr)[0, 1])
        slope = float(np.cov(d, fr, bias=True)[0, 1] / np.var(fr))
    else:
        corr, slope = float("nan"), float("nan")
    print("LANGUAGE asymmetry (forward vs time-reversed model, single-token contexts)")
    print(f"  transitions: {len(rows)}   mean|delta|={np.abs(d).mean():.3f}   "
          f"mean|frequency part|={np.abs(fr).mean():.3f}   mean|residual|={np.abs(res).mean():.3f}")
    print(f"  corr(delta, frequency part)={corr:+.3f}   slope={slope:.3f}  "
          f"(slope < 1: smoothing shrinks delta, so the residual is partly shrinkage, not direction)\n")
    ordered = sorted(rows, key=lambda r: r[2])
    print("  most forward-favoured (easy to say, hard to un-say):")
    for c, t, dl, _, rs in ordered[:top]:
        print(f"    {c!r:>14} -> {t!r:<14} delta={dl:+.2f} residual={rs:+.2f}")
    print("  most backward-favoured:")
    for c, t, dl, _, rs in ordered[::-1][:top]:
        print(f"    {c!r:>14} -> {t!r:<14} delta={dl:+.2f} residual={rs:+.2f}")
    biggest = sorted(rows, key=lambda r: abs(r[4]), reverse=True)[:top]
    print("  largest directional residuals (not explained by frequency):")
    for c, t, dl, _, rs in biggest:
        print(f"    {c!r:>14} -> {t!r:<14} residual={rs:+.2f}")


class NGramModel:
    """Smoothed n-gram model with backoff to shorter contexts."""

    reverse = False

    def __init__(self, order: int = 3, alpha: float = 0.25):
        self.order = max(1, int(order))
        self.alpha = max(1e-9, float(alpha))
        self.counts = [defaultdict(Counter) for _ in range(self.order)]
        self.vocab = Counter()
        self.total_tokens = 0
        self._cache: Dict[tuple, Tuple[Counter, float]] = {}

    def _sequence(self, text: str) -> List[str]:
        toks = tokenize(text)
        if self.reverse:
            toks = toks[::-1]
        return ["<bos>"] * (self.order - 1) + toks + ["<eos>"]

    def train(self, text: str) -> None:
        self._cache.clear()
        tokens = self._sequence(text)
        for i in range(self.order - 1, len(tokens)):
            target = tokens[i]
            self.vocab[target] += 1
            self.total_tokens += 1
            for ctx_len in range(self.order):
                context = tuple(tokens[i - ctx_len:i]) if ctx_len else ()
                self.counts[ctx_len][context][target] += 1

    def _mix(self, context_tokens: Sequence[str]) -> Tuple[Counter, float]:
        context = list(context_tokens)
        key = tuple(context[-(self.order - 1):]) if self.order > 1 else ()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        counter = Counter()
        for ctx_len in range(min(self.order - 1, len(context)), -1, -1):
            k = tuple(context[-ctx_len:]) if ctx_len else ()
            found = self.counts[ctx_len].get(k)
            if not found:
                continue
            weight = 1.0 + 0.7 * ctx_len
            for token, count in found.items():
                counter[token] += weight * count
            if ctx_len >= 1:
                break
        if not counter:
            counter.update(self.vocab)
        result = (counter, sum(counter.values()))
        self._cache[key] = result
        return result

    def logprob(self, context_tokens: Sequence[str], token: str) -> float:
        counter, total = self._mix(context_tokens)
        vocab_size = max(1, len(self.vocab))
        prob = (counter.get(token, 0.0) + self.alpha) / (total + self.alpha * vocab_size)
        return math.log(max(prob, 1e-12))

    def candidates(self, context_tokens: Sequence[str], limit: int = 80) -> List[Tuple[str, float]]:
        vocab_size = max(1, len(self.vocab))
        counter, total = self._mix(context_tokens)
        ranked = []
        for token, count in counter.items():
            prob = (count + self.alpha) / (total + self.alpha * vocab_size)
            ranked.append((token, math.log(max(prob, 1e-12))))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:max(1, limit)]


class ReverseNGramModel(NGramModel):
    """Mirror-world language model: the same corpus read right to left (time reversal)."""
    reverse = True




def print_asymmetry_map(cells: Sequence[AsymmetryCell]) -> None:
    print("Asymmetry map (block minus mirror):")
    print("  macro   d_lam    d_tau   d_temp  d_len  anchor  hamming  handed  grad")
    for c in cells:
        print(f"  {c.index:5d} {c.d_lam:+7.3f} {c.d_tau:+8.3f} {c.d_temp:+8.3f} {c.d_len:+6d}  "
              f"{'split' if c.anchor_split else 'same':>6}  {c.hamming:7.3f} {c.handedness:+7.2f} {c.gradient:+5.2f}",
              file=sys.stderr)
    mean_h = float(np.mean([c.hamming for c in cells])) if cells else 0.0
    print(f"  mean Hamming {mean_h:.3f} (0 = perfect mirror, ~0.5 = unrelated worlds)")




def main() -> None:
    parser = argparse.ArgumentParser(description="Map the asymmetry between a cubic block-world planner and its mirror world.")
    parser.add_argument("corpus", default="xaa", nargs="?", help="UTF-8 text corpus file; uses a small built-in corpus if the default file is absent.")
    parser.add_argument("--only", choices=["macrosteps", "sources", "language"], help="Print just one map.")
    parser.add_argument("--prompt", default="A process", help="Prompt used as planner seed material.")
    parser.add_argument("--seed", type=int, default=7, help="Seed used as planner seed material.")
    parser.add_argument("--tokens", type=int, default=800, help="Token budget the plans are sized for.")
    parser.add_argument("--order", type=int, default=4, help="N-gram order.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Additive smoothing value.")
    parser.add_argument("--macrosteps", type=int, default=8, help="Number of macrosteps (1..64).")
    parser.add_argument("--map-seeds", type=int, default=16, help="Digests averaged in the per-source map.")
    parser.add_argument("--pure-mirror", action="store_true", help="Macrostep map: reflection only, no chiral defects.")
    parser.add_argument("--top", type=int, default=8, help="Rows shown per list in the language map.")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if corpus_path.is_file():
        try:
            corpus_text = corpus_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            parser.error(f"Could not decode corpus as UTF-8: {exc}")
    elif args.corpus == parser.get_default("corpus"):
        print(f"[corpus '{args.corpus}' not found; using the built-in corpus]", file=sys.stderr)
        corpus_text = BUILTIN_CORPUS
    else:
        parser.error(f"Corpus file not found: {corpus_path}")

    corpus_digest = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()
    corpus_text = prepare_corpus_with_causal_ascent(corpus_text, alpha_mix=0.3)
    if not tokenize(corpus_text):
        parser.error("Corpus contains no recognizable tokens.")

    fwd = NGramModel(order=args.order, alpha=args.alpha)
    fwd.train(corpus_text)
    vocab = list(fwd.vocab)
    total = max(1, args.tokens)
    want = lambda name: args.only in (None, name)

    if want("macrosteps"):
        block = CubicMacroPlanner(macrosteps=args.macrosteps)
        plan, start = block.plan(args.prompt, args.seed, corpus_digest, args.order, vocab, total)
        mirror = MirrorMacroPlanner(macrosteps=args.macrosteps, chiral=not args.pure_mirror)
        msteps = mirror.mirror_plan(args.prompt, args.seed, corpus_digest, args.order, vocab, total, start)
        print(f"MACROSTEP asymmetry ({'pure' if args.pure_mirror else 'chiral'} mirror, start macrostep {start} of {block.n})")
        print_asymmetry_map(map_asymmetry(plan, msteps))
        print()

    if want("sources"):
        attribute_planner_asymmetry(args.prompt, corpus_digest, args.order, vocab,
                                    args.macrosteps, total, seeds=max(1, args.map_seeds))
        print()

    if want("language"):
        back = ReverseNGramModel(order=args.order, alpha=args.alpha)
        back.train(corpus_text)
        map_language_asymmetry(fwd, back, top=max(1, args.top))


if __name__ == "__main__":
    main()
