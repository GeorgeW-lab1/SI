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
import dataclasses
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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


# ---------------------------------------------------------------------------
# Sorted linked list of sentences, each linked to a cognitive instruction
# ---------------------------------------------------------------------------

PUNCT = set(".,!?;:")
END = {".", "!", "?"}
SPECIAL = {"<bos>", "<eos>"}


def _content(tok: str) -> bool:
    return len(tok) > 3 and tok not in PUNCT


# ---------------------------------------------------------------------------
# Markovlets: tiny, self-contained Markov chains learned from natural text.
#
# A markovlet knows only the transitions of the fragment it was learned from
# (a sentence, or a module's output). Because every transition comes from real
# text, anything a markovlet walk emits is a continuation that actually
# occurred. Markovlets compose: merge / restrict / join build new markovlets
# out of old ones without ever touching the source text again.
# ---------------------------------------------------------------------------

MARKOV_ORDER = 3          # order-1 chains share contexts, so fragments can recombine


def _norm(d: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not d:
        return {}
    total = sum(d.values())
    return {t: c / total for t, c in d.items()} if total > 0 else {}


def _mix(a: Dict[str, float], b: Dict[str, float], w: float) -> Dict[str, float]:
    a, b = _norm(a), _norm(b)
    if a and b:
        return {t: (1.0 - w) * a.get(t, 0.0) + w * b.get(t, 0.0) for t in set(a) | set(b)}
    return dict(a or b)


class Markovlet:
    def __init__(self, name: str, order: int = MARKOV_ORDER,
                 trans: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
                 starts: Optional[Dict[str, float]] = None,
                 sources: Optional[List[str]] = None):
        self.name = name
        self.order = order
        self.trans = trans if trans is not None else {}
        self.starts = starts if starts is not None else {}
        self.sources = sources if sources is not None else [name]
        self.bridges: List[str] = []          # shared tokens found by join()

    # -- construction -------------------------------------------------------
    @classmethod
    def from_tokens(cls, name: str, tokens: Sequence[str], order: int = MARKOV_ORDER) -> "Markovlet":
        m = cls(name, order)
        toks = list(tokens)
        if not toks:
            return m
        m.starts[toks[0]] = 1.0
        for i in range(1, len(toks)):
            for n in range(1, order + 1):
                if i - n < 0:
                    break
                d = m.trans.setdefault(tuple(toks[i - n:i]), {})
                d[toks[i]] = d.get(toks[i], 0.0) + 1.0
        return m

    # -- queries ------------------------------------------------------------
    @property
    def vocab(self) -> set:
        v = set(self.starts)
        for ctx, d in self.trans.items():
            v.update(ctx)
            v.update(d)
        return v

    def is_empty(self) -> bool:
        return not self.trans

    def successors(self, history: Sequence[str]) -> Dict[str, float]:
        """Longest-context lookup with backoff inside this markovlet only."""
        for n in range(min(self.order, len(history)), 0, -1):
            d = self.trans.get(tuple(history[-n:]))
            if d:
                return d
        return {}

    # -- algebra ------------------------------------------------------------
    def merge(self, other: "Markovlet", w: float = 0.5, name: Optional[str] = None) -> "Markovlet":
        out = Markovlet(name or f"merge({self.name},{other.name})",
                        max(self.order, other.order), sources=self.sources + other.sources)
        for ctx in set(self.trans) | set(other.trans):
            out.trans[ctx] = _mix(self.trans.get(ctx, {}), other.trans.get(ctx, {}), w)
        out.starts = _mix(self.starts, other.starts, w)
        return out

    def restrict(self, keep, name: Optional[str] = None) -> "Markovlet":
        """Sub-markovlet whose transitions only lead to tokens passing `keep`."""
        out = Markovlet(name or f"restrict({self.name})", self.order, sources=list(self.sources))
        for ctx, d in self.trans.items():
            nd = {t: c for t, c in d.items() if keep(t)}
            if nd:
                out.trans[ctx] = nd
        out.starts = {t: c for t, c in self.starts.items() if keep(t)}
        return out

    def join(self, other: "Markovlet", name: Optional[str] = None) -> "Markovlet":
        """Overlap join: chains connect through tokens they share, and every
        sentence-end of this chain gets an edge into the other's start."""
        out = self.merge(other, 0.5, name or f"join({self.name},{other.name})")
        for e in END:
            if e in self.vocab:
                d = out.trans.setdefault((e,), {})
                for t, c in _norm(other.starts).items():
                    d[t] = d.get(t, 0.0) + c
        out.bridges = sorted(self.vocab & other.vocab)
        return out

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name, "order": self.order, "sources": self.sources,
            "starts": self.starts, "bridges": self.bridges,
            "trans": {" ".join(ctx): d for ctx, d in self.trans.items()},
        }

    def stats(self) -> Dict[str, object]:
        return {"name": self.name, "contexts": len(self.trans), "vocab": len(self.vocab),
                "sources": len(self.sources), "bridges": len(self.bridges)}


def fold_merge(chains: Sequence[Markovlet], name: str) -> Markovlet:
    """Equal-weight merge of any number of markovlets."""
    chains = list(chains)
    out = chains[0]
    for i, c in enumerate(chains[1:], start=2):
        out = out.merge(c, 1.0 / i)
    out = Markovlet(name, out.order, out.trans, out.starts, out.sources)
    return out


# ---------------------------------------------------------------------------
# Modules: each one CONSTRUCTS a markovlet from its inputs (markovlet algebra),
# walks it, and publishes the markovlet it learned from its own output.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModuleResult:
    module: str
    provides: str
    node_idx: int
    pass_no: int
    tokens: Tuple[str, ...]
    text: str
    markovlet: Markovlet              # markovlet learned from this module's output
    chain: str                        # provenance of the chain that was walked
    natural_rate: float               # share of in-segment bigrams seen in the corpus
    novelty: float                    # share of tokens NOT inside a verbatim corpus 4-gram
    features: Dict[str, object]
    consumed: Tuple[str, ...]
    missing: Tuple[str, ...]
    injected: int                     # unused, kept for report compatibility
    fallbacks: int                    # steps where the chain dead-ended
    closed: bool


def result_to_dict(r: ModuleResult) -> Dict[str, object]:
    d = {k: getattr(r, k) for k in ("module", "provides", "node_idx", "pass_no", "text",
                                     "chain", "natural_rate", "novelty", "features", "consumed",
                                     "missing", "fallbacks", "closed")}
    d["tokens"] = list(r.tokens)
    d["markovlet"] = r.markovlet.to_dict()
    return d


@dataclass
class ModuleInput:
    history: Tuple[str, ...]
    node: "SentenceNode"
    nxt: Optional["SentenceNode"]
    upstream: Dict[str, ModuleResult]
    all_vocab: frozenset
    ngrams: Dict[int, set] = field(default_factory=dict)   # corpus n-grams by length
    tokens: List[str] = field(default_factory=list)
    steps: int = 0
    state: Dict[str, object] = field(default_factory=dict)

    def up(self, key: str) -> Optional[ModuleResult]:
        return self.upstream.get(key)

    def prev(self) -> str:
        if self.tokens:
            return self.tokens[-1]
        return self.history[-1] if self.history else ""


class Module:
    name = "NOOP"
    provides = "noop"
    requires: Tuple[str, ...] = ()
    min_steps = 4
    max_steps = 20
    max_copy: Optional[int] = None    # longest verbatim corpus run allowed (None = unlimited)

    def admit(self, io: ModuleInput, token: str) -> bool:
        """Hard filter applied to every candidate, whatever its source."""
        return True

    def construct(self, io: ModuleInput) -> Markovlet:
        """Build the markovlet this module will walk."""
        return io.node.markov

    def begin(self, io: ModuleInput) -> None: pass
    def adjust(self, io: ModuleInput, token: str, score: float) -> float: return score
    def on_emit(self, io: ModuleInput, token: str) -> None: pass
    def features(self, io: ModuleInput) -> Dict[str, object]: return {}

    def done(self, io: ModuleInput, token: str) -> bool:
        return (token in END and io.steps >= self.min_steps) or io.steps >= self.max_steps


def _content_words(res: Optional[ModuleResult]) -> set:
    return set(res.features.get("content", [])) if res else set()


def _upstream_chains(io: ModuleInput, keys: Sequence[str]) -> List[Markovlet]:
    return [io.up(k).markovlet for k in keys if io.up(k)]


class Observe(Module):
    """Walk the node's own markovlet: a faithful restatement of local flow."""
    name, provides, requires = "OBSERVE", "observation", ()
    min_steps = 3
    max_copy = None            # observing IS faithful copying

    def features(self, io):
        return {"content": sorted({t for t in io.tokens if _content(t)})}


class Describe(Module):
    """Walk the node's markovlet with the observation's content words cut out
    (restrict), so the segment can only add new attributes."""
    name, provides, requires = "DESCRIBE", "description", ("observation",)
    max_copy = 3

    def admit(self, io, token):
        return not (_content(token) and token in io.state.get("used", ()))

    def construct(self, io):
        seen = _content_words(io.up("observation"))
        io.state["used"] = set(seen)
        chain = io.node.markov.restrict(lambda t: t not in seen) if seen else io.node.markov
        return chain

    def adjust(self, io, token, score):
        if _content(token):
            return score + (-2.0 if token in io.state["used"] else 0.8)
        return score

    def on_emit(self, io, token):
        if _content(token):
            io.state["used"].add(token)

    def features(self, io):
        return {"content": sorted({t for t in io.tokens if _content(t)})}


class Relate(Module):
    """Join this node's markovlet to the next node's (overlap join), blended
    with the description markovlet. Done once a next-node-only word was used."""
    name, provides, requires = "RELATE", "relation", ("description",)
    min_steps = 5
    max_steps = 22
    max_copy = 3

    def construct(self, io):
        chain = io.node.markov.join(io.nxt.markov) if io.nxt else io.node.markov
        desc = io.up("description")
        if desc:
            chain = chain.merge(desc.markovlet, 0.25)
        mine = set(io.node.tokens)
        only_next = {t for t in (io.nxt.tokens if io.nxt else []) if _content(t) and t not in mine}
        io.state.update(only_next=only_next, bridged=not only_next, bridge_word=None)
        return chain

    def adjust(self, io, token, score):
        if not io.state["bridged"] and io.steps >= 2 and token in io.state["only_next"]:
            return score + 2.0
        return score

    def on_emit(self, io, token):
        if token in io.state["only_next"] and io.state["bridge_word"] is None:
            io.state["bridged"], io.state["bridge_word"] = True, token

    def done(self, io, token):
        return (token in END and io.steps >= self.min_steps and io.state["bridged"]) \
            or io.steps >= self.max_steps

    def features(self, io):
        return {"bridge_word": io.state["bridge_word"]}


class Abstract(Module):
    """Restrict both the node's and the relation's markovlets to long words and
    short function words, merge them, and walk. Done after two long words."""
    name, provides, requires = "ABSTRACT", "abstraction", ("relation",)
    min_steps = 4
    max_steps = 24
    max_copy = 2

    def admit(self, io, token):
        return self._keep(token)

    @staticmethod
    def _keep(t: str) -> bool:
        return t in PUNCT or len(t) <= 3 or len(t) >= 7

    def construct(self, io):
        chain = io.node.markov.restrict(self._keep)
        rel = io.up("relation")
        if rel:
            chain = chain.merge(rel.markovlet.restrict(self._keep), 0.5)
        io.state["longs"] = 0
        return chain

    def adjust(self, io, token, score):
        return score + (1.0 if len(token) >= 7 else 0.0)

    def on_emit(self, io, token):
        if len(token) >= 7 and token not in PUNCT:
            io.state["longs"] += 1

    def done(self, io, token):
        return (token in END and io.steps >= self.min_steps and io.state["longs"] >= 2) \
            or io.steps >= self.max_steps

    def features(self, io):
        return {"long_words": [t for t in io.tokens if len(t) >= 7 and t not in PUNCT]}


class Synthesize(Module):
    """Equal-weight merge of the description and abstraction markovlets."""
    name, provides, requires = "SYNTHESIZE", "synthesis", ("description", "abstraction")
    min_steps = 6
    max_copy = 2

    def construct(self, io):
        parts = _upstream_chains(io, self.requires) or [io.node.markov]
        return fold_merge(parts, "synthesis-chain")

    def features(self, io):
        return {"content": sorted({t for t in io.tokens if _content(t)})}


class Reflect(Module):
    """Merge every upstream markovlet into one memory chain and walk it."""
    name, provides = "REFLECT", "reflection"
    requires = ("observation", "description", "relation", "abstraction", "synthesis")
    max_copy = 3

    def construct(self, io):
        parts = _upstream_chains(io, self.requires) or [io.node.markov]
        return fold_merge(parts, "memory-chain")

    def features(self, io):
        return {"recalled_from": [k for k in self.requires if io.up(k)]}


INSTRUCTIONS: List[Module] = [Observe(), Describe(), Relate(), Abstract(), Synthesize(), Reflect()]
MODULES: Dict[str, Module] = {m.name: m for m in INSTRUCTIONS}


def check_dependencies(order: Sequence[Module]) -> List[str]:
    provided = {m.provides for m in order}
    return [f"{m.name} requires '{r}' but no module provides it"
            for m in order for r in m.requires if r not in provided]


def logsumexp(values: Sequence[float]) -> float:
    """Stable log(sum(exp(v))). Plain log(exp(x)) would overflow and is just x."""
    if not values:
        return 0.0
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


class SentenceNode:
    __slots__ = ("text", "tokens", "score", "instruction", "next", "markov")

    def __init__(self, text: str, tokens: List[str], score: float):
        self.text = text
        self.tokens = tokens
        self.score = score                      # log(sum(exp(token surprisal)))
        self.instruction: Module = INSTRUCTIONS[0]
        self.next: Optional["SentenceNode"] = None
        self.markov: Markovlet = Markovlet.from_tokens("sentence", tokens)


class SortedSentenceList:
    """Singly linked list kept in ascending score order (causal ascent)."""

    def __init__(self) -> None:
        self.head: Optional[SentenceNode] = None
        self.size = 0

    def insert(self, node: SentenceNode) -> None:
        if self.head is None or node.score < self.head.score:
            node.next = self.head
            self.head = node
        else:
            cur = self.head
            while cur.next is not None and cur.next.score <= node.score:
                cur = cur.next
            node.next = cur.next
            cur.next = node
        self.size += 1

    def extend_sorted(self, nodes: Sequence[SentenceNode]) -> None:
        """Bulk build: sort once, link with a tail pointer (O(n log n) instead of
        O(n^2) repeated insert()). Result is the same sorted linked list."""
        tail = None
        for node in sorted(nodes, key=lambda n: n.score):    # stable: ties keep corpus order
            node.next = None
            if tail is None:
                self.head = node
            else:
                tail.next = node
            tail = node
            self.size += 1

    def __iter__(self) -> Iterator[SentenceNode]:
        cur = self.head
        while cur is not None:
            yield cur
            cur = cur.next

    def link_instructions(self, order: Optional[Sequence[Module]] = None,
                          assign: str = "cycle") -> None:
        """Attach a module to every node.

        cycle  (default): rank % len(order) -> consecutive nodes run OBSERVE,
                          DESCRIBE, RELATE, ... so every module appears within the
                          first len(order) nodes, however large the corpus is, and
                          each module's upstream result is the node just before it.
        blocks:           old behaviour, one contiguous block of nodes per module.
                          On a large corpus the token budget is spent inside the
                          first block, so later modules never run.
        """
        order = list(order) if order else INSTRUCTIONS
        for rank, node in enumerate(self):
            if assign == "blocks":
                idx = min(len(order) - 1, rank * len(order) // max(1, self.size))
            else:
                idx = rank % len(order)
            node.instruction = order[idx]
            node.markov.name = f"node{rank}"
            node.markov.sources = [node.markov.name]


def build_sentence_list(corpus_text: str, alpha_mix: float = 0.3,
                        order: Optional[Sequence[Module]] = None,
                        assign: str = "cycle") -> SortedSentenceList:
    """
    Split corpus into sentences, mix local/global token frequencies, score each
    sentence as log(sum(exp(surprisal))), insert into a sorted linked list, and
    link each node to a cognitive instruction.
    """
    raw = re.split(r'(?<=[.!?])\s+', corpus_text.strip())
    sentences = [x.strip() for x in raw if x.strip()]
    token_lists = [tokenize(x) for x in sentences]

    global_counter: Counter = Counter()
    for toks in token_lists:
        global_counter.update(toks)
    total_global = max(1, sum(global_counter.values()))

    sl = SortedSentenceList()
    nodes: List[SentenceNode] = []
    for text, toks in zip(sentences, token_lists):
        if not toks:
            continue
        local = Counter(toks)
        total_local = max(1, len(toks))
        surprisals = []
        for tok in toks:
            p_local = local[tok] / total_local
            p_global = global_counter[tok] / total_global
            p_mix = (1.0 - alpha_mix) * p_local + alpha_mix * p_global
            surprisals.append(-math.log(max(p_mix, 1e-12)))
        nodes.append(SentenceNode(text, toks, logsumexp(surprisals)))

    sl.extend_sorted(nodes)
    sl.link_instructions(order, assign)
    return sl


def flatten_sentence_list(sl: SortedSentenceList) -> str:
    tokens: List[str] = []
    for node in sl:
        tokens.extend(node.tokens)
    return detokenize(tokens)


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


def natural_rate(model: NGramModel, tokens: Sequence[str]) -> float:
    """Share of bigrams inside the segment that occur somewhere in the corpus."""
    pairs = list(zip(tokens, tokens[1:]))
    if not pairs or len(model.counts) < 2:
        return 1.0
    seen = sum(1 for a_, b_ in pairs if model.counts[1].get((a_,), {}).get(b_))
    return seen / len(pairs)


def novelty(ngrams: Dict[int, set], tokens: Sequence[str], L: int = 4) -> float:
    """1 - (share of tokens covered by a verbatim corpus L-gram)."""
    grams, n = ngrams.get(L, set()), len(tokens)
    if n < L or not grams:
        return 1.0
    covered = set()
    for i in range(n - L + 1):
        if tuple(tokens[i:i + L]) in grams:
            covered.update(range(i, i + L))
    return 1.0 - len(covered) / n


def run_module(
    module: Module,
    model: NGramModel,
    fragments: NonIsomorphicFragments,
    io: ModuleInput,
    rng: random.Random,
    *,
    budget: int,
    step0: int,
    total_steps: int,
    beam_size: int,
    tau: float,
    temperature: float,
    show_trace: bool = False,
) -> ModuleResult:
    """Run ONE module in isolation: construct its markovlet, walk it, and
    return a ModuleResult carrying the markovlet learned from the output."""
    history = list(io.history)
    chain = module.construct(io)
    module.begin(io)
    fallbacks = 0
    forced_copy = 0
    closed = False
    forced = False

    for local in range(max(1, min(module.max_steps, budget))):
        lam = adiabatic_lambda(step0 + local, max(1, total_steps))

        succ: Dict[str, float] = {}
        if not chain.is_empty():
            succ = dict(_norm(chain.successors(history)))
            if io.steps == 0 and chain.starts and (not history or history[-1] in END or not succ):
                succ = dict(_norm(chain.starts))                # entry state
        if module.max_copy is not None and history and len(model.counts) > 1:
            g = model.counts[1].get((history[-1],), {})         # corpus-attested bigram successors
            gt = float(sum(g.values())) or 1.0
            for t, c in g.items():
                succ[t] = succ.get(t, 0.0) + 0.25 * c / gt
        succ = {t: c for t, c in succ.items() if t not in SPECIAL and module.admit(io, t)}
        if io.steps < module.min_steps:
            no_end = {t: c for t, c in succ.items() if t not in END}
            succ = no_end or succ
        if module.max_copy is not None and succ and io.tokens:
            seen_pairs = set(zip(io.tokens, io.tokens[1:]))      # no loops inside a segment
            prev = io.tokens[-1]
            unlooped = {t: c for t, c in succ.items() if (prev, t) not in seen_pairs}
            if unlooped:
                succ = unlooped
        if module.max_copy is not None and succ:
            L = module.max_copy + 1                              # cap on verbatim runs
            grams, tail = io.ngrams.get(L, set()), tuple(history[-(L - 1):])
            fresh = {t: c for t, c in succ.items()
                     if len(tail) < L - 1 or (tail + (t,)) not in grams}
            if fresh:
                succ = fresh
            elif ("," in model.vocab and io.steps >= 2 and history[-1] not in PUNCT
                  and io.state.get("splices", 0) < 2):
                io.state["splices"] = io.state.get("splices", 0) + 1
                succ = {",": 1.0}                                # splice: break the copy with a comma,
                                                                 # then continue from a clause start
            else:
                forced_copy += 1                                 # nothing but a copy was available

        fell_back = not succ
        if fell_back:                                           # dead end -> global n-gram
            ranked = [(t, lp) for t, lp in model.candidates(history, limit=max(40, beam_size * 8))
                      if t not in SPECIAL]
            if io.steps < module.min_steps:
                ranked = [(t, lp) for t, lp in ranked if t not in END] or ranked
        else:
            total = sum(succ.values())
            ranked = sorted(((t, math.log(max(c / total, 1e-12))) for t, c in succ.items()),
                            key=lambda item: item[1], reverse=True)

        scored: List[Candidate] = []
        for rank, (token, logp) in enumerate(ranked):
            score, consistency = fragments.score(history, token, logp, rank, lam)
            score = module.adjust(io, token, score)
            alignment = float(np.clip((score - logp) / 2.5 + 0.5, 0.0, 1.0))
            if consistency <= tau:
                scored.append(Candidate(token, logp, score, consistency, alignment))
        if not scored:
            token, logp = ranked[0]
            score, consistency = fragments.score(history, token, logp, 0, lam)
            scored = [Candidate(token, logp, score, consistency, 0.5)]

        scored.sort(key=lambda c: c.score, reverse=True)
        beam = scored[:max(1, beam_size)]
        probs = softmax([c.score for c in beam], temperature)
        for candidate, prob in zip(beam, probs):
            candidate.probability = float(prob)
        chosen = beam[0] if temperature <= 0.01 else rng.choices(beam, weights=probs, k=1)[0]

        history.append(chosen.token)
        io.tokens.append(chosen.token)
        io.steps += 1
        module.on_emit(io, chosen.token)
        fallbacks += int(fell_back)
        if show_trace:
            print(f"[{step0 + local:03d}] {module.name:<10} {chosen.token!r:<14} "
                  f"p={math.exp(chosen.base_logp):.2f} score={chosen.score:+.3f} "
                  f"D={chosen.consistency:.3f}{' FALLBACK' if fell_back else ''}")
        if module.done(io, chosen.token):
            closed = chosen.token in END
            break

    if io.tokens and io.tokens[-1] not in END and io.steps >= module.max_steps:
        if io.tokens[-1] in PUNCT:
            io.tokens[-1] = "."                                 # never end on a dangling comma
        else:
            io.tokens.append(".")                               # close an over-long sentence
        closed, forced = True, True

    feats = module.features(io)
    feats["forced_close"] = forced
    feats["forced_copy_steps"] = forced_copy
    feats["splices"] = io.state.get("splices", 0)
    tokens = tuple(io.tokens)
    needed = tuple(module.requires)
    return ModuleResult(
        module=module.name,
        provides=module.provides,
        node_idx=0,
        pass_no=0,
        tokens=tokens,
        text=detokenize(tokens),
        markovlet=Markovlet.from_tokens(module.provides, tokens),
        chain=chain.name,
        natural_rate=natural_rate(model, tokens),
        novelty=novelty(io.ngrams, tokens),
        features=feats,
        consumed=tuple(k for k in needed if k in io.upstream),
        missing=tuple(k for k in needed if k not in io.upstream),
        injected=0,
        fallbacks=fallbacks,
        closed=closed,
    )


def generate(
    model: NGramModel,
    prompt: str,
    token_count: int = 800,
    beam_size: int = 8,
    tau: float = 0.65,
    temperature: float = 0.85,
    seed: int = 7,
    show_trace: bool = False,
    sentences: Optional[SortedSentenceList] = None,
    results: Optional[List[ModuleResult]] = None,
    layout: str = "lines",
    labels: bool = False,
    arrange: str = "grouped",
) -> str:
    """Pipeline: walk the linked list, run each node's module, publish the
    resulting markovlet to a blackboard that later modules compose from."""
    rng = random.Random(seed)
    fragments = NonIsomorphicFragments(seed)
    output = tokenize(prompt) or ["The"]
    prompt_line = detokenize(output)
    entries: List[Tuple[str, str]] = []              # (module name, line) in run order
    all_vocab = frozenset(set(model.vocab) - SPECIAL)
    results = results if results is not None else []
    blackboard: Dict[str, ModuleResult] = {}

    if not (sentences and sentences.head):
        raise ValueError("generate() needs a non-empty SortedSentenceList")

    ngrams = {L: {tuple(n.tokens[i:i + L]) for n in sentences for i in range(len(n.tokens) - L + 1)}
              for L in range(2, 9)}
    node, node_idx, pass_no, produced = sentences.head, 0, 0, 0
    while produced < token_count:
        nxt = node.next or (sentences.head if sentences.size > 1 else None)
        module = node.instruction
        io = ModuleInput(history=tuple(output), node=node, nxt=nxt,
                         upstream=dict(blackboard), all_vocab=all_vocab, ngrams=ngrams)
        res = run_module(module, model, fragments, io, rng,
                         budget=token_count - produced, step0=produced,
                         total_steps=token_count, beam_size=beam_size, tau=tau,
                         temperature=temperature, show_trace=show_trace)
        res = dataclasses.replace(res, node_idx=node_idx, pass_no=pass_no)
        res.markovlet.name = f"{res.provides}#{pass_no}.{node_idx}"
        res.markovlet.sources = [res.markovlet.name]

        output.extend(res.tokens)
        entries.append((res.module, f"[{res.module}] {res.text}" if labels else res.text))
        blackboard[res.provides] = res
        results.append(res)
        produced += len(res.tokens)

        if node.next is None:
            node, node_idx, pass_no = sentences.head, 0, pass_no + 1
        else:
            node, node_idx = node.next, node_idx + 1

    if layout != "lines":
        return detokenize(output)
    if arrange == "sequence":                        # chronological: one line per module run
        return "\n".join([prompt_line] + [line for _, line in entries])

    # grouped: every module's lines together, groups in module order
    groups: Dict[str, List[str]] = {}
    for name, line in entries:                       # dict keeps first-appearance (= module) order
        groups.setdefault(name, []).append(line)
    out = [prompt_line]
    for name, group_lines in groups.items():
        out.append("")
        out.append(f"== {name} x{len(group_lines)} ==")
        out.extend(group_lines)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="N-gram text generator with non-isomorphic adiabatic fragment scoring.")
    parser.add_argument("corpus", nargs="?", help="UTF-8 text corpus file; uses a small built-in corpus if omitted.")
    parser.add_argument("--prompt", default="A process", help="Initial text context.")
    parser.add_argument("--tokens", type=int, default=1800, help="Maximum number of new tokens.")
    parser.add_argument("--order", type=int, default=2, help="N-gram order (default: 3).")
    parser.add_argument("--alpha", type=float, default=0.25, help="Additive smoothing value.")
    parser.add_argument("--beam", type=int, default=18, help="Number of candidate next words retained before sampling.")
    parser.add_argument("--tau", type=float, default=0.95, help="Maximum allowed fragment consistency score (0..1).")
    parser.add_argument("--temperature", type=float, default=.85, help="Sampling temperature; near zero selects the top candidate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable generation.")
    parser.add_argument("--trace", action="store_true", help="Print per-token scoring diagnostics.")
    parser.add_argument("--save", help="Optional output text file.")
    parser.add_argument("--inline", action="store_true",
                        help="Join all module segments into one paragraph instead of one line each.")
    parser.add_argument("--labels", action="store_true",
                        help="Prefix each line with the module that produced it, e.g. [OBSERVE].")
    parser.add_argument("--modules", default=",".join(m.name for m in INSTRUCTIONS),
                        help="Comma-separated module order, e.g. OBSERVE,DESCRIBE,RELATE.")
    parser.add_argument("--arrange", choices=["grouped", "sequence"], default="grouped",
                        help="grouped (default): all OBSERVE lines, then all DESCRIBE lines, ... in module "
                             "order; sequence: lines in the order the modules actually ran.")
    parser.add_argument("--assign", choices=["cycle", "blocks"], default="cycle",
                        help="How modules are spread over the sorted list: cycle (every module "
                             "appears, default) or blocks (one contiguous block per module).")
    parser.add_argument("--results-json", help="Write every ModuleResult and its markovlet to this JSON file.")
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

    names = [n.strip().upper() for n in args.modules.split(",") if n.strip()]
    unknown = [n for n in names if n not in MODULES]
    if unknown or not names:
        parser.error(f"Unknown module(s) {unknown}; choose from {sorted(MODULES)}")
    order = [MODULES[n] for n in names]
    for warning in check_dependencies(order):
        print(f"[warn] {warning} (module will run without it)")

    # Sorted linked list of sentences, each linked to a module
    sentence_list = build_sentence_list(corpus_text, alpha_mix=0.3, order=order, assign=args.assign)
    corpus_text = flatten_sentence_list(sentence_list)
    if args.trace:
        print("--- SORTED SENTENCE LIST ---")
        for i, n in enumerate(sentence_list):
            print(f"{i}: [{n.instruction.name:<10}] score={n.score:.3f}  {n.text[:60]}")
        print()

    if not tokenize(corpus_text):
        parser.error("Corpus contains no recognizable tokens.")

    model = NGramModel(order=args.order, alpha=args.alpha)
    model.train(corpus_text)

    exec_log: List[ModuleResult] = []
    result = generate(
        model=model,
        prompt=args.prompt,
        token_count=args.tokens,
        beam_size=args.beam,
        tau=float(np.clip(args.tau, 0.0, 1.0)),
        temperature=max(0.0, args.temperature),
        seed=args.seed,
        show_trace=args.trace,
        sentences=sentence_list,
        results=exec_log,
        layout="inline" if args.inline else "lines",
        labels=args.labels,
        arrange=args.arrange,
    )

    print("\n--- GENERATED TEXT ---\n")
    print(result)

    coverage = Counter(r.module for r in exec_log)
    print("\n--- MODULE COVERAGE ---")
    print("  ".join(f"{m.name}x{coverage.get(m.name, 0)}" for m in order))
    idle = [m.name for m in order if coverage.get(m.name, 0) == 0]
    if idle:
        print(f"[note] never ran: {', '.join(idle)} (raise --tokens, or use --assign cycle)")

    if args.trace:
        print("\n--- MARKOVLETS BUILT ---")
        for r in exec_log:
            st = r.markovlet.stats()
            print(f"pass {r.pass_no} node {r.node_idx} {r.module:<10} -> {r.provides:<11} "
                  f"walked={r.chain}\n    consumed={list(r.consumed)} missing={list(r.missing)} "
                  f"natural={r.natural_rate:.0%} novelty={r.novelty:.0%} fallbacks={r.fallbacks} "
                  f"learned(contexts={st['contexts']}, vocab={st['vocab']})\n    text={r.text}")

    if args.results_json:
        Path(args.results_json).write_text(
            json.dumps([result_to_dict(r) for r in exec_log], indent=2), encoding="utf-8")
        print(f"[Saved {len(exec_log)} module results (with markovlets) to {args.results_json}]")

    if args.save:
        Path(args.save).write_text(result, encoding="utf-8")
        print(f"\n[Saved output to {args.save}]")


if __name__ == "__main__":
    main()
