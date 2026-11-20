#!/usr/bin/env python3
"""Neural instruction-following generator, scaled to multi-GB datasets.

Model: P(next response token | last N response tokens, bag-of-words of the instruction)
  window embeddings  ┐
                     ├─ concat -> tanh hidden layer -> softmax over vocab
  mean instruction   ┘
  embedding (own table)

PyTorch tensors on the GPU, but with the SAME hand-written forward pass, manual backprop
and manual Adam as the NumPy original (no autograd, no nn.Module, no torch.optim) so every
bit of the math is unchanged. The deconstruct / regenerate loop is kept.

GPU notes
  * All parameters, Adam moments, activations and gradients live on the GPU (float32).
  * The on-disk memmap corpus is still sampled on the CPU, but a background thread
    prefetches pinned batches and copies them to the GPU asynchronously.
  * Checkpoints are still .npz, so models trained by the NumPy version load here and vice versa.
  * --gradcheck compares the manual gradients against torch.autograd as a sanity test.

What changed for large data
  * The dataset is STREAMED (HF streaming=True or line-by-line JSONL) up to --max-bytes
    (default 2 GB of text) so nothing close to the full text is ever held in RAM.
  * The vocabulary is built from the first --vocab-examples rows (a standard shortcut);
    everything else is encoded in parallel worker processes.
  * Encoded data is written once to a compact on-disk corpus (uint16 token ids,
    memory-mapped during training). Re-runs reuse it, so tokenizing happens only once.
  * Training windows are built on the fly per batch from random positions in the corpus
    (no giant list of windows). float32 throughout for speed.
  * --passes / --max-minutes bound the training time; Ctrl-C stops training cleanly and
    keeps the weights; --save checkpoints periodically.

Generation follows a modal constructor / deconstructor loop:
  Construct (add N tokens) -> Evaluate (exponential search for the "liar shift", the
  index where the text turns self-contradictory/looping) -> Deconstruct (reset to the
  last consistent state, collapse repeats) -> Construct again, reporting damage metrics
  (logical collapse, information loss, cognitive overload) for every round.

Usage:
  pip install datasets numpy torch
  python neural_instruct_generator_torch.py --dataset Open-Orca/OpenOrca --max-bytes 2e9 \
         --save model.npz --max-minutes 120
  python neural_instruct_generator_torch.py --dataset my_data.jsonl --max-bytes 2e9
  python neural_instruct_generator_torch.py --load model.npz
"""

import argparse
import json
import os
import random
import queue
import re
import threading
import time
from collections import Counter, deque
from itertools import chain, islice

import numpy as np
import torch

# Special tokens occupy the first ids.
SPECIALS = ["<unk>", "<pad>", "<|resp|>", "<|end|>"]
UNK, PAD, RESP, END = range(4)
SPECIAL_SET = set(SPECIALS)
TOKEN_RE = re.compile(r"<\|\w+\|>|\w+|[^\w\s]", re.UNICODE)

COLUMN_CANDIDATES = {
    "instruction": ["instruction", "prompt", "question"],
    "context": ["input", "context"],
    "response": ["output", "response", "answer", "completion"],
}


def tok(text):
    """Tokenize, dropping any literal special-token strings found in the data."""
    return [t for t in TOKEN_RE.findall(text) if t not in SPECIAL_SET]


# ------------------------------------------------------------ streaming input
def _pick(keys, candidates):
    for c in candidates:
        if c in keys:
            return c
    return None


def _s(v):
    return v.strip() if isinstance(v, str) else ""


def _iter_rows(source, split, seed):
    if source.endswith(".jsonl"):
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif source.endswith(".json"):
        print("Warning: .json arrays are loaded fully into memory; use .jsonl for huge files.")
        with open(source, "r", encoding="utf-8") as f:
            rows = json.load(f)
        random.Random(seed).shuffle(rows)
        yield from rows
    else:
        from datasets import load_dataset  # lazy: local use needs no extra deps
        ds = load_dataset(source, split=split, streaming=True)
        yield from ds.shuffle(seed=seed, buffer_size=10_000)


def stream_pairs(source, split="train", max_bytes=2_000_000_000, max_examples=None,
                 seed=0, columns=None, stats=None):
    """Yield (instruction_text, response_text) until ~max_bytes of text were produced."""
    rows = _iter_rows(source, split, seed)
    first = next(rows, None)
    if first is None:
        raise ValueError("Dataset is empty.")
    keys = set(first.keys())
    if columns:
        c_inst, c_ctx, c_resp = columns
    else:
        c_inst = _pick(keys, COLUMN_CANDIDATES["instruction"])
        c_ctx = _pick(keys, COLUMN_CANDIDATES["context"])
        c_resp = _pick(keys, COLUMN_CANDIDATES["response"])
    if not c_inst or not c_resp:
        raise ValueError(f"Couldn't detect columns in {sorted(keys)}; pass columns=(...).")
    print(f"Columns: instruction={c_inst}, context={c_ctx}, response={c_resp}")

    used = n = 0
    for r in chain([first], rows):
        inst = _s(r.get(c_inst))
        ctx = _s(r.get(c_ctx)) if c_ctx else ""
        resp = _s(r.get(c_resp))
        if not inst or not resp:
            continue
        text = f"{inst} {ctx}".strip()
        used += len(text) + len(resp)
        n += 1
        if stats is not None:
            stats["bytes"], stats["examples"] = used, n
        yield text, resp
        if used >= max_bytes or (max_examples and n >= max_examples):
            return


# --------------------------------------------------------- corpus build / read
_W = {}


def _init_worker(stoi, max_inst, max_resp):
    _W["stoi"], _W["mi"], _W["mr"] = stoi, max_inst, max_resp


def _encode_chunk(pairs):
    stoi, mi, mr = _W["stoi"], _W["mi"], _W["mr"]
    n = len(pairs)
    inst = np.full((n, mi), PAD, dtype=np.uint16)
    lens = np.empty(n, dtype=np.int64)
    flat = []
    for k, (i, r) in enumerate(pairs):
        it = [stoi.get(t, UNK) for t in tok(i)[:mi]]
        inst[k, :len(it)] = it
        rt = [stoi.get(t, UNK) for t in tok(r)[:mr]]
        rt.append(END)
        flat.extend(rt)
        lens[k] = len(rt)
    return inst, np.array(flat, dtype=np.uint16), lens


def build_corpus(source, cache_dir, split="train", max_bytes=2_000_000_000,
                 max_examples=None, max_vocab=80000, max_inst=32, max_resp=128,
                 vocab_examples=100_000_000, workers=1, seed=0, columns=None, chunk=2000):
    os.makedirs(cache_dir, exist_ok=True)
    stats = {"bytes": 0, "examples": 0}
    it = stream_pairs(source, split, max_bytes, max_examples, seed, columns, stats)
    t0 = time.time()

    # 1) buffer the first rows (raw text only) to build the vocabulary
    buf = list(islice(it, vocab_examples))
    if not buf:
        raise ValueError("No usable instruction/response pairs found.")
    counts = Counter()
    for i, r in buf:
        counts.update(tok(i)[:max_inst])
        counts.update(tok(r)[:max_resp])
    vocab = SPECIALS + [w for w, _ in counts.most_common(max_vocab - len(SPECIALS))]
    stoi = {w: i for i, w in enumerate(vocab)}
    del counts
    print(f"Vocab {len(vocab)} built from {len(buf)} examples. Encoding the stream...")

    # 2) encode everything (buffer first, then the rest of the stream) in parallel
    inst_f = open(os.path.join(cache_dir, "inst.u16"), "wb")
    resp_f = open(os.path.join(cache_dir, "resp.u16"), "wb")
    all_lens, n_tok = [], 0

    def chunks():
        src = chain(buf, it)
        while True:
            c = list(islice(src, chunk))
            if not c:
                return
            yield c

    def write(result):
        nonlocal n_tok
        inst, flat, lens = result
        inst_f.write(inst.tobytes())
        resp_f.write(flat.tobytes())
        all_lens.append(lens)
        n_tok += len(flat)
        done = sum(len(x) for x in all_lens)
        if (done // chunk) % 25 == 0:
            el = time.time() - t0
            print(f"  {done:,} examples, {stats['bytes'] / 1e6:,.0f} MB read, "
                  f"{n_tok / 1e6:,.1f}M response tokens, {el:,.0f}s")

    if workers > 1:
        import multiprocessing as mp
        pool = mp.Pool(workers, initializer=_init_worker, initargs=(stoi, max_inst, max_resp))
        pending = deque()
        try:
            for c in chunks():
                pending.append(pool.apply_async(_encode_chunk, (c,)))
                if len(pending) >= workers * 3:      # bounded in-flight work -> bounded RAM
                    write(pending.popleft().get())
            while pending:
                write(pending.popleft().get())
        finally:
            pool.close()
            pool.join()
    else:
        _init_worker(stoi, max_inst, max_resp)
        for c in chunks():
            write(_encode_chunk(c))

    inst_f.close()
    resp_f.close()
    lens = np.concatenate(all_lens)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    np.save(os.path.join(cache_dir, "offsets.npy"), offsets)
    meta = dict(vocab=vocab, n_examples=int(len(lens)), n_tokens=int(offsets[-1]),
                max_inst=max_inst, max_resp=max_resp, source=source)
    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"Corpus ready: {meta['n_examples']:,} examples, {meta['n_tokens']:,} target tokens "
          f"in {time.time() - t0:,.0f}s -> {cache_dir}/")


class Corpus:
    """Memory-mapped encoded corpus; builds training windows on the fly."""

    def __init__(self, cache_dir):
        with open(os.path.join(cache_dir, "meta.json"), "r", encoding="utf-8") as f:
            m = json.load(f)
        self.vocab = m["vocab"]
        self.n = m["n_examples"]
        self.n_tokens = m["n_tokens"]
        self.max_inst, self.max_resp = m["max_inst"], m["max_resp"]
        self.off = np.load(os.path.join(cache_dir, "offsets.npy"))
        self.resp = np.memmap(os.path.join(cache_dir, "resp.u16"), dtype=np.uint16, mode="r")
        self.inst = np.memmap(os.path.join(cache_dir, "inst.u16"), dtype=np.uint16,
                              mode="r").reshape(self.n, self.max_inst)

    def batch(self, rng, B, N):
        p = rng.integers(0, self.n_tokens, B)                  # random target positions
        k = np.searchsorted(self.off, p, side="right") - 1     # example of each position
        j = p - self.off[k]                                    # index inside the response
        r = j[:, None] + np.arange(N)[None, :] - N             # response index per window slot
        idx = np.clip(self.off[k][:, None] + r, 0, self.n_tokens - 1)
        win = self.resp[idx].astype(np.int64)
        win = np.where(r >= 0, win, np.where(r == -1, RESP, PAD))
        y = self.resp[p].astype(np.int64)
        inst = self.inst[k].astype(np.int64)
        return win, inst, y


# ------------------------------------------------------------------ GPU helpers
def get_device(name=None):
    if name:
        dev = torch.device(name)
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU.")
            return torch.device("cpu")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("Warning: no CUDA GPU found, running on CPU.")
    return torch.device("cpu")


class Prefetcher:
    """Samples batches from the memmap corpus in a background thread and ships them
    to the GPU (pinned memory + non_blocking copies) so the GPU never waits on disk/NumPy."""

    def __init__(self, corpus, B, N, device, seed=0, depth=6):
        self.corpus, self.B, self.N, self.device = corpus, B, N, device
        self.pin = device.type == "cuda"
        self.rng = np.random.default_rng(seed)
        self.q = queue.Queue(maxsize=depth)
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_evt.is_set():
            arrays = self.corpus.batch(self.rng, self.B, self.N)
            ts = [torch.from_numpy(a) for a in arrays]
            if self.pin:
                ts = [t.pin_memory() for t in ts]
            while not self.stop_evt.is_set():
                try:
                    self.q.put(ts, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def get(self):
        return [t.to(self.device, non_blocking=True) for t in self.q.get()]

    def close(self):
        self.stop_evt.set()


# ----------------------------------------------------------------------- model
class InstructGenerator:
    PARAM_KEYS = ["E", "C", "W1", "b1", "W2", "b2"]

    def __init__(self, context=3, embed_dim=48, hidden=192, max_vocab=80000,
                 max_inst=32, max_resp=128, seed=0, device=None):
        self.N = context
        self.d = embed_dim
        self.h = hidden
        self.max_vocab = max_vocab
        self.max_inst = max_inst
        self.max_resp = max_resp
        self.seed = seed
        self.device = device if isinstance(device, torch.device) else get_device(device)
        self.rng = np.random.default_rng(seed)              # init + corpus sampling
        self.tgen = torch.Generator(device=self.device)     # token sampling on the GPU
        self.tgen.manual_seed(seed)
        self.vocab = []
        self.stoi = {}
        self.params = None

    def set_vocab(self, vocab):
        self.vocab = list(vocab)
        self.stoi = {w: i for i, w in enumerate(self.vocab)}

    def _encode(self, tokens):
        return [self.stoi.get(t, UNK) for t in tokens]

    def _to_dev(self, a):
        return torch.as_tensor(a).to(self.device)

    def _init_params(self):
        V, d, h, N = len(self.vocab), self.d, self.h, self.N
        r = self.rng
        in_dim = (N + 1) * d

        def nrm(scale, shape):
            return self._to_dev(r.normal(0, scale, shape).astype(np.float32))

        dev = self.device
        self.params = {
            "E": nrm(0.1, (V, d)),
            "C": nrm(0.1, (V, d)),
            "W1": nrm(1 / np.sqrt(in_dim), (in_dim, h)),
            "b1": torch.zeros(h, dtype=torch.float32, device=dev),
            "W2": nrm(1 / np.sqrt(h), (h, V)),
            "b2": torch.zeros(V, dtype=torch.float32, device=dev),
        }
        self._m = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    # -- forward / backward  (manual math, identical to the NumPy version)
    def _forward(self, ctx, inst):
        """ctx: (B, N) window ids; inst: (B, L) instruction ids (PAD-padded). Long tensors on device."""
        p = self.params
        B = ctx.shape[0]
        emb = p["E"][ctx].reshape(B, -1)
        mask = (inst != PAD).to(torch.float32)
        cnt = mask.sum(1, keepdim=True).clamp(min=1.0)
        cond = (p["C"][inst] * mask[:, :, None]).sum(1) / cnt
        x = torch.cat([emb, cond], dim=1)
        a = torch.tanh(x @ p["W1"] + p["b1"])
        logits = a @ p["W2"] + p["b2"]
        return logits, (ctx, inst, mask, cnt, x, a)

    @staticmethod
    def _softmax(z):
        z = z - z.max(dim=1, keepdim=True).values
        e = torch.exp(z)
        return e / e.sum(dim=1, keepdim=True)

    def _backward(self, probs, y, cache):
        ctx, inst, mask, cnt, x, a = cache
        p = self.params
        B = y.shape[0]
        dlogits = probs.clone()
        dlogits[torch.arange(B, device=y.device), y] -= 1
        dlogits /= B
        g = {"W2": a.T @ dlogits, "b2": dlogits.sum(0)}
        dz = (dlogits @ p["W2"].T) * (1 - a ** 2)
        g["W1"] = x.T @ dz
        g["b1"] = dz.sum(0)
        dx = dz @ p["W1"].T
        nd = self.N * self.d
        gE = torch.zeros_like(p["E"])                       # np.add.at -> index_add_
        gE.index_add_(0, ctx.reshape(-1), dx[:, :nd].reshape(B * self.N, self.d))
        gC = torch.zeros_like(p["C"])
        rows = dx[:, None, nd:] * (mask / cnt)[:, :, None]
        gC.index_add_(0, inst.reshape(-1), rows.reshape(-1, self.d))
        g["E"], g["C"] = gE, gC
        return g

    def _adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        c1, c2 = 1 - b1 ** self._t, 1 - b2 ** self._t
        for k, g in grads.items():
            m, v = self._m[k], self._v[k]
            m.mul_(b1).add_(g, alpha=1 - b1)                # m = b1*m + (1-b1)*g
            v.mul_(b2).addcmul_(g, g, value=1 - b2)         # v = b2*v + (1-b2)*g*g
            denom = (v / c2).sqrt_().add_(eps)              # sqrt(v/c2) + eps
            self.params[k].addcdiv_(m, denom, value=-lr / c1)   # p -= lr * (m/c1) / denom

    @torch.no_grad()
    def gradcheck(self, corpus, batch=64):
        """Compare the manual gradients with torch.autograd on one real batch (float64)."""
        if self.params is None:
            self._init_params()
        X, I, y = corpus.batch(self.rng, batch, self.N)
        X, I, y = (self._to_dev(a) for a in (X, I, y))
        saved = self.params
        self.params = {k: v.double() for k, v in saved.items()}
        logits, cache = self._forward(X, I)
        probs = self._softmax(logits)
        manual = self._backward(probs, y, cache)
        with torch.enable_grad():
            ag = {k: v.clone().requires_grad_(True) for k, v in self.params.items()}
            self.params = ag
            lg, _ = self._forward(X, I)
            loss = torch.nn.functional.cross_entropy(lg, y)
            loss.backward()
        print("Gradient check (manual vs autograd, max abs diff):")
        ok = True
        for k in self.PARAM_KEYS:
            diff = (manual[k] - ag[k].grad).abs().max().item()
            ok &= diff < 1e-8
            print(f"  {k:3s} {diff:.3e}")
        self.params = saved
        print("PASS" if ok else "FAIL")
        return ok

    # -- training
    def _pad_inst(self, ids):
        ids = ids[:self.max_inst]
        return ids + [PAD] * (self.max_inst - len(ids))

    def train_corpus(self, corpus, passes=1.0, batch=512, lr=3e-3, max_seconds=None,
                     log_every=200, save_path=None, checkpoint_every=0):
        """Train on random positions of the on-disk corpus (on the GPU).

        passes: how many times the total number of target tokens to visit (1.0 = about
        one epoch; positions are sampled randomly, so it is approximate).
        max_seconds: hard wall-clock cap; the LR schedule follows whichever limit is closer.
        """
        self._init_params()
        steps_total = max(1, int(passes * corpus.n_tokens / batch))
        print(f"Device {self.device}, vocab {len(self.vocab)}, {corpus.n_tokens:,} positions, "
              f"{steps_total:,} steps of {batch} ({passes} pass(es)).")
        ar = torch.arange(batch, device=self.device)
        loader = Prefetcher(corpus, batch, self.N, self.device, seed=self.seed)
        t0 = last_t = time.time()
        run_loss = torch.zeros((), device=self.device)
        run_n = 0
        step = 0
        try:
            for step in range(steps_total):
                now = time.time()
                elapsed = now - t0
                if max_seconds and elapsed >= max_seconds:
                    print("Time cap reached.")
                    break
                frac = step / steps_total
                if max_seconds:
                    frac = max(frac, elapsed / max_seconds)
                cur_lr = lr * (1 - 0.9 * min(frac, 1.0))

                X, I, y = loader.get()
                with torch.no_grad():
                    logits, cache = self._forward(X, I)
                    probs = self._softmax(logits)
                    loss = -torch.log(probs[ar, y] + 1e-9).mean()
                    self._adam(self._backward(probs, y, cache), cur_lr)
                run_loss += loss                             # stays on the GPU, no sync per step
                run_n += 1

                if (step + 1) % log_every == 0:
                    mean_loss = run_loss.item() / run_n      # one sync per log line
                    now = time.time()
                    rate = log_every * batch / max(now - last_t, 1e-9)
                    eta = (steps_total - step - 1) * batch / rate
                    if max_seconds:
                        eta = min(eta, max_seconds - (now - t0))
                    print(f"  step {step + 1:,}/{steps_total:,}  loss {mean_loss:.4f}  "
                          f"{rate:,.0f} pos/s  ETA {max(eta, 0) / 60:.1f} min")
                    run_loss.zero_()
                    run_n, last_t = 0, now
                if save_path and checkpoint_every and (step + 1) % checkpoint_every == 0:
                    self.save(save_path)
        except KeyboardInterrupt:
            print("\nInterrupted - keeping the current weights.")
        finally:
            loader.close()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print(f"Training finished after {step + 1:,} steps ({(time.time() - t0) / 60:.1f} min).")

    # -- save / load  (same .npz format as the NumPy version)
    def save(self, path):
        cfg = json.dumps(dict(N=self.N, d=self.d, h=self.h, max_vocab=self.max_vocab,
                              max_inst=self.max_inst, max_resp=self.max_resp))
        arrays = {k: v.detach().cpu().numpy() for k, v in self.params.items()}
        np.savez_compressed(path, vocab=np.array(self.vocab), cfg=np.array(cfg), **arrays)

    @classmethod
    def load(cls, path, device=None):
        z = np.load(path, allow_pickle=False)
        c = json.loads(str(z["cfg"]))
        m = cls(context=c["N"], embed_dim=c["d"], hidden=c["h"], max_vocab=c["max_vocab"],
                max_inst=c["max_inst"], max_resp=c["max_resp"], device=device)
        m.set_vocab(str(w) for w in z["vocab"])
        m.params = {k: torch.from_numpy(z[k].astype(np.float32)).to(m.device)
                    for k in cls.PARAM_KEYS}
        return m

    # -- text helpers
    @staticmethod
    def detokenize(tokens):
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:)\]])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        return re.sub(r"(\w)\s'\s(\w)", r"\1'\2", text)   # don ' t -> don't

    # -- modal constructor / deconstructor loop
    @staticmethod
    def _collapse(tokens):
        """Collapse immediate token repeats and repeated adjacent phrase loops."""
        collapsed = []
        for token in tokens:
            if not collapsed or token.lower() != collapsed[-1].lower():
                collapsed.append(token)

        changed = True
        while changed:
            changed = False
            for size in range(min(8, len(collapsed) // 2), 1, -1):
                i = 0
                while i + 2 * size <= len(collapsed):
                    left = collapsed[i:i + size]
                    right = collapsed[i + size:i + 2 * size]
                    if [x.lower() for x in left] == [x.lower() for x in right]:
                        del collapsed[i + size:i + 2 * size]
                        changed = True
                    else:
                        i += 1
                if changed:
                    break
        return collapsed

    def deconstruct(self, text):
        """Collapse repeats/loops in text (kept for API compatibility)."""
        return self.detokenize(self._collapse(TOKEN_RE.findall(text)))

    @staticmethod
    def _repeat_ngrams(tokens, n=3):
        """Positions whose n-gram already appeared earlier (catches non-adjacent loops)."""
        low = [t.lower() for t in tokens]
        seen, rep = set(), 0
        for i in range(n - 1, len(low)):
            g = tuple(low[i - n + 1:i + 1])
            if g in seen:
                rep += 1
            else:
                seen.add(g)
        return rep

    def _inconsistency(self, tokens):
        """Evaluate: self-repetition in this prefix = adjacent repeats/loops that
        deconstruct would delete + repeated trigrams anywhere earlier in the text.
        Both counts only grow as tokens are added, so truth is a step in the index."""
        return (len(tokens) - len(self._collapse(tokens))) + self._repeat_ngrams(tokens)

    def liar_shift(self, tokens, tolerance=2):
        """Index n of the first paradoxical prefix tokens[:n] (None if there is none).

        Truth value T(n) = "inconsistency of tokens[:n] < tolerance" is a step function
        of n (repeat counts only grow with n), so the flip is located with an
        exponential search followed by a binary search.
        """
        def paradox(n):
            return self._inconsistency(tokens[:n]) >= tolerance

        L = len(tokens)
        if L == 0 or not paradox(L):
            return None
        hi = 1
        while hi < L and not paradox(hi):
            hi *= 2
        lo = hi // 2                      # invariant: not paradox(lo), paradox(hi)
        hi = min(hi, L)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if paradox(mid):
                hi = mid
            else:
                lo = mid
        return hi

    @torch.no_grad()
    def _surprisal(self, inst_row, ids):
        """Mean negative log-likelihood (nats/token) of ids under the model."""
        if not ids:
            return 0.0
        L = len(ids)
        S = torch.tensor([PAD] * (self.N - 1) + [RESP] + ids, dtype=torch.long,
                         device=self.device)
        W = S.unfold(0, self.N, 1)[:L]                       # (L, N) sliding windows
        logits, _ = self._forward(W, inst_row[None, :].expand(L, -1))
        z = logits - logits.max(dim=1, keepdim=True).values
        logp = z - torch.log(torch.exp(z).sum(dim=1, keepdim=True))
        tgt = torch.tensor(ids, dtype=torch.long, device=self.device)
        return float(-logp[torch.arange(L, device=self.device), tgt].mean())

    # -- generation
    @torch.no_grad()
    def _sample(self, window, inst_row, temperature, top_k):
        w = torch.tensor([window], dtype=torch.long, device=self.device)
        logits, _ = self._forward(w, inst_row[None, :])
        logits = logits[0].double() / max(temperature, 1e-4)
        for banned in (UNK, PAD, RESP):
            logits[banned] = float("-inf")
        if top_k and top_k < logits.shape[0]:
            cutoff = torch.topk(logits, top_k).values[-1]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        probs = torch.softmax(logits, dim=0)
        probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1, generator=self.tgen).item())

    def _continue(self, prefix_tokens, inst_row, length, temperature, top_k):
        """Continue a response from prefix_tokens; stops at <|end|> or `length` tokens."""
        ids = [PAD] * self.N + [RESP] + self._encode(prefix_tokens)
        window = ids[-self.N:]
        out = []
        for _ in range(length):
            t = self._sample(window, inst_row, temperature, top_k)
            if t == END:
                break
            out.append(self.vocab[t])
            window = window[1:] + [t]
        return out

    def generate(self, instruction, length=64, deconstruction_rounds=1,
                 temperature=0.8, top_k=40, tolerance=2, return_trace=False):
        """Modal constructor / deconstructor loop.

        Construct : add up to `length` tokens (M).
        Evaluate  : exponential search for the liar shift - the index where the text
                    turns paradoxical (inconsistency score >= `tolerance`).
        Deconstruct: reset to the last consistent state (just before the shift) and
                    collapse remaining repeats (D), replacing the text.
        The trace holds the damage metrics of every round.
        """
        if self.params is None:
            raise RuntimeError("Train or load a model first.")
        if length < 0:
            raise ValueError("Token length must be non-negative.")
        if deconstruction_rounds < 0:
            raise ValueError("Deconstruction rounds must be non-negative.")
        if tolerance < 1:
            raise ValueError("Tolerance must be at least 1.")

        inst_row = torch.tensor(self._pad_inst(self._encode(tok(instruction))),
                                dtype=torch.long, device=self.device)
        resp = self._continue([], inst_row, length, temperature, top_k)   # Construct

        trace = []
        for r in range(deconstruction_rounds):
            before = len(resp)
            shift = self.liar_shift(resp, tolerance)                      # Evaluate
            kept = resp if shift is None else resp[:shift - 1]            # reset point
            reduced = self._collapse(kept)                                # Deconstruct
            trace.append(dict(
                round=r + 1,
                liar_shift=shift,
                length=before,
                logical_collapse=min(1.0, self._inconsistency(resp) / max(before, 1)),
                information_loss=(before - len(reduced)) / max(before, 1),
                cognitive_overload=self._surprisal(inst_row, self._encode(resp)),
            ))
            resp = reduced + self._continue(reduced, inst_row, length,    # Construct
                                            temperature, top_k)
        text = self.detokenize(resp)
        return (text, trace) if return_trace else text


# ------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tatsu-lab/alpaca",
                    help="HF dataset id or local .jsonl file (.json loads fully into RAM)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-bytes", type=float, default=2e9,
                    help="approx. characters of text to ingest (default 2e9 = ~2 GB)")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--cache-dir", default="corpus_cache", help="encoded corpus location")
    ap.add_argument("--rebuild", action="store_true", help="re-tokenize even if a cache exists")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--vocab-examples", type=int, default=100_000_000)
    ap.add_argument("--max-vocab", type=int, default=80000)
    ap.add_argument("--max-inst", type=int, default=32, help="instruction tokens kept")
    ap.add_argument("--max-resp", type=int, default=128, help="response tokens kept")
    ap.add_argument("--epochs", "--passes", dest="passes", type=float, default=1.0,
                    help="passes over the corpus's target tokens (fractions allowed)")
    ap.add_argument("--max-minutes", type=float, default=None, help="wall-clock training cap")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--save", default=None, help="save trained model to this .npz")
    ap.add_argument("--checkpoint-every", type=int, default=20000, help="steps (needs --save)")
    ap.add_argument("--load", default=None, help="load a saved .npz instead of training")
    ap.add_argument("--tolerance", type=int, default=2,
                    help="inconsistency score (repeats) that makes a text 'paradoxical'; "
                         "1 = most sensitive")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: cuda if available)")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 tensor-core matmuls (faster on Ampere+, slightly lower precision)")
    ap.add_argument("--gradcheck", action="store_true",
                    help="verify manual gradients against torch.autograd, then exit")
    ap.add_argument("--train-only", action="store_true", help="skip the interactive prompt")
    args = ap.parse_args()

    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")

    if args.load:
        model = InstructGenerator.load(args.load, device=device)
        print(f"Loaded model from {args.load} (vocab {len(model.vocab)}).")
    else:
        meta_path = os.path.join(args.cache_dir, "meta.json")
        if args.rebuild or not os.path.exists(meta_path):
            build_corpus(args.dataset, args.cache_dir, args.split, int(args.max_bytes),
                         args.max_examples, args.max_vocab, args.max_inst, args.max_resp,
                         args.vocab_examples, args.workers)
        else:
            print(f"Reusing encoded corpus in {args.cache_dir}/ (use --rebuild to redo).")
        corpus = Corpus(args.cache_dir)
        model = InstructGenerator(context=args.context, max_vocab=args.max_vocab,
                                  max_inst=corpus.max_inst, max_resp=corpus.max_resp,
                                  device=device)
        model.set_vocab(corpus.vocab)
        if args.gradcheck:
            ok = model.gradcheck(corpus)
            raise SystemExit(0 if ok else 1)
        model.train_corpus(corpus, passes=args.passes, batch=args.batch, lr=args.lr,
                           max_seconds=args.max_minutes * 60 if args.max_minutes else None,
                           save_path=args.save, checkpoint_every=args.checkpoint_every)
        if args.save:
            model.save(args.save)
            print(f"Saved to {args.save}")

    if args.train_only:
        return
    while True:
        instruction = input("\nInstruction (or EXIT): ").strip()
        if instruction.upper() == "EXIT":
            break
        if not instruction:
            continue
        try:
            length = int(input("Max tokens per pass: "))
            rounds = int(input("Deconstruction/regeneration rounds (0 = off): "))
            text, trace = model.generate(instruction, length, rounds,
                                         temperature=args.temperature, top_k=args.top_k,
                                         tolerance=args.tolerance, return_trace=True)
            print("\n" + text)
        except ValueError as e:
            print("Invalid input:", e)
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
