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

Simulated computer agent (--agent, needs sim_computer.py next to this file)
  The same model is trained on synthetic (instruction -> action trace) pairs and its output
  is parsed into actions (open / click / type / press ...) that run on an IN-MEMORY fake
  computer: virtual files, a mini terminal, editor, file manager and a canned browser.
  Nothing touches the real machine. The model is open-loop (it plans from the instruction
  alone), so --eval-agent reports how often the simulated end state meets the goal.
    python neural_instruct_generator_torch.py --agent
  Downloadable data instead of generated tasks: osunlp/Mind2Web (human web-action
  trajectories such as "[combobox] Where to? -> TYPE: Paris"). Actions run on SimWeb, a
  dry-run executor (no browser, no network) and are scored against held-out human runs:
    python neural_instruct_generator_torch.py --agent --agent-data mind2web

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
import zlib
from collections import Counter, deque
from itertools import chain, islice

import numpy as np
import torch

# Special tokens occupy the first ids.
SPECIALS = ["<unk>", "<pad>", "<|resp|>", "<|end|>", "<|nl|>", "<|ind|>"]
UNK, PAD, RESP, END = range(4)
NL_TOK, IND_TOK = "<|nl|>", "<|ind|>"      # newline / one indent unit (2 spaces)
SPECIAL_SET = set(SPECIALS)
TOKEN_RE = re.compile(r"<\|\w+\|>|\w+|[^\w\s]", re.UNICODE)

COLUMN_CANDIDATES = {
    "instruction": ["instruction", "prompt", "question"],
    "context": ["input", "context"],
    "response": ["output", "response", "answer", "completion"],
}


# Options for downloadable datasets, set from the CLI (kept global to avoid threading them
# through every function): parquet file glob, columns to read, and a hold-out rule.
HF_OPTS = dict(data_files=None, read_columns=None, holdout=None)


def tok(text):
    """Tokenize, keeping line structure: newlines -> <|nl|>, leading indent -> <|ind|> per
    2 spaces (tab = 4). Literal special-token strings found in the data are dropped."""
    out = []
    for li, line in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n")):
        if li:
            out.append(NL_TOK)
        body = line.lstrip(" \t")
        lead = line[:len(line) - len(body)].replace("\t", "    ")
        out.extend([IND_TOK] * (len(lead) // 2))
        out.extend(t for t in TOKEN_RE.findall(body) if t not in SPECIAL_SET)
    return out


def _wordy(t):
    """True for real words/numbers; False for punctuation and the structure tokens."""
    return t not in (NL_TOK, IND_TOK) and any(c.isalnum() for c in t)


# ------------------------------------------------------------ streaming input
def _pick(keys, candidates):
    for c in candidates:
        if c in keys:
            return c
    return None


def _s(v):
    if isinstance(v, (list, tuple)):               # e.g. a list of action strings
        return "\n".join(str(x).strip() for x in v if str(x).strip())
    return v.strip() if isinstance(v, str) else ""


def _is_holdout(row):
    """True for rows reserved for evaluation (stable hash of an id column)."""
    ho = HF_OPTS["holdout"]
    return bool(ho) and zlib.crc32(str(row.get(ho[0])).encode()) % int(ho[1]) == 0


def _iter_rows(source, split, seed, config=None, shuffle=True):
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
        it, opts = None, HF_OPTS
        if opts["data_files"]:
            # Fast path: read only the needed columns from parquet files, so huge unused
            # columns (e.g. raw HTML) are never downloaded or held in memory.
            try:
                kind = "parquet" if ".parquet" in opts["data_files"] else "json"
                kw = {"columns": opts["read_columns"]} if kind == "parquet" and \
                    opts["read_columns"] else {}
                ds = load_dataset(kind, data_files=opts["data_files"], split="train",
                                  streaming=True, **kw)
                it = iter(ds.shuffle(seed=seed, buffer_size=10_000) if shuffle else ds)
                first = next(it, None)
                it = chain([first], it) if first is not None else None
            except Exception as e:
                print(f"Warning: could not read {opts['data_files']} "
                      f"({type(e).__name__}: {e}).\nFalling back to the dataset repo itself; "
                      f"that can download several GB.")
                it = None
        if it is None:
            ds = load_dataset(source, config, split=split, streaming=True)
            big = bool(opts["data_files"])               # fallback rows can be huge
            it = iter(ds.shuffle(seed=seed, buffer_size=64 if big else 10_000)
                      if shuffle else ds)
        yield from it


def _chat_pairs(rows, text_col, group_col, max_bytes, max_examples, stats):
    """Chat logs -> (message, reply) pairs: every message is the 'instruction' for the next
    message in the same room/thread (rows must be stored in conversation order)."""
    prev_text = prev_group = None
    used = n = 0
    for r in rows:
        text = _s(r.get(text_col))
        if not text:
            continue
        group = r.get(group_col) if group_col else None
        if prev_text is not None and group == prev_group:
            used += len(prev_text) + len(text)
            n += 1
            if stats is not None:
                stats["bytes"], stats["examples"] = used, n
            yield prev_text, text
            if used >= max_bytes or (max_examples and n >= max_examples):
                return
        prev_text, prev_group = text, group


def stream_pairs(source, split="train", max_bytes=2_000_000_000, max_examples=None,
                 seed=0, columns=None, stats=None, config=None, chat=None):
    """Yield (instruction_text, response_text) until ~max_bytes of text were produced.

    chat=(text_col, group_col) switches to chat-log mode (consecutive messages of a group)."""
    if chat:
        print(f"Chat mode: text={chat[0]}, group={chat[1]} (no shuffling, order matters).")
        yield from _chat_pairs(_iter_rows(source, split, seed, config, shuffle=False),
                               chat[0], chat[1], max_bytes, max_examples, stats)
        return
    rows = _iter_rows(source, split, seed, config)
    first = next(rows, None)
    if first is None:
        raise ValueError("Dataset is empty.")
    keys = set(first.keys())
    if columns:
        c_inst, c_ctx, c_resp = (None if c in (None, "-", "") else c for c in columns)
    else:
        c_inst = _pick(keys, COLUMN_CANDIDATES["instruction"])
        c_ctx = _pick(keys, COLUMN_CANDIDATES["context"])
        c_resp = _pick(keys, COLUMN_CANDIDATES["response"])
    if not c_inst or not c_resp:
        raise ValueError(f"Couldn't detect columns in {sorted(keys)}; pass columns=(...).")
    print(f"Columns: instruction={c_inst}, context={c_ctx}, response={c_resp}")

    used = n = 0
    for r in chain([first], rows):
        if _is_holdout(r):                  # reserved for --eval-agent
            continue
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


def inspect_dataset(source, split="train", config=None, n=3):
    """Print configs, columns and the first rows so you can pick --columns / --chat-*."""
    if not source.endswith((".json", ".jsonl")):
        try:
            from datasets import get_dataset_config_names
            print("Configs:", get_dataset_config_names(source))
        except Exception as e:
            print("(could not list configs:", e, ")")
    for k, r in enumerate(islice(_iter_rows(source, split, 0, config, shuffle=False), n)):
        print(f"--- row {k} ---")
        for key, v in r.items():
            v = str(v).replace("\n", " ")
            print(f"  {key}: {v[:150]}{'...' if len(v) > 150 else ''}")


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
        rtoks = tok(r)
        rt = [stoi.get(t, UNK) for t in rtoks[:mr]]
        if len(rtoks) <= mr:          # only a COMPLETE response ends with <|end|>
            rt.append(END)
        flat.extend(rt)
        lens[k] = len(rt)
    return inst, np.array(flat, dtype=np.uint16), lens


def build_corpus(source, cache_dir, split="train", max_bytes=2_000_000_000,
                 max_examples=None, max_vocab=8000, max_inst=32, max_resp=128,
                 vocab_examples=100_000, workers=1, seed=0, columns=None, chunk=2000,
                 config=None, chat=None):
    assert max_vocab <= 65535, "token ids are stored as uint16"
    os.makedirs(cache_dir, exist_ok=True)
    stats = {"bytes": 0, "examples": 0}
    it = stream_pairs(source, split, max_bytes, max_examples, seed, columns, stats,
                      config, chat)
    t0 = time.time()

    # 1) buffer the first rows (raw text only) to build the vocabulary
    buf = list(islice(it, vocab_examples))
    if not buf:
        raise ValueError("No usable instruction/response pairs found.")
    counts = Counter()
    for i, r in buf:
        counts.update(tok(i)[:max_inst])
        counts.update(tok(r)[:max_resp])
    vocab = SPECIALS + [w for w, _ in counts.most_common(max_vocab)
                        if w not in SPECIAL_SET][:max_vocab - len(SPECIALS)]
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

    def __init__(self, context=3, embed_dim=48, hidden=192, max_vocab=8000,
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
    _CODE_FIXES = [
        (r"`\s+`\s+`", "```"), (r"```\s+(\w+)$", r"```\1"),
        (r"\s+([,;)\].:])", r"\1"), (r"([(\[.])\s+", r"\1"),
        (r"::\s+(?=\w)", "::"), (r"(\w)\s+\(", r"\1("), (r"(\w)\s+\[", r"\1["),
        (r"-\s+>", "->"), (r"=\s+>", "=>"), (r"([+\-*/%<>=!&|^])\s+=", r"\1="),
        (r"\+\s+\+", "++"), (r"-\s+-", "--"), (r"&\s+&", "&&"), (r"\|\s+\|", "||"),
        (r"<\s+<", "<<"), (r">\s+>", ">>"),
        (r"#\s*include\s*<\s*([\w./]+)\s*>", r"#include <\1>"),
        (r"#\s+(include|define|pragma|ifdef|ifndef|endif|else|if)\b", r"#\1"),
    ]
    _PROSE_FIXES = [
        (r"\s+([.,!?;:)\]])", r"\1"), (r"([(\[])\s+", r"\1"),
        (r"(\w)\s'\s(\w)", r"\1'\2"),               # don ' t -> don't
    ]

    @classmethod
    def detokenize(cls, tokens):
        """Tokens -> text. Rebuilds newlines/indentation and uses code-style spacing when
        the output looks like code (indent tokens, backticks, or { and ;)."""
        code = (IND_TOK in tokens or "`" in tokens or ("{" in tokens and ";" in tokens))
        fixes = cls._CODE_FIXES if code else cls._PROSE_FIXES
        lines, cur = [], []
        for t in list(tokens) + [NL_TOK]:
            if t != NL_TOK:
                cur.append(t)
                continue
            n = 0
            while n < len(cur) and cur[n] == IND_TOK:
                n += 1
            text = " ".join(x for x in cur[n:] if x != IND_TOK)
            for pat, repl in fixes:
                text = re.sub(pat, repl, text)
            lines.append("  " * n + text.rstrip())
            cur = []
        return "\n".join(lines).strip("\n")

    # -- modal constructor / deconstructor loop
    @staticmethod
    def _collapse(tokens):
        """Collapse immediate token repeats and repeated adjacent phrase loops."""
        collapsed = []
        for token in tokens:
            if (not collapsed or token.lower() != collapsed[-1].lower()
                    or not _wordy(token)):       # keep "}" "}" and indentation runs
                collapsed.append(token)

        changed = True
        while changed:
            changed = False
            for size in range(min(8, len(collapsed) // 2), 1, -1):
                i = 0
                while i + 2 * size <= len(collapsed):
                    left = collapsed[i:i + size]
                    right = collapsed[i + size:i + 2 * size]
                    if (any(_wordy(x) for x in left)
                            and [x.lower() for x in left] == [x.lower() for x in right]):
                        del collapsed[i + size:i + 2 * size]
                        changed = True
                    else:
                        i += 1
                if changed:
                    break
        return collapsed

    def deconstruct(self, text):
        """Collapse repeats/loops in text (kept for API compatibility)."""
        return self.detokenize(self._collapse(tok(text)))

    @staticmethod
    def _repeat_ngrams(tokens, n=None):
        """Positions whose n-gram already appeared earlier (catches non-adjacent loops).
        Line-structured text (code) legitimately repeats short n-grams, so it uses n=8."""
        if n is None:
            n = 8 if (IND_TOK in tokens or NL_TOK in tokens) else 3
        low = [t.lower() for t in tokens]
        seen, rep = set(), 0
        for i in range(n - 1, len(low)):
            g = tuple(low[i - n + 1:i + 1])
            if not any(_wordy(x) for x in g):
                continue
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
                 temperature=0.8, top_k=40, tolerance=2, return_trace=False, raw=False):
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
        text = resp if raw else self.detokenize(resp)      # raw=True -> token list
        return (text, trace) if return_trace else text


def load_heldout(source, split, config, columns, n):
    """(instruction, [reference action lines]) for rows reserved by --holdout."""
    if not HF_OPTS["holdout"]:
        raise SystemExit("Hold-out evaluation needs --holdout COL MOD.")
    c_inst, c_resp = columns[0], columns[2]
    out = []
    for r in _iter_rows(source, split, 0, config, shuffle=False):
        if not _is_holdout(r):
            continue
        resp = r.get(c_resp)
        ref = [str(x).strip() for x in resp] if isinstance(resp, (list, tuple)) \
            else _s(resp).split("\n")
        ref = [x for x in ref if x]
        if _s(r.get(c_inst)) and ref:
            out.append((_s(r.get(c_inst)), ref))
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------ simulated computer agent
def _load_sim():
    try:
        import sim_computer as sc
    except ImportError:
        raise SystemExit("--agent needs sim_computer.py in the same folder as this script.")
    return sc


def make_policy(model, args):
    """instruction -> list of action lines, produced by the neural model."""
    sc = _load_sim()

    def policy(instruction):
        toks = model.generate(instruction, length=args.agent_len, deconstruction_rounds=0,
                              temperature=args.temperature, top_k=args.top_k, raw=True)
        return sc.tokens_to_actions(toks)
    return policy


def run_agent_eval(model, args, n):
    sc = _load_sim()
    t0 = time.time()
    if args.agent_data == "mind2web":
        tasks = load_heldout(args.dataset, args.split, args.config, args.columns, n)
        if not tasks:
            print("No held-out tasks found (check --holdout).")
            return
        r = sc.evaluate_web(make_policy(model, args), tasks, verbose=args.show_screens)
        print(f"\nWeb-agent evaluation on {r['n']} held-out human trajectories "
              f"({time.time() - t0:.0f}s), steps compared position by position:")
        print(f"  step accuracy (all four right) {r['step_accuracy']:.1%}")
        print(f"    element type {r['etype_accuracy']:.1%} | operation {r['op_accuracy']:.1%} | "
              f"label {r['label_accuracy']:.1%} | value {r['value_accuracy']:.1%}")
        print(f"  whole trajectory exact match   {r['exact_match']:.1%}")
        print(f"  same (type, operation) sequence {r['op_sequence_match']:.1%}")
        print(f"  actions the simulator accepts  {r['valid_actions']:.1%}")
        print(f"  mean length: predicted {r['avg_pred_len']:.1f} vs human {r['avg_ref_len']:.1f}")
        return
    res = sc.evaluate(make_policy(model, args), n=n, verbose=args.show_screens)
    print(f"\nAgent evaluation on {n} fresh simulated tasks ({time.time() - t0:.0f}s):")
    print(f"  goal reached        {res['success']:.1%}")
    print(f"  exact expert match  {res['exact_match']:.1%}")
    print(f"  emitted 'done'      {res['finished']:.1%}")
    print(f"  valid actions       {res['valid_actions']:.1%}")
    for k, v in res["per_kind"].items():
        print(f"    {k:<16} {v:.0%}")


def agent_repl(model, args):
    sc = _load_sim()
    policy = make_policy(model, args)
    env = sc.SimWeb if args.agent_data == "mind2web" else sc.SimComputer
    print("\nSimulated agent. Nothing you type here touches your real machine or the web:")
    print("every action is executed against an in-memory simulation.")
    while True:
        try:
            task = input("\nTask (or EXIT): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if task.upper() == "EXIT":
            break
        if not task:
            continue
        actions = policy(task)
        if not actions:
            print("The agent produced no actions.")
            continue
        sim, trace = env(), []
        for a in actions[:40]:                       # execute one by one to show each screen
            ok, msg = sim.step(a)
            trace.append((a, ok, msg))
            print(f"  {len(trace):>2}. {'ok ' if ok else 'ERR'} {a:<34} {msg}")
            if args.show_screens:
                print("     " + sim.screen().replace("\n", "\n     "))
            if sim.finished:
                break
        if not args.show_screens:
            print("\nFinal simulated screen:\n  " + sim.screen().replace("\n", "\n  "))
        changes = sim.diff()
        print("Changes to the virtual disk:", "; ".join(changes) if changes else "none")
        if sim.needs_done and not sim.finished:
            print("(the agent never said 'done')")
        print("[simulation only - no real files, apps or network were touched]")


# ------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tatsu-lab/alpaca",
                    help="HF dataset id or local .jsonl file (.json loads fully into RAM)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--config", default=None, help="HF dataset config / subset name")
    ap.add_argument("--columns", nargs=3, metavar=("INST", "CTX", "RESP"),
                    default=["problem", "-", "solution"],
                    help="column names; use - for no context column")
    ap.add_argument("--chat-text", default=None,
                    help="chat-log mode: column with the message text (pairs consecutive messages)")
    ap.add_argument("--chat-group", default=None,
                    help="chat-log mode: column identifying the room/thread")
    ap.add_argument("--inspect", action="store_true",
                    help="print the dataset's configs, columns and first rows, then exit")
    ap.add_argument("--max-bytes", type=float, default=2e9,
                    help="approx. characters of text to ingest (default 2e9 = ~2 GB)")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--cache-dir", default="magicoder_cache", help="encoded corpus location")
    ap.add_argument("--rebuild", action="store_true", help="re-tokenize even if a cache exists")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--vocab-examples", type=int, default=100_000)
    ap.add_argument("--max-vocab", type=int, default=16000)
    ap.add_argument("--max-inst", type=int, default=128, help="instruction tokens kept")
    ap.add_argument("--max-resp", type=int, default=512, help="response tokens kept")
    ap.add_argument("--epochs", "--passes", dest="passes", type=float, default=5.0,
                    help="passes over the corpus's target tokens (fractions allowed)")
    ap.add_argument("--max-minutes", type=float, default=None, help="wall-clock training cap")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--save", default="magicoder.npz", help="save trained model to this .npz")
    ap.add_argument("--checkpoint-every", type=int, default=20000, help="steps (needs --save)")
    ap.add_argument("--load", default=None, help="load a saved .npz instead of training")
    ap.add_argument("--tolerance", type=int, default=2,
                    help="inconsistency score (repeats) that makes a text 'paradoxical'; "
                         "1 = most sensitive")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=140)
    ap.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: cuda if available)")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 tensor-core matmuls (faster on Ampere+, slightly lower precision)")
    ap.add_argument("--gradcheck", action="store_true",
                    help="verify manual gradients against torch.autograd, then exit")
    ap.add_argument("--embed-dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--agent", action="store_true",
                    help="simulated computer agent: train on synthetic action traces and run "
                         "the model's actions on an in-memory fake computer (sim_computer.py)")
    ap.add_argument("--agent-data", choices=["synthetic", "mind2web"], default="synthetic",
                    help="agent training data: generated tasks, or the downloadable "
                         "osunlp/Mind2Web human web-action trajectories")
    ap.add_argument("--hf-files", default=None,
                    help="parquet/json file glob to stream instead of the dataset's own files")
    ap.add_argument("--read-columns", nargs="+", default=None,
                    help="only read these columns (parquet), skipping huge unused ones")
    ap.add_argument("--holdout", nargs=2, metavar=("COL", "MOD"), default=None,
                    help="reserve rows with crc32(COL) %% MOD == 0 for evaluation")
    ap.add_argument("--tasks-file", default="agent_tasks.jsonl",
                    help="agent training set (generated if missing)")
    ap.add_argument("--make-tasks", type=int, default=None,
                    help="write N synthetic (instruction, action trace) pairs to --tasks-file")
    ap.add_argument("--eval-agent", type=int, default=None,
                    help="score the agent on N fresh simulated tasks (agent default: 200)")
    ap.add_argument("--agent-len", type=int, default=160, help="max tokens per agent plan")
    ap.add_argument("--show-screens", action="store_true",
                    help="print the simulated screen after every action")
    ap.add_argument("--train-only", action="store_true", help="skip the interactive prompt")
    args = ap.parse_args()

    if args.make_tasks and not args.agent:               # just write the dataset and stop
        _load_sim().write_tasks(args.tasks_file, args.make_tasks)
        print(f"Wrote {args.make_tasks:,} tasks to {args.tasks_file}")
        return
    if args.agent:
        if args.agent_data == "mind2web":
            preset = dict(dataset="osunlp/Mind2Web", columns=["confirmed_task", "-", "action_reprs"],
                          cache_dir="agent_web_cache", save="agent_web.npz", context=8,
                          max_inst=48, max_resp=320, max_vocab=12000, passes=30.0, batch=512,
                          temperature=0.3, top_k=20, embed_dim=64, hidden=256,
                          checkpoint_every=5000, holdout=["annotation_id", "10"],
                          read_columns=["annotation_id", "confirmed_task", "action_reprs"],
                          hf_files="hf://datasets/osunlp/Mind2Web@~parquet/default/train/*.parquet")
        else:
            preset = dict(dataset=args.tasks_file, columns=["instruction", "-", "response"],
                          cache_dir="agent_cache", save="agent.npz", context=8, max_inst=24,
                          max_resp=160, max_vocab=3000, passes=30.0, batch=1024,
                          temperature=0.3, top_k=20, embed_dim=64, hidden=256,
                          checkpoint_every=5000)
        for k, v in preset.items():                      # only where you left the default
            if getattr(args, k) == ap.get_default(k):
                setattr(args, k, v)
        if args.eval_agent is None:
            args.eval_agent = 100 if args.agent_data == "mind2web" else 200
        if (args.agent_data == "synthetic" and not args.load
                and (args.make_tasks or not os.path.exists(args.tasks_file))):
            n = args.make_tasks or 30000
            _load_sim().write_tasks(args.tasks_file, n)
            print(f"Wrote {n:,} synthetic tasks to {args.tasks_file}")
            args.rebuild = True                          # the cached corpus would be stale
    HF_OPTS.update(data_files=args.hf_files, read_columns=args.read_columns,
                   holdout=tuple(args.holdout) if args.holdout else None)

    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")

    if args.inspect:
        inspect_dataset(args.dataset, args.split, args.config)
        return

    if args.load:
        model = InstructGenerator.load(args.load, device=device)
        print(f"Loaded model from {args.load} (vocab {len(model.vocab)}).")
    else:
        meta_path = os.path.join(args.cache_dir, "meta.json")
        if args.rebuild or not os.path.exists(meta_path):
            build_corpus(args.dataset, args.cache_dir, args.split, int(args.max_bytes),
                         args.max_examples, args.max_vocab, args.max_inst, args.max_resp,
                         args.vocab_examples, args.workers,
                         columns=tuple(args.columns) if args.columns else None,
                         config=args.config,
                         chat=(args.chat_text, args.chat_group) if args.chat_text else None)
        else:
            print(f"Reusing encoded corpus in {args.cache_dir}/ (use --rebuild to redo).")
        corpus = Corpus(args.cache_dir)
        model = InstructGenerator(context=args.context, embed_dim=args.embed_dim,
                                  hidden=args.hidden, max_vocab=args.max_vocab,
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

    if args.agent and args.eval_agent:
        run_agent_eval(model, args, args.eval_agent)
    if args.train_only:
        return
    if args.agent:
        agent_repl(model, args)
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
