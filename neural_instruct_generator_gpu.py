#!/usr/bin/env python3
"""Neural instruction-following generator, PyTorch / GPU version, built for ~2 GB of text.

Model: P(next response token | last N response tokens, bag-of-words of the instruction)
  window embeddings  ┐
                     ├─ concat -> tanh hidden layer -> softmax over vocab
  mean instruction   ┘
  embedding (own table)

Pipeline for large data
  1. Stream a HuggingFace dataset (shuffled) until --target-gb of raw text has been seen,
     and cache it as a local JSONL file (so the full dataset is never downloaded or held in RAM).
  2. Count words on a sample of that file (multiprocessing) to build the vocab.
  3. Tokenize / encode the whole file in parallel into compact int16 arrays.
  4. Keep the token arrays ON THE GPU and build every (window, instruction, target) batch there,
     so there is no dataloader and no per-position Python loop.
  5. Train with bf16 autocast, fused AdamW, warmup + linear decay.

The deconstruct / regenerate loop from the original script is unchanged.

Usage:
  pip install torch datasets numpy tqdm
  python neural_instruct_generator_gpu.py --target-gb 2 --save model.pt
  python neural_instruct_generator_gpu.py --load model.pt
  python neural_instruct_generator_gpu.py --dataset my_data.jsonl      # local file
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# Special tokens occupy the first ids.
SPECIALS = ["<unk>", "<pad>", "<|resp|>", "<|end|>"]
UNK, PAD, RESP, END = range(4)
SPECIAL_SET = set(SPECIALS)
TOKEN_RE = re.compile(r"<\|\w+\|>|\w+|[^\w\s]", re.UNICODE)
CHARS_PER_TOKEN_CAP = 12  # we only tokenize the first max_tokens*12 chars of each text (big speedup)

COLUMN_CANDIDATES = {
    "instruction": ["instruction", "prompt", "question"],
    "context": ["input", "context"],
    "response": ["output", "response", "answer", "completion"],
}
HUMAN_ROLES = {"human", "user"}
BOT_ROLES = {"gpt", "assistant", "bot"}


# ---------------------------------------------------------------- data loading
def _pick(keys, candidates):
    for c in candidates:
        if c in keys:
            return c
    return None


def _first_by_role(turns, roles):
    for t in turns:
        role = t.get("from") or t.get("role")
        if role in roles:
            return t.get("value") or t.get("content") or ""
    return ""


def row_to_pair(row, columns=None):
    """Turn one dataset row into (instruction, response). Handles flat columns,
    Alpaca-style input/context, and chat formats ('conversations' / 'messages')."""
    if columns:
        c_inst, c_ctx, c_resp = columns
    else:
        for key in ("conversations", "messages"):
            turns = row.get(key)
            if isinstance(turns, str):  # some datasets store the chat as a JSON string
                try:
                    turns = json.loads(turns)
                except ValueError:
                    turns = None
            if isinstance(turns, list):
                return (_first_by_role(turns, HUMAN_ROLES).strip(),
                        _first_by_role(turns, BOT_ROLES).strip())
        keys = set(row.keys())
        c_inst = _pick(keys, COLUMN_CANDIDATES["instruction"])
        c_ctx = _pick(keys, COLUMN_CANDIDATES["context"])
        c_resp = _pick(keys, COLUMN_CANDIDATES["response"])
        if not c_inst or not c_resp:
            raise ValueError(f"Couldn't detect columns in {sorted(keys)}; use --columns.")
    inst = (row.get(c_inst) or "").strip()
    ctx = (row.get(c_ctx) or "").strip() if c_ctx else ""
    resp = (row.get(c_resp) or "").strip()
    return f"{inst} {ctx}".strip(), resp


def iter_rows(source, config, split, seed):
    if source.endswith(".jsonl"):
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif source.endswith(".json"):
        with open(source, "r", encoding="utf-8") as f:
            rows = json.load(f)
        random.Random(seed).shuffle(rows)
        yield from rows
    else:
        from datasets import (get_dataset_split_names, interleave_datasets,
                              load_dataset)  # lazy: local files need no extra deps
        names = get_dataset_split_names(source, config)
        if split and split != "all":
            if split in names:
                names = [split]
            else:
                tqdm.write(f"Split '{split}' not found; using all splits instead.")
        elif "train" in names:
            names = ["train"]
        tqdm.write(f"Streaming splits: {names}")
        parts = [load_dataset(source, config, split=n, streaming=True)
                 .shuffle(seed=seed, buffer_size=5_000) for n in names]
        # Interleave so a 2 GB budget samples every split (e.g. every task category)
        ds = parts[0] if len(parts) == 1 else interleave_datasets(
            parts, seed=seed, stopping_strategy="all_exhausted")
        yield from ds


def prepare_pairs(source, config, split, target_gb, cache_dir, columns, seed):
    """Stream rows until ~target_gb of raw text is collected; cache as JSONL. Returns path."""
    os.makedirs(cache_dir, exist_ok=True)
    slug = re.sub(r"[^\w.-]+", "_", os.path.basename(source) if os.path.exists(source) else source)
    path = os.path.join(cache_dir, f"{slug}_{target_gb:g}gb.jsonl")
    if os.path.exists(path):
        print(f"Using cached pairs: {path} ({os.path.getsize(path) / 1e9:.2f} GB)")
        return path

    budget = int(target_gb * 1e9)
    seen = kept = 0
    with open(path + ".tmp", "w", encoding="utf-8") as out, \
            tqdm(total=budget, unit="B", unit_scale=True, unit_divisor=1000,
                 desc="Downloading/caching", smoothing=0.1) as bar:
        for row in iter_rows(source, config, split, seed):
            inst, resp = row_to_pair(row, columns)
            if inst and resp:
                out.write(json.dumps([inst, resp], ensure_ascii=False) + "\n")
                nbytes = len(inst.encode("utf-8")) + len(resp.encode("utf-8"))
                seen += nbytes
                kept += 1
                bar.update(min(nbytes, max(budget - bar.n, 0)))
                if kept % 2000 == 0:
                    bar.set_postfix(rows=f"{kept:,}")
            if seen >= budget:
                break
    os.replace(path + ".tmp", path)
    if seen < budget * 0.95:
        print(f"Warning: source ended at {seen / 1e9:.2f} GB, below the {target_gb:g} GB target.")
    print(f"Cached {kept:,} pairs, {seen / 1e9:.2f} GB raw text -> {path}")
    return path


def tok(text, limit=None):
    """Tokenize (optionally only the first `limit` tokens), dropping literal special-token strings."""
    if limit:
        text = text[: limit * CHARS_PER_TOKEN_CAP]
    toks = [t for t in TOKEN_RE.findall(text) if t not in SPECIAL_SET]
    return toks[:limit] if limit else toks


# ------------------------------------------------- multiprocessing workers (top level)
_W = {}


def _init_worker(stoi, max_inst, max_resp, dtype_name):
    _W.update(stoi=stoi, max_inst=max_inst, max_resp=max_resp, dtype=np.dtype(dtype_name))


def _iter_chunk(path, start, end):
    with open(path, "rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()  # partial line belongs to the previous chunk
        while f.tell() <= end:
            line = f.readline()
            if not line:
                break
            yield json.loads(line)


def _count_chunk(args):
    path, start, end, max_inst, max_resp = args
    c = Counter()
    for inst, resp in _iter_chunk(path, start, end):
        c.update(tok(inst, max_inst))
        c.update(tok(resp, max_resp))
    return c


def _encode_chunk(args):
    path, start, end = args
    stoi, mi, mr, dt = _W["stoi"], _W["max_inst"], _W["max_resp"], _W["dtype"]
    inst_rows, flat, lens = [], [], []
    for inst, resp in _iter_chunk(path, start, end):
        it = [stoi.get(t, UNK) for t in tok(inst, mi)]
        inst_rows.append(it + [PAD] * (mi - len(it)))
        rt = [stoi.get(t, UNK) for t in tok(resp, mr)] + [END]
        flat.extend(rt)
        lens.append(len(rt))
    if not lens:
        return np.zeros((0, mi), dt), np.zeros(0, dt), np.zeros(0, np.int64)
    return np.array(inst_rows, dtype=dt), np.array(flat, dtype=dt), np.array(lens, dtype=np.int64)


def _ranges(total, parts):
    size = max(1, (total + parts - 1) // parts)
    return [(s, min(s + size, total)) for s in range(0, total, size)]


# ----------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, V, N, d, h):
        super().__init__()
        self.N = N
        self.E = nn.Embedding(V, d)  # response-window embeddings
        self.C = nn.EmbeddingBag(V, d, mode="mean", padding_idx=PAD)  # instruction (mean, PAD ignored)
        self.fc1 = nn.Linear((N + 1) * d, h)
        self.fc2 = nn.Linear(h, V)
        with torch.no_grad():
            self.E.weight.normal_(0, 0.1)
            self.C.weight.normal_(0, 0.1)
            self.C.weight[PAD] = 0
            self.fc1.weight.normal_(0, 1 / np.sqrt((N + 1) * d))
            self.fc2.weight.normal_(0, 1 / np.sqrt(h))
            self.fc1.bias.zero_()
            self.fc2.bias.zero_()

    def forward(self, ctx, inst):
        """ctx: (B, N) window ids; inst: (B, L) instruction ids (PAD-padded)."""
        x = torch.cat([self.E(ctx).reshape(ctx.shape[0], -1), self.C(inst)], dim=1)
        return self.fc2(torch.tanh(self.fc1(x)))


class Corpus:
    """Flat token store for responses. Training position q is the q-th (response token + END)
    over all examples; its context window is rebuilt on the fly from the flat array, so no
    per-position window/example arrays are stored."""

    def __init__(self, inst, flat, lens, N, data_device):
        self.N = N
        self.dev = data_device
        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        self.inst = torch.from_numpy(inst).to(data_device)          # (E, max_inst) int16/32
        self.flat = torch.from_numpy(flat).to(data_device)          # (T,) int16/32
        self.offsets = torch.from_numpy(offsets).to(data_device)    # (E,) int64
        self.T = int(flat.shape[0])
        self.E = int(inst.shape[0])
        self.k = torch.arange(N, 0, -1, device=data_device)         # distance back: N..1

    @property
    def nbytes(self):
        return sum(t.element_size() * t.numel() for t in (self.inst, self.flat, self.offsets))

    def batch(self, q):
        e = torch.searchsorted(self.offsets, q, right=True) - 1     # example of each position
        rel = (q - self.offsets[e])[:, None]                        # position within response
        k = self.k
        tok_ = self.flat[(q[:, None] - k).clamp_(min=0)].long()
        ctx = torch.where(k <= rel, tok_,
                          torch.where(k == rel + 1, torch.full_like(tok_, RESP),
                                      torch.full_like(tok_, PAD)))
        return ctx, self.inst[e].long(), self.flat[q].long()


# ------------------------------------------------------------------- generator
class InstructGenerator:
    def __init__(self, context=3, embed_dim=128, hidden=1024, max_vocab=30000,
                 max_inst=32, max_resp=128, seed=0, device=None):
        self.N, self.d, self.h = context, embed_dim, hidden
        self.max_vocab, self.max_inst, self.max_resp = max_vocab, max_inst, max_resp
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = seed
        self.vocab, self.stoi, self.net = [], {}, None

    def _encode(self, tokens):
        return [self.stoi.get(t, UNK) for t in tokens]

    def _pad_inst(self, ids):
        ids = ids[:self.max_inst]
        return ids + [PAD] * (self.max_inst - len(ids))

    # -- data -> GPU corpus
    def build_corpus(self, path, workers, vocab_sample_mb, data_device="auto"):
        workers = workers or max(1, cpu_count() - 1)
        total = os.path.getsize(path)
        t0 = time.time()

        # pass 1: vocab from a sample of the (already shuffled) file
        sample_end = min(total, int(vocab_sample_mb * 1e6))
        jobs = [(path, s, e, self.max_inst, self.max_resp) for s, e in _ranges(sample_end, workers * 8)]
        counts = Counter()
        with Pool(workers) as pool:
            for c in tqdm(pool.imap_unordered(_count_chunk, jobs), total=len(jobs),
                          desc="Counting words"):
                counts.update(c)
        common = [w for w, _ in counts.most_common(self.max_vocab - len(SPECIALS))]
        self.vocab = SPECIALS + common
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        dtype_name = "int16" if V < 32768 else "int32"
        print(f"Vocab {V} (from {sample_end / 1e6:.0f} MB sample, {time.time() - t0:.0f}s)")

        # pass 2: encode everything in parallel
        jobs = [(path, s, e) for s, e in _ranges(total, workers * 16)]
        parts = []
        with Pool(workers, initializer=_init_worker,
                  initargs=(self.stoi, self.max_inst, self.max_resp, dtype_name)) as pool:
            for r in tqdm(pool.imap_unordered(_encode_chunk, jobs), total=len(jobs),
                          desc="Encoding"):
                parts.append(r)
        inst = np.concatenate([p[0] for p in parts])
        flat = np.concatenate([p[1] for p in parts])
        lens = np.concatenate([p[2] for p in parts])
        del parts

        est = inst.nbytes + flat.nbytes + lens.nbytes
        if data_device == "auto":
            use_gpu = self.device.type == "cuda" and est < 0.5 * torch.cuda.mem_get_info()[0]
            data_device = self.device if use_gpu else torch.device("cpu")
        else:
            data_device = torch.device(data_device)
        corpus = Corpus(inst, flat, lens, self.N, data_device)
        print(f"{corpus.E:,} examples, {corpus.T:,} training positions, "
              f"{corpus.nbytes / 1e9:.2f} GB on {data_device} ({time.time() - t0:.0f}s total)")
        return corpus

    # -- training
    def train(self, corpus, epochs=1.0, batch=8192, lr=2e-3, warmup=200,
              log_every=100, compile_model=False):
        torch.manual_seed(self.seed)
        dev = self.device
        if dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.net = Net(len(self.vocab), self.N, self.d, self.h).to(dev)
        fwd = torch.compile(self.net) if compile_model else self.net
        use_amp = dev.type == "cuda" and torch.cuda.is_bf16_supported()
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=0.0,
                                fused=dev.type == "cuda")
        steps = max(1, int(epochs * corpus.T / batch))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(warmup, 1)) * (1 - 0.9 * s / steps))
        n_params = sum(p.numel() for p in self.net.parameters())
        print(f"Training {n_params / 1e6:.1f}M params on {dev} "
              f"(bf16={use_amp}), {steps:,} steps x batch {batch} (~{epochs:g} epochs)")

        self.net.train()
        running = torch.zeros((), device=dev)
        bar = tqdm(range(1, steps + 1), desc="Training", unit="step", smoothing=0.05)
        for step in bar:
            q = torch.randint(0, corpus.T, (batch,), device=corpus.dev)  # sampling with replacement
            ctx, inst, y = (t.to(dev, non_blocking=True) for t in corpus.batch(q))
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = F.cross_entropy(fwd(ctx, inst), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.detach()
            if step % log_every == 0 or step == steps:  # only sync with the GPU occasionally
                n = log_every if step % log_every == 0 else step % log_every
                bar.set_postfix(loss=f"{running.item() / n:.4f}",
                                lr=f"{sched.get_last_lr()[0]:.2e}")
                running.zero_()
        self.net.eval()

    # -- save / load
    def save(self, path):
        cfg = dict(N=self.N, d=self.d, h=self.h, max_vocab=self.max_vocab,
                   max_inst=self.max_inst, max_resp=self.max_resp)
        torch.save({"cfg": cfg, "vocab": self.vocab, "state": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path, device=None):
        z = torch.load(path, map_location="cpu", weights_only=True)
        c = z["cfg"]
        m = cls(context=c["N"], embed_dim=c["d"], hidden=c["h"], max_vocab=c["max_vocab"],
                max_inst=c["max_inst"], max_resp=c["max_resp"], device=device)
        m.vocab = z["vocab"]
        m.stoi = {w: i for i, w in enumerate(m.vocab)}
        m.net = Net(len(m.vocab), m.N, m.d, m.h)
        m.net.load_state_dict(z["state"])
        m.net.to(m.device).eval()
        return m

    # -- text helpers
    @staticmethod
    def detokenize(tokens):
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:)\]])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        return re.sub(r"(\w)\s'\s(\w)", r"\1'\2", text)   # don ' t -> don't

    def deconstruct(self, text):
        """Collapse immediate token repeats and repeated adjacent phrase loops."""
        tokens = TOKEN_RE.findall(text)
        if not tokens:
            return ""
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
        return self.detokenize(collapsed)

    # -- generation
    @torch.no_grad()
    def _sample(self, window, inst_t, temperature, top_k):
        ctx = torch.tensor([window], device=self.device)
        logits = self.net(ctx, inst_t)[0].float() / max(temperature, 1e-4)
        logits[[UNK, PAD, RESP]] = -float("inf")
        if top_k and top_k < logits.numel():
            cutoff = torch.topk(logits, top_k).values[-1]
            logits[logits < cutoff] = -float("inf")
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1))

    def _continue(self, prefix_tokens, inst_t, length, temperature, top_k):
        """Continue a response from prefix_tokens; stops at <|end|> or `length` tokens."""
        ids = [PAD] * self.N + [RESP] + self._encode(prefix_tokens)
        window = ids[-self.N:]
        out = []
        for _ in range(length):
            t = self._sample(window, inst_t, temperature, top_k)
            if t == END:
                break
            out.append(self.vocab[t])
            window = window[1:] + [t]
        return out

    def generate(self, instruction, length=64, deconstruction_rounds=1,
                 temperature=0.8, top_k=40):
        """Respond to an instruction, then deconstruct (replacing the text) and regenerate."""
        if self.net is None:
            raise RuntimeError("Train or load a model first.")
        if length < 0:
            raise ValueError("Token length must be non-negative.")
        if deconstruction_rounds < 0:
            raise ValueError("Deconstruction rounds must be non-negative.")

        row = self._pad_inst(self._encode(tok(instruction, self.max_inst)))
        inst_t = torch.tensor([row], device=self.device)
        resp = self._continue([], inst_t, length, temperature, top_k)

        for _ in range(deconstruction_rounds):
            reduced = self.deconstruct(self.detokenize(resp))   # replacement, not a side analysis
            resp = TOKEN_RE.findall(reduced)
            resp += self._continue(resp, inst_t, length, temperature, top_k)
        return self.detokenize(resp)


# ------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="microsoft/orca-agentinstruct-1M-v1",
                    help="HF dataset id or local .json/.jsonl file")
    ap.add_argument("--config", default=None, help="HF dataset config name, if any")
    ap.add_argument("--split", default=None,
                    help='split name, or "all"; default: "train" if it exists, '
                         'otherwise every split interleaved')
    ap.add_argument("--columns", nargs=3, metavar=("INST", "CTX", "RESP"), default=None,
                    help="override column names; use '' for no context column")
    ap.add_argument("--target-gb", type=float, default=2.0, help="raw text to collect")
    ap.add_argument("--cache-dir", default="./data_cache")
    ap.add_argument("--workers", type=int, default=0, help="CPU processes (0 = all but one)")
    ap.add_argument("--vocab-sample-mb", type=float, default=300)
    ap.add_argument("--data-device", default="auto", help="auto | cuda | cpu (where token arrays live)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--max-vocab", type=int, default=30000)
    ap.add_argument("--max-inst", type=int, default=32)
    ap.add_argument("--max-resp", type=int, default=128)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="save trained model to this .pt")
    ap.add_argument("--load", default=None, help="load a saved .pt instead of training")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.load:
        model = InstructGenerator.load(args.load)
        print(f"Loaded model from {args.load} (vocab {len(model.vocab)}, device {model.device}).")
    else:
        columns = tuple(c or None for c in args.columns) if args.columns else None
        path = prepare_pairs(args.dataset, args.config, args.split, args.target_gb,
                             args.cache_dir, columns, args.seed)
        model = InstructGenerator(context=args.context, embed_dim=args.embed_dim,
                                  hidden=args.hidden, max_vocab=args.max_vocab,
                                  max_inst=args.max_inst, max_resp=args.max_resp, seed=args.seed)
        corpus = model.build_corpus(path, args.workers, args.vocab_sample_mb, args.data_device)
        model.train(corpus, epochs=args.epochs, batch=args.batch, lr=args.lr,
                    compile_model=args.compile)
        del corpus
        if args.save:
            model.save(args.save)
            print(f"Saved to {args.save}")

    while True:
        instruction = input("\nInstruction (or EXIT): ").strip()
        if instruction.upper() == "EXIT":
            break
        if not instruction:
            continue
        try:
            length = int(input("Max tokens per pass: "))
            rounds = int(input("Deconstruction/regeneration rounds (0 = off): "))
            print("\n" + model.generate(instruction, length, rounds))
        except ValueError as e:
            print("Invalid input:", e)
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
