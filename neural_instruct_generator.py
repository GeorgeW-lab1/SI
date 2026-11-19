#!/usr/bin/env python3
"""Neural instruction-following generator trained on a HuggingFace instruction dataset.

Model: P(next response token | last N response tokens, bag-of-words of the instruction)
  window embeddings  ┐
                     ├─ concat -> tanh hidden layer -> softmax over vocab
  mean instruction   ┘
  embedding (own table)

Pure NumPy with manual backprop and Adam. The deconstruct / regenerate loop from the
original script is kept: generate a response, collapse repeats/loops (replacing the
text), then let the model continue from the reduced context.

Usage:
  pip install datasets numpy
  python neural_instruct_generator.py --dataset tatsu-lab/alpaca --max-examples 10000
  python neural_instruct_generator.py --dataset my_data.jsonl      # offline / local
"""

import argparse
import json
import random
import re
from collections import Counter

import numpy as np

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


# ---------------------------------------------------------------- data loading
def _pick(keys, candidates):
    for c in candidates:
        if c in keys:
            return c
    return None


def load_pairs(source, split="train", max_examples=10000, seed=0, columns=None):
    """Return a list of (instruction_text, response_text).

    source: a HuggingFace dataset id (e.g. "tatsu-lab/alpaca",
            "databricks/databricks-dolly-15k") or a local .json/.jsonl file.
    columns: optional (instruction_col, context_col_or_None, response_col).
    """
    if source.endswith((".json", ".jsonl")):
        with open(source, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if source.endswith(".jsonl") or not raw.startswith("["):
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            rows = json.loads(raw)
        random.Random(seed).shuffle(rows)
        rows = rows[:max_examples]
    else:
        from datasets import load_dataset  # imported lazily so local use needs no extra deps
        ds = load_dataset(source, split=split)
        ds = ds.shuffle(seed=seed)
        rows = [ds[i] for i in range(min(max_examples, len(ds)))]

    if not rows:
        raise ValueError("Dataset is empty.")
    keys = set(rows[0].keys())
    if columns:
        c_inst, c_ctx, c_resp = columns
    else:
        c_inst = _pick(keys, COLUMN_CANDIDATES["instruction"])
        c_ctx = _pick(keys, COLUMN_CANDIDATES["context"])
        c_resp = _pick(keys, COLUMN_CANDIDATES["response"])
    if not c_inst or not c_resp:
        raise ValueError(f"Couldn't detect columns in {sorted(keys)}; pass columns=(...).")

    pairs = []
    for r in rows:
        inst = (r.get(c_inst) or "").strip()
        ctx = (r.get(c_ctx) or "").strip() if c_ctx else ""
        resp = (r.get(c_resp) or "").strip()
        if inst and resp:
            pairs.append((f"{inst} {ctx}".strip(), resp))
    print(f"Loaded {len(pairs)} instruction/response pairs "
          f"(columns: {c_inst}, {c_ctx}, {c_resp}).")
    return pairs


def tok(text):
    """Tokenize, dropping any literal special-token strings found in the data."""
    return [t for t in TOKEN_RE.findall(text) if t not in SPECIAL_SET]


# ----------------------------------------------------------------------- model
class InstructGenerator:
    def __init__(self, context=3, embed_dim=48, hidden=192, max_vocab=8000,
                 max_inst=32, max_resp=64, seed=0):
        self.N = context
        self.d = embed_dim
        self.h = hidden
        self.max_vocab = max_vocab
        self.max_inst = max_inst
        self.max_resp = max_resp
        self.rng = np.random.default_rng(seed)
        self.vocab = []
        self.stoi = {}
        self.params = None

    # -- vocab / params
    def _build_vocab(self, token_lists):
        counts = Counter(t for toks in token_lists for t in toks)
        common = [w for w, _ in counts.most_common(self.max_vocab - len(SPECIALS))]
        self.vocab = SPECIALS + common
        self.stoi = {w: i for i, w in enumerate(self.vocab)}

    def _encode(self, tokens):
        return [self.stoi.get(t, UNK) for t in tokens]

    def _init_params(self):
        V, d, h, N = len(self.vocab), self.d, self.h, self.N
        r = self.rng
        in_dim = (N + 1) * d
        self.params = {
            "E": r.normal(0, 0.1, (V, d)),       # response-window embeddings
            "C": r.normal(0, 0.1, (V, d)),       # instruction (condition) embeddings
            "W1": r.normal(0, 1 / np.sqrt(in_dim), (in_dim, h)),
            "b1": np.zeros(h),
            "W2": r.normal(0, 1 / np.sqrt(h), (h, V)),
            "b2": np.zeros(V),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    # -- forward / backward
    def _forward(self, ctx, inst):
        """ctx: (B, N) window ids; inst: (B, L) instruction ids (PAD-padded)."""
        p = self.params
        B = len(ctx)
        emb = p["E"][ctx].reshape(B, -1)                       # (B, N*d)
        mask = (inst != PAD).astype(np.float64)                # (B, L)
        cnt = np.maximum(mask.sum(1, keepdims=True), 1.0)      # (B, 1)
        cond = (p["C"][inst] * mask[:, :, None]).sum(1) / cnt  # (B, d)
        x = np.concatenate([emb, cond], axis=1)
        a = np.tanh(x @ p["W1"] + p["b1"])
        logits = a @ p["W2"] + p["b2"]
        return logits, (ctx, inst, mask, cnt, x, a)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _backward(self, probs, y, cache):
        ctx, inst, mask, cnt, x, a = cache
        p = self.params
        B = len(y)
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1
        dlogits /= B
        g = {"W2": a.T @ dlogits, "b2": dlogits.sum(0)}
        dz = (dlogits @ p["W2"].T) * (1 - a ** 2)
        g["W1"] = x.T @ dz
        g["b1"] = dz.sum(0)
        dx = dz @ p["W1"].T
        nd = self.N * self.d
        gE = np.zeros_like(p["E"])
        np.add.at(gE, ctx, dx[:, :nd].reshape(B, self.N, self.d))
        gC = np.zeros_like(p["C"])
        rows = dx[:, None, nd:] * (mask / cnt)[:, :, None]     # (B, L, d)
        np.add.at(gC, inst, rows)
        g["E"], g["C"] = gE, gC
        return g

    def _adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            mh = self._m[k] / (1 - b1 ** self._t)
            vh = self._v[k] / (1 - b2 ** self._t)
            self.params[k] -= lr * mh / (np.sqrt(vh) + eps)

    # -- training
    def _pad_inst(self, ids):
        ids = ids[:self.max_inst]
        return ids + [PAD] * (self.max_inst - len(ids))

    def train(self, pairs, epochs=3, batch=256, lr=3e-3):
        inst_tok = [tok(i)[:self.max_inst] for i, _ in pairs]
        resp_tok = [tok(r)[:self.max_resp] for _, r in pairs]
        self._build_vocab(inst_tok + resp_tok)

        I = np.array([self._pad_inst(self._encode(t)) for t in inst_tok])
        Xs, ys, idx = [], [], []
        for k, rt in enumerate(resp_tok):
            window = [PAD] * (self.N - 1) + [RESP]
            for t in self._encode(rt) + [END]:
                Xs.append(window)
                ys.append(t)
                idx.append(k)
                window = window[1:] + [t]
        X, y, idx = np.array(Xs), np.array(ys), np.array(idx)
        print(f"Vocab {len(self.vocab)}, {len(X)} training positions.")

        self._init_params()
        steps_total = epochs * ((len(X) + batch - 1) // batch)
        step = 0
        for ep in range(epochs):
            perm = self.rng.permutation(len(X))
            total, n = 0.0, 0
            for s in range(0, len(X), batch):
                b = perm[s:s + batch]
                cur_lr = lr * (1 - 0.9 * step / max(steps_total, 1))  # linear decay to 10%
                logits, cache = self._forward(X[b], I[idx[b]])
                probs = self._softmax(logits)
                loss = -np.log(probs[np.arange(len(b)), y[b]] + 1e-12).mean()
                self._adam(self._backward(probs, y[b], cache), cur_lr)
                total += loss * len(b)
                n += len(b)
                step += 1
                if step % 1000 == 0:
                    print(f"  step {step}/{steps_total}  loss {total / n:.4f}")
            print(f"epoch {ep + 1}/{epochs}  loss {total / n:.4f}")

    # -- save / load
    def save(self, path):
        cfg = json.dumps(dict(N=self.N, d=self.d, h=self.h, max_vocab=self.max_vocab,
                              max_inst=self.max_inst, max_resp=self.max_resp))
        np.savez_compressed(path, vocab=np.array(self.vocab), cfg=np.array(cfg), **self.params)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        c = json.loads(str(z["cfg"]))
        m = cls(context=c["N"], embed_dim=c["d"], hidden=c["h"], max_vocab=c["max_vocab"],
                max_inst=c["max_inst"], max_resp=c["max_resp"])
        m.vocab = [str(w) for w in z["vocab"]]
        m.stoi = {w: i for i, w in enumerate(m.vocab)}
        m.params = {k: z[k] for k in ["E", "C", "W1", "b1", "W2", "b2"]}
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
    def _sample(self, window, inst_row, temperature, top_k):
        logits, _ = self._forward(np.array([window]), inst_row[None, :])
        logits = logits[0] / max(temperature, 1e-4)
        for banned in (UNK, PAD, RESP):
            logits[banned] = -np.inf
        if top_k and top_k < len(logits):
            cutoff = np.partition(logits, -top_k)[-top_k]
            logits[logits < cutoff] = -np.inf
        probs = self._softmax(logits[None, :])[0]
        return int(self.rng.choice(len(probs), p=probs))

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
                 temperature=0.8, top_k=40):
        """Respond to an instruction, then deconstruct (replacing the text) and regenerate."""
        if self.params is None:
            raise RuntimeError("Train or load a model first.")
        if length < 0:
            raise ValueError("Token length must be non-negative.")
        if deconstruction_rounds < 0:
            raise ValueError("Deconstruction rounds must be non-negative.")

        inst_row = np.array(self._pad_inst(self._encode(tok(instruction))))
        resp = self._continue([], inst_row, length, temperature, top_k)

        for _ in range(deconstruction_rounds):
            reduced = self.deconstruct(self.detokenize(resp))   # replacement, not a side analysis
            resp = TOKEN_RE.findall(reduced)
            resp += self._continue(resp, inst_row, length, temperature, top_k)
        return self.detokenize(resp)


# ------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tatsu-lab/alpaca",
                    help="HF dataset id or local .json/.jsonl file")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-examples", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--save", default=None, help="save trained model to this .npz")
    ap.add_argument("--load", default=None, help="load a saved .npz instead of training")
    args = ap.parse_args()

    if args.load:
        model = InstructGenerator.load(args.load)
        print(f"Loaded model from {args.load} (vocab {len(model.vocab)}).")
    else:
        pairs = load_pairs(args.dataset, args.split, args.max_examples)
        model = InstructGenerator(context=args.context)
        model.train(pairs, epochs=args.epochs)
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
