#!/usr/bin/env python3
"""Neural-net version of the corpus generator with an iterative text deconstructor.

Instead of counting trigrams, a small feed-forward neural language model
(embeddings -> tanh hidden layer -> softmax) learns P(next token | last N tokens)
by gradient descent. Pure NumPy, manual backprop, Adam optimizer.
"""

import random
import re
from collections import Counter

import numpy as np

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
UNK = "<unk>"


class NeuralGenerator:
    def __init__(self, context=2, embed_dim=32, hidden=128, max_vocab=5000, seed=0):
        self.N = context
        self.d = embed_dim
        self.h = hidden
        self.max_vocab = max_vocab
        self.rng = np.random.default_rng(seed)
        self.words = []
        self.vocab = []
        self.stoi = {}
        self.params = None

    # ------------------------------------------------------------------ data
    def _build_vocab(self, tokens):
        counts = Counter(tokens)
        most_common = [w for w, _ in counts.most_common(self.max_vocab - 1)]
        self.vocab = [UNK] + most_common
        self.stoi = {w: i for i, w in enumerate(self.vocab)}

    def _encode(self, tokens):
        return [self.stoi.get(t, 0) for t in tokens]

    def _init_params(self):
        V, d, h, N = len(self.vocab), self.d, self.h, self.N
        r = self.rng
        self.params = {
            "E": r.normal(0, 0.1, (V, d)),
            "W1": r.normal(0, 1 / np.sqrt(N * d), (N * d, h)),
            "b1": np.zeros(h),
            "W2": r.normal(0, 1 / np.sqrt(h), (h, V)),
            "b2": np.zeros(V),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    # ----------------------------------------------------------------- model
    def _forward(self, ctx):
        """ctx: int array (B, N) -> logits (B, V), cache."""
        p = self.params
        emb = p["E"][ctx]                      # (B, N, d)
        x = emb.reshape(len(ctx), -1)          # (B, N*d)
        a = np.tanh(x @ p["W1"] + p["b1"])     # (B, h)
        logits = a @ p["W2"] + p["b2"]         # (B, V)
        return logits, (ctx, x, a)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _backward(self, probs, y, cache):
        ctx, x, a = cache
        p = self.params
        B = len(y)
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1
        dlogits /= B
        g = {}
        g["W2"] = a.T @ dlogits
        g["b2"] = dlogits.sum(axis=0)
        da = dlogits @ p["W2"].T
        dz = da * (1 - a ** 2)
        g["W1"] = x.T @ dz
        g["b1"] = dz.sum(axis=0)
        dx = (dz @ p["W1"].T).reshape(B, self.N, self.d)
        gE = np.zeros_like(p["E"])
        np.add.at(gE, ctx, dx)
        g["E"] = gE
        return g

    def _adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            mh = self._m[k] / (1 - b1 ** self._t)
            vh = self._v[k] / (1 - b2 ** self._t)
            self.params[k] -= lr * mh / (np.sqrt(vh) + eps)

    # -------------------------------------------------------------- training
    def load_corpus(self, path, epochs=5, batch=256, lr=3e-3):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        self.train_on_text(text, epochs, batch, lr)

    def train_on_text(self, text, epochs=5, batch=256, lr=3e-3):
        self.words = TOKEN_RE.findall(text)
        if len(self.words) <= self.N:
            raise ValueError("Corpus too small for the chosen context size.")
        self._build_vocab(self.words)
        ids = np.array(self._encode(self.words))
        print(f"Loaded {len(self.words)} tokens, vocab {len(self.vocab)}.")

        # Sliding windows: X[i] = ids[i:i+N], y[i] = ids[i+N]
        idx = np.arange(len(ids) - self.N)
        X = np.stack([ids[idx + k] for k in range(self.N)], axis=1)
        y = ids[idx + self.N]

        self._init_params()
        for ep in range(epochs):
            perm = self.rng.permutation(len(X))
            total, n = 0.0, 0
            for s in range(0, len(X), batch):
                b = perm[s:s + batch]
                logits, cache = self._forward(X[b])
                probs = self._softmax(logits)
                loss = -np.log(probs[np.arange(len(b)), y[b]] + 1e-12).mean()
                self._adam(self._backward(probs, y[b], cache), lr)
                total += loss * len(b)
                n += len(b)
            print(f"epoch {ep + 1}/{epochs}  loss {total / n:.4f}")

    # --------------------------------------------------------------- helpers
    @staticmethod
    def detokenize(tokens):
        text = " ".join(tokens)
        return re.sub(r"\s+([.,!?;:])", r"\1", text)

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

    # ------------------------------------------------------------ generation
    def _sample(self, context_tokens, temperature=1.0, top_k=40):
        ids = self._encode(context_tokens[-self.N:])
        while len(ids) < self.N:                 # left-pad short prompts
            ids.insert(0, random.randrange(1, len(self.vocab)))
        logits, _ = self._forward(np.array([ids]))
        logits = logits[0] / max(temperature, 1e-4)
        logits[0] = -np.inf                      # never emit <unk>
        if top_k and top_k < len(logits):
            cutoff = np.partition(logits, -top_k)[-top_k]
            logits[logits < cutoff] = -np.inf
        probs = self._softmax(logits[None, :])[0]
        return self.vocab[self.rng.choice(len(probs), p=probs)]

    def _continue(self, tokens, length, temperature, top_k):
        context = list(tokens[-self.N:]) or [random.choice(self.vocab[1:])]
        out = []
        for _ in range(length):
            tok = self._sample(context, temperature, top_k)
            out.append(tok)
            context = (context + [tok])[-self.N:]
        return out

    def generate(self, prompt, length=100, deconstruction_rounds=1,
                 temperature=0.9, top_k=40):
        """Generate, deconstruct (replacing the text), then regenerate from its reduced context."""
        if self.params is None:
            raise RuntimeError("Load a corpus first.")
        if length < 0:
            raise ValueError("Token length must be non-negative.")
        if deconstruction_rounds < 0:
            raise ValueError("Deconstruction rounds must be non-negative.")

        tokens = TOKEN_RE.findall(prompt.strip())
        tokens += self._continue(tokens, length, temperature, top_k)
        current = self.detokenize(tokens)

        for _ in range(deconstruction_rounds):
            current = self.deconstruct(current)
            tokens = TOKEN_RE.findall(current)
            tokens += self._continue(tokens, length, temperature, top_k)
            current = self.detokenize(tokens)
        return current


def main():
    model = NeuralGenerator()
    path = input("Corpus file (.txt): ").strip().strip('"')
    epochs = int(input("Training epochs (e.g. 5): ") or 5)
    model.load_corpus(path, epochs=epochs)

    while True:
        prompt = input("\nPrompt (or EXIT): ").strip()
        if prompt.upper() == "EXIT":
            break
        if not prompt:
            continue
        try:
            length = int(input("Tokens per pass: "))
            rounds = int(input("Deconstruction/regeneration rounds (0 = off): "))
            print("\n" + model.generate(prompt, length, rounds))
        except ValueError as e:
            print("Invalid input:", e)
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
