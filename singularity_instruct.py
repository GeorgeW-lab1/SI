"""
Singularity Computer - instruct model
=====================================
Trains on an instruction dataset (Alpaca format: instruction / input / output) and answers prompts.

Chat template the model is trained on (as word tokens):
    <user> instruction [input] <bot> response <end>

Components
  * Plate         fixed noisy physical reservoir, driven word by word (acoustic codes)
  * Head          small trained digital MLP on the plate readout -> next-word probabilities
  * N-gram        word trigram counts over the WHOLE dataset (memory-efficient numpy arrays)
  * Retriever     finds the most similar training instructions; words from their responses
                  get boosted, which keeps the answer on topic (retrieval-guided generation)
  * Blend         plate head + trigram mixed; weight tuned on held-out data

Honest expectation: this is a small word-level model, not a transformer. It learns the
STYLE and vocabulary of answers and stays roughly on topic, but it does not reason and
facts will often be wrong. The built-in evaluation measures this against simple baselines.

Usage
  python singularity_instruct.py                       auto-download alpaca_gpt4_data.json (43 MB), train, chat
  python singularity_instruct.py data.json [more.json] use your own file(s) (json or jsonl)
  python singularity_instruct.py --prompt "Give three tips for sleeping better." --len 80
  Flags: --fast  --retrain  --no-chat  --max-examples N  --plate-tokens N
Chat commands: /temp /top /len /mode blend|plate|trigram /rep /beta /show /quit
Accepted fields: instruction|prompt|question  +  input|context  ->  output|response|answer|completion
Dataset license note: alpaca-gpt4 is CC BY-NC 4.0 (research / non-commercial use).
Only load .pkl model files that you created yourself.
"""
import os
import re
import sys
import json
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAST = "--fast" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def flag(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


PROMPT = flag("--prompt")
GEN_LEN = int(flag("--len", 80))
MAX_EX = int(flag("--max-examples", 6000 if FAST else 0))
PLATE_TOK = int(flag("--plate-tokens", 30000 if FAST else 150000))
VALUE_FLAGS = ("--prompt", "--len", "--max-examples", "--plate-tokens")
ARGS = [a for i, a in enumerate(sys.argv) if i > 0 and not a.startswith("--")
        and sys.argv[i - 1] not in VALUE_FLAGS]

SPECIAL = ["<unk>", "<user>", "<bot>", "<end>", "<nl>"]
UNK, USER, BOT, END, NL = range(5)
NS = len(SPECIAL)
STOP_IDS = 40                       # ids below this = specials + most frequent words/punct
TOK = re.compile(r"<nl>|[a-z0-9']+|[.,;:!?]")
PUNCT = set(".,;:!?")
MODEL_PATH = os.path.join(HERE, "singularity_instruct_model.pkl")
LOAD = os.path.exists(MODEL_PATH) and "--retrain" not in sys.argv
URLS = ["https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data/resolve/main/data/alpaca_gpt4_data.json",
        "https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data.json"]


def tokenize(text):
    return TOK.findall(text.lower().replace("\n", " <nl> "))


def detok(tokens):
    out, cap = [], True
    for t in tokens:
        if t == "<nl>":
            out.append("\n")
            cap = True
        elif t in PUNCT:
            out.append(t)
            cap = t in ".!?"
        else:
            w = "I" if t == "i" else t
            if cap:
                w = w[:1].upper() + w[1:]
            cap = False
            out.append(w if (not out or out[-1].endswith("\n")) else " " + w)
    return "".join(out)


# ------------------------------------------------------------- dataset
def find_dataset():
    paths = [p for p in ARGS if os.path.exists(p)]
    if paths:
        return paths
    for name in ("alpaca_gpt4_data.json", "instruct.json", "instruct.jsonl"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            return [p]
    import urllib.request
    dest = os.path.join(HERE, "alpaca_gpt4_data.json")
    for url in URLS:
        try:
            log(f"Downloading dataset: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            log(f"Saved {os.path.getsize(dest) / 1e6:.1f} MB to {dest}")
            return [dest]
        except Exception as e:
            log(f"  failed ({e})")
    sys.exit("Could not download a dataset. Download alpaca_gpt4_data.json manually "
             "(Hugging Face: llm-wizard/alpaca-gpt4-data) and put it next to this script, "
             "or run: python singularity_instruct.py path\\to\\file.json")


def load_examples(paths):
    ex = []
    for p in paths:
        txt = open(p, encoding="utf-8", errors="ignore").read()
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            data = [json.loads(l) for l in txt.splitlines() if l.strip()]
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        for d in data:
            if not isinstance(d, dict):
                continue
            g = lambda *ks: next((str(d[k]) for k in ks if d.get(k)), "")
            ins = g("instruction", "prompt", "question", "query")
            inp = g("input", "context")
            out = g("output", "response", "answer", "completion")
            if ins and out:
                ex.append((ins + (" " + inp if inp else ""), out))
    return ex


def prepare_data():
    """Tokenise, build vocabulary, encode. Sets globals vocab, V, w2i."""
    global vocab, V, w2i
    from collections import Counter
    ex = load_examples(find_dataset())
    if len(ex) < 200:
        sys.exit(f"Only {len(ex)} usable examples found; need at least 200.")
    r = np.random.default_rng(0)
    order = r.permutation(len(ex))
    if MAX_EX:
        order = order[:MAX_EX]
    ex = [ex[i] for i in order]
    log(f"{len(ex):,} examples loaded")

    counts, kept = Counter(), []
    for i, (a, b) in enumerate(ex):
        ta, tb = tokenize(a), tokenize(b)
        if 2 <= len(ta) <= 100 and 3 <= len(tb) <= 250:     # keep answers un-truncated
            kept.append(i)
            counts.update(ta)
            counts.update(tb)
    nv = 2500 if FAST else 6000
    vocab = SPECIAL + [w for w, _ in counts.most_common(nv - NS) if w not in SPECIAL]
    V = len(vocab)
    w2i = {w: i for i, w in enumerate(vocab)}
    log(f"kept {len(kept):,} examples (length filter) | vocab {V}")

    ins_list, resp_list = [], []
    for i in kept:
        a, b = ex[i]
        ins_list.append(np.array([w2i.get(t, UNK) for t in tokenize(a)], np.int32))
        resp_list.append(np.array([w2i.get(t, UNK) for t in tokenize(b)], np.int32))
    del ex
    n_hold = min(600, max(40, len(ins_list) // 50))
    split = len(ins_list) - 2 * n_hold
    parts = lambda lo, hi: (ins_list[lo:hi], resp_list[lo:hi])
    return dict(train=parts(0, split), val=parts(split, split + n_hold),
                test=parts(split + n_hold, len(ins_list)))


def stream_of(ins, resp, limit=None):
    out, n = [], 0
    for a, b in zip(ins, resp):
        seq = np.concatenate([[USER], a, [BOT], b, [END]]).astype(np.int64)
        out.append(seq)
        n += len(seq)
        if limit and n >= limit:
            break
    return np.concatenate(out)


if LOAD:
    with open(MODEL_PATH, "rb") as _f:
        vocab = pickle.load(_f)                 # first object: vocabulary only
    V = len(vocab)
    w2i = {w: i for i, w in enumerate(vocab)}
    log(f"Found saved model ({V} words); loading")
    DATA = None
else:
    DATA = prepare_data()


# ---------------------------------------------------------- components
def softmax(z):
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


class NGram:
    """Word trigram with backoff, stored as sorted numpy key arrays (handles millions of words)."""

    def __init__(self, ids, k=8.0):
        self.k = k
        u = np.bincount(ids, minlength=V).astype(np.float64) + 0.5
        self.uni = u / u.sum()
        a, b = ids[:-1].astype(np.int64), ids[1:].astype(np.int64)
        self.k2, self.c2 = np.unique(a * V + b, return_counts=True)
        a2, b2, c3 = ids[:-2].astype(np.int64), ids[1:-1].astype(np.int64), ids[2:].astype(np.int64)
        self.k3, self.c3 = np.unique((a2 * V + b2) * V + c3, return_counts=True)

    def _slice(self, keys, counts, ctx):
        lo = np.searchsorted(keys, ctx * V)
        hi = np.searchsorted(keys, (ctx + 1) * V)
        return keys[lo:hi] - ctx * V, counts[lo:hi]

    def _mix(self, base, words, counts):
        tot = counts.sum()
        a = tot / (tot + self.k)
        q = (1 - a) * base
        q[words] += a * counts / tot
        return q

    def probs(self, w1, w2):
        q = self.uni
        w, c = self._slice(self.k2, self.c2, int(w2))
        if len(w):
            q = self._mix(q, w, c)
        w, c = self._slice(self.k3, self.c3, int(w1) * V + int(w2))
        if len(w):
            q = self._mix(q, w, c)
        return q

    def matrix(self, seq, wash):
        return np.stack([self.probs(seq[t - 1], seq[t]) for t in range(wash, len(seq) - 1)]).astype(np.float32)


class Plate:
    """Fixed noisy physical network with heterogeneous settling rates, word-coded acoustic drive."""

    def __init__(self, n, n_ch=64, noise=0.03, rho=0.97, in_scale=0.8, leak_lo=0.1, leak_hi=0.8, seed=0):
        r = np.random.default_rng(seed)
        W = r.normal(size=(n, n)) / np.sqrt(n)
        W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W.astype(np.float32)
        lk = np.exp(np.linspace(np.log(leak_lo), np.log(leak_hi), n))
        r.shuffle(lk)
        self.leak = lk.astype(np.float32)
        self.code = r.choice([-1.0, 1.0], size=(V, n_ch)).astype(np.float32)
        self.Win = (r.uniform(-1, 1, size=(n, n_ch)) * in_scale).astype(np.float32)
        self.n, self.noise = n, noise

    def step(self, x, w):
        return (1 - self.leak) * x + self.leak * np.tanh(self.W @ x + self.Win @ self.code[w])

    def feats(self, seq):
        x = np.zeros(self.n, np.float32)
        F = np.zeros((len(seq), self.n + self.code.shape[1]), np.float32)
        for t in range(len(seq)):
            x = self.step(x, seq[t])
            F[t, :self.n] = x + self.noise * 0.3 * rng.normal(size=self.n).astype(np.float32)
        F[:, self.n:] = self.code[seq]
        return F


class Head:
    def __init__(self, d, H):
        r = np.random.default_rng(1)
        self.P = {"W1": (r.normal(size=(d, H)) * np.sqrt(2 / d)).astype(np.float32),
                  "b1": np.zeros(H, np.float32),
                  "W2": (r.normal(size=(H, V)) * 0.02).astype(np.float32),
                  "b2": np.zeros(V, np.float32)}

    def logits(self, X, P=None):
        P = P or self.P
        return np.maximum(0, X @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]

    def probs(self, X, chunk=2000):
        return np.concatenate([softmax(self.logits(X[i:i + chunk])) for i in range(0, len(X), chunk)])

    def fit(self, Xtr, ytr, Xva, yva, epochs, bs=256, lr=1e-3, wd=2e-3, drop=0.5, patience=3):
        P = self.P
        M = {k: np.zeros_like(v) for k, v in P.items()}
        S = {k: np.zeros_like(v) for k, v in P.items()}
        best, bestP, bad, step = 1e9, None, 0, 0
        for ep in range(1, epochs + 1):
            idx = rng.permutation(len(Xtr))
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                xb = Xtr[b]
                pre = xb @ P["W1"] + P["b1"]
                h = np.maximum(0, pre)
                m = (rng.random(h.shape) > drop).astype(np.float32) / (1 - drop)
                hd = h * m
                g = softmax(hd @ P["W2"] + P["b2"])
                g[np.arange(len(b)), ytr[b]] -= 1.0
                g /= len(b)
                dh = (g @ P["W2"].T) * m * (pre > 0)
                G = {"W2": hd.T @ g, "b2": g.sum(0), "W1": xb.T @ dh, "b1": dh.sum(0)}
                step += 1
                for k in P:
                    gk = G[k] + (wd * P[k] if k[0] == "W" else 0)
                    M[k] = 0.9 * M[k] + 0.1 * gk
                    S[k] = 0.999 * S[k] + 0.001 * gk ** 2
                    P[k] -= lr * (M[k] / (1 - 0.9 ** step)) / (np.sqrt(S[k] / (1 - 0.999 ** step)) + 1e-8)
            ppl = ppl_of(self.probs(Xva), yva)
            log(f"   epoch {ep:2d}  val perplexity {ppl:.1f}")
            if ppl < best - 0.2:
                best, bestP, bad = ppl, {k: v.copy() for k, v in P.items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        self.P = bestP


def ppl_of(P, y):
    return float(np.exp(-np.log(P[np.arange(len(y)), y] + 1e-12).mean()))


class Retriever:
    """Inverted index over training instructions -> similar examples -> response word prior."""

    def __init__(self, ins_list, resp_list):
        N = len(ins_list)
        lens = np.array([len(a) for a in ins_list])
        flat = np.concatenate(ins_list).astype(np.int64)
        ex = np.repeat(np.arange(N), lens)
        pairs = np.unique(flat * N + ex)
        w, self.ex = pairs // N, (pairs % N).astype(np.int32)
        self.starts = np.searchsorted(w, np.arange(V + 1))
        df = np.diff(self.starts).astype(np.float64)
        self.idf = np.log((N + 1) / (df + 1))
        self.ilen, self.N, self.ins, self.resp = lens, N, ins_list, resp_list

    def top(self, qids, k=8):
        q = [w for w in set(int(t) for t in qids) if w >= NS and self.idf[w] > 1.5]
        if not q:
            return []
        sc = np.zeros(self.N)
        for w in q:
            sc[self.ex[self.starts[w]:self.starts[w + 1]]] += self.idf[w]
        sc /= np.sqrt(self.ilen + 4)
        k = min(k, self.N)
        idx = np.argpartition(-sc, k - 1)[:k]
        idx = idx[np.argsort(-sc[idx])]
        return [int(i) for i in idx if sc[i] > 0]

    def boost(self, idxs):
        if not idxs:
            return np.zeros(V)
        c = np.bincount(np.concatenate([self.resp[i] for i in idxs]), minlength=V).astype(np.float64)
        b = c * np.maximum(self.idf, 0)
        b[:STOP_IDS] = 0.0
        return b / (b.max() + 1e-9)


# ----------------------------------------------------------- generation
def generate(S, prompt_ids, n, temp, top_p, mode, rep, beta, boost, min_len=6):
    plate, head, mu, sd, ng, a = S["plate"], S["head"], S["mu"], S["sd"], S["ng"], S["a"]
    hist = [END, USER] + [int(t) for t in prompt_ids] + [BOT]
    x = np.zeros(plate.n, np.float32)
    for w in hist:
        x = plate.step(x, w)
    out = []
    punct_ids = {w2i[t] for t in PUNCT if t in w2i}
    for _ in range(n):
        pt = ng.probs(hist[-2], hist[-1])
        if mode == "trigram":
            p = pt.copy()
        else:
            xn = x + plate.noise * 0.3 * rng.normal(size=plate.n).astype(np.float32)
            f = (np.concatenate([xn, plate.code[hist[-1]]]) - mu) / sd
            pn = softmax(head.logits(f[None, :]))[0]
            p = pn.astype(np.float64) if mode == "plate" else a * pn + (1 - a) * pt
        p[UNK] = p[USER] = p[BOT] = 0.0
        if len(out) < min_len:
            p[END] = 0.0
        if hist[-1] in punct_ids:
            for w in punct_ids:
                p[w] = 0.0
        if beta > 0 and boost is not None:
            p = p * (1.0 + beta * boost)
        if rep > 1.0:
            for w in set(hist[-30:]):
                if w >= STOP_IDS:
                    p[w] /= rep ** hist[-30:].count(w)
        p = p ** (1.0 / temp)
        p /= p.sum()
        if top_p < 1.0:
            o = np.argsort(-p)
            keep = o[:int(np.searchsorted(np.cumsum(p[o]), top_p)) + 1]
            q = np.zeros_like(p)
            q[keep] = p[keep]
            p = q / q.sum()
        w = int(rng.choice(V, p=p))
        if w == END:
            break
        out.append(w)
        hist.append(w)
        x = plate.step(x, w)
    return out


def answer(S, text, n, temp, top_p, mode, rep, beta):
    toks = tokenize(text)
    ids_ = [w2i.get(t, UNK) for t in toks]
    unknown = sorted({t for t, i in zip(toks, ids_) if i == UNK})
    near = S["retr"].top(ids_)
    boost = S["retr"].boost(near)
    gen = generate(S, ids_, n, temp, top_p, mode, rep, beta, boost)
    return detok([vocab[w] for w in gen]), unknown, near


def f1_content(gen, ref):
    g = {w for w in gen if w >= STOP_IDS}
    r = {w for w in ref if w >= STOP_IDS}
    if not g or not r:
        return 0.0
    hit = len(g & r)
    if hit == 0:
        return 0.0
    p, rc = hit / len(g), hit / len(r)
    return 2 * p * rc / (p + rc)


# ----------------------------------------------------------------- train
def train_model():
    wash = 50
    tr_ins, tr_resp = DATA["train"]
    log(f"Building trigram over all {len(tr_ins):,} training examples")
    full = stream_of(tr_ins, tr_resp)
    log(f"   {len(full):,} tokens")
    retr = Retriever(tr_ins, tr_resp)

    vs = stream_of(*DATA["val"], limit=2500 if FAST else 4000)
    ts = stream_of(*DATA["test"], limit=2500 if FAST else 4000)
    best_k, best_p = 8.0, 1e9
    for k in (4.0, 8.0, 16.0):
        ng = NGram(full, k)
        pv = ppl_of(ng.matrix(vs, wash), vs[1:][wash:])
        log(f"   trigram smoothing k={k}: val perplexity {pv:.1f}")
        if pv < best_p:
            best_k, best_p = k, pv
    ng = NGram(full, best_k)

    n_nodes = 256 if FAST else 768
    log(f"Plate: {n_nodes} electrodes; driving over ~{PLATE_TOK:,} training tokens")
    plate = Plate(n_nodes, seed=3)
    order = np.random.default_rng(1).permutation(len(tr_ins))
    sub = stream_of([tr_ins[i] for i in order], [tr_resp[i] for i in order], limit=PLATE_TOK)
    Ftr, ytr = plate.feats(sub[:-1])[wash:], sub[1:][wash:]
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr -= mu
    Ftr /= sd
    Fva = (plate.feats(vs[:-1])[wash:] - mu) / sd
    Fte = (plate.feats(ts[:-1])[wash:] - mu) / sd
    yva, yte = vs[1:][wash:], ts[1:][wash:]

    log("Training digital head")
    head = Head(Ftr.shape[1], 192 if FAST else 384)
    head.fit(Ftr, ytr, Fva, yva, epochs=4 if FAST else 10)
    del Ftr

    Pn_va, Pn_te = head.probs(Fva), head.probs(Fte)
    Pt_va, Pt_te = ng.matrix(vs, wash), ng.matrix(ts, wash)
    best_a, best_b = 0.5, 1e9
    for a in np.linspace(0, 1, 11):
        v = ppl_of(a * Pn_va + (1 - a) * Pt_va, yva)
        if v < best_b:
            best_a, best_b = float(a), v
    ppl = {"trigram": ppl_of(Pt_te, yte), "plate+head": ppl_of(Pn_te, yte),
           f"blend (a={best_a:.1f})": ppl_of(best_a * Pn_te + (1 - best_a) * Pt_te, yte)}
    log("Held-out perplexity (all tokens):")
    for k, v in ppl.items():
        log(f"   {k:<18} {v:8.1f}")
    del Pn_va, Pn_te, Pt_va, Pt_te

    S = dict(plate=plate, head=head, mu=mu, sd=sd, ng=ng, a=best_a, retr=retr,
             temp=0.7, top_p=0.9, rep=1.3, beta=4.0)

    log("Instruction-following check on held-out prompts (content-word F1 vs reference answer)")
    te_ins, te_resp = DATA["test"]
    nev = 25 if FAST else 60
    gen_len = 45
    res = {}
    configs = [("trigram only", "trigram", 0.0), ("blend", "blend", 0.0),
               ("blend + retrieval b=3", "blend", 3.0), ("blend + retrieval b=8", "blend", 8.0)]
    for name, mode, beta in configs:
        sc = []
        for i in range(nev):
            near = retr.top(te_ins[i])
            bst = retr.boost(near)
            g = generate(S, te_ins[i], gen_len, S["temp"], S["top_p"], mode, S["rep"], beta, bst)
            sc.append(f1_content(g, te_resp[i]))
        res[name] = float(np.mean(sc))
        log(f"   {name:<24} F1 {res[name]:.3f}")
    nn, rnd = [], []
    for i in range(nev):
        near = retr.top(te_ins[i], k=1)
        nn.append(f1_content(retr.resp[near[0]][:gen_len], te_resp[i]) if near else 0.0)
        rnd.append(f1_content(retr.resp[int(rng.integers(retr.N))][:gen_len], te_resp[i]))
    res["nearest-neighbour copy"], res["random answer"] = float(np.mean(nn)), float(np.mean(rnd))
    log(f"   {'nearest-neighbour copy':<24} F1 {res['nearest-neighbour copy']:.3f}")
    log(f"   {'random answer':<24} F1 {res['random answer']:.3f}")
    gen_models = {k: v for k, v in res.items() if "retrieval" in k}
    S["beta"] = 3.0 if res.get("blend + retrieval b=3", 0) >= res.get("blend + retrieval b=8", 0) else 8.0
    log(f"   chosen retrieval strength beta = {S['beta']}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(vocab, f)               # vocab first so it loads before the classes exist
        pickle.dump(S, f)
    log(f"Saved model to {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1e6:.0f} MB)")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].barh(list(ppl), list(ppl.values()), color=["#888", "#2a7", "#37c"])
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Held-out perplexity (lower is better)")
    ax[0].set_title("Language modelling on instruction data")
    ax[1].barh(list(res), list(res.values()), color="#c63")
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Content-word F1 vs reference answer (higher is better)")
    ax[1].set_title("Instruction-following proxy (held-out prompts)")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "singularity_instruct.png"), dpi=140)
    return S


# ----------------------------------------------------------------- chat
def chat(S):
    st = dict(temp=S["temp"], top_p=S["top_p"], mode="blend", n=GEN_LEN, rep=S["rep"], beta=S["beta"])
    show = False
    print("\nAsk or instruct. Commands: /temp /top /len /mode /rep /beta /show /quit")
    print(f"Settings: {st}")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            try:
                if cmd in ("quit", "exit", "q"):
                    break
                elif cmd == "show":
                    show = not show
                    print(f"show nearest training instructions: {show}")
                    continue
                elif cmd in ("temp", "top", "rep", "beta"):
                    st[{"top": "top_p"}.get(cmd, cmd)] = float(arg)
                elif cmd == "len":
                    st["n"] = int(arg)
                elif cmd == "mode" and arg in ("blend", "plate", "trigram"):
                    st["mode"] = arg
                else:
                    print("Unknown command or bad value.")
                    continue
                print(f"Settings: {st}")
            except ValueError:
                print("Bad value.")
            continue
        out, unk, near = answer(S, line, st["n"], st["temp"], st["top_p"], st["mode"], st["rep"], st["beta"])
        if unk:
            print(f"(not in vocabulary, treated as unknown: {', '.join(unk)})")
        if show:
            for i in near[:3]:
                print("  ~ similar training instruction:", detok([vocab[w] for w in S["retr"].ins[i][:14]]))
        print("\nbot> " + out)


def main():
    if LOAD:
        with open(MODEL_PATH, "rb") as f:
            pickle.load(f)
            S = pickle.load(f)
    else:
        S = train_model()
    if PROMPT is not None:
        out, unk, _ = answer(S, PROMPT, GEN_LEN, S["temp"], S["top_p"], "blend", S["rep"], S["beta"])
        if unk:
            print(f"(not in vocabulary, treated as unknown: {', '.join(unk)})")
        print(out)
    elif "--no-chat" not in sys.argv:
        chat(S)


if __name__ == "__main__":
    main()
