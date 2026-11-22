"""
GPT-2 x SUBSET-SUM DECODER

Uses MatchSubsetSumSolver (subset_solver.py, same folder) on GPT-2's next-token
distribution.

  numbers  = GPT-2's next-token probabilities, scaled to integers (index = token id)
  target   = a probability mass T (e.g. 0.90) scaled the same way
  solver   = finds a SET OF TOKEN IDS whose probabilities sum to T (error-matching search)
  decode   = the chosen token ids -> token strings (the "token state" for this step)
  sample   = next token drawn from that subset (renormalised), appended, repeat

So instead of top-k / top-p (always a prefix of the ranked list), the allowed token set is
whatever subset the solver lands on. The report shows what that did to the probabilities:
which head tokens were kept or dropped, how much mass came from the tail, the solver error,
and how the generated text differs from greedy and from plain sampling.

Setup:   pip install torch transformers numpy
Run:     python gpt2_subset_decode.py --prompt "The meaning of life is" --mass 0.9 --steps 12
Options: --engine "Swap search" | "Anneal" | "Random queue"
         --scale-bits 52   (integer precision of the probabilities)
         --plot out.png    (needs matplotlib)
         --mock            (synthetic distribution, no GPT-2 download; pipeline test only)
"""

import argparse
import math
import random
import sys
import types

# subset_solver imports tkinter for its GUI; this script only needs the solver class.
try:
    import tkinter  # noqa: F401
except ImportError:
    for name in ("tkinter", "tkinter.filedialog", "tkinter.messagebox", "tkinter.ttk"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["tkinter"].filedialog = sys.modules["tkinter.filedialog"]
    sys.modules["tkinter"].messagebox = sys.modules["tkinter.messagebox"]
    sys.modules["tkinter"].ttk = sys.modules["tkinter.ttk"]
    sys.modules["tkinter"].Tk = object

import numpy as np

from subset_solver import MatchSubsetSumSolver


# ---------------------------------------------------------------- model backends
class GPT2Backend:
    def __init__(self, name: str = "gpt2"):
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.torch = torch
        self.tok = GPT2TokenizerFast.from_pretrained(name)
        self.model = GPT2LMHeadModel.from_pretrained(name).eval()
        self.vocab = self.model.config.vocab_size

    def encode(self, text):
        return self.tok.encode(text)

    def probs(self, ids):
        with self.torch.no_grad():
            logits = self.model(self.torch.tensor([ids])).logits[0, -1].double()
        return self.torch.softmax(logits, dim=-1).numpy()

    def decode(self, tid):
        return self.tok.decode([int(tid)])


class MockBackend:
    """Synthetic Zipf-like distribution that depends on the context. Pipeline test only."""

    vocab = 50257

    def encode(self, text):
        return [sum(map(ord, w)) % self.vocab for w in text.split()] or [0]

    def probs(self, ids):
        rng = np.random.default_rng(sum(ids) * 7919 + len(ids))
        ranks = np.arange(1, self.vocab + 1, dtype=np.float64)
        p = ranks ** -1.1
        p = p[rng.permutation(self.vocab)]
        p *= rng.lognormal(0, 0.4, self.vocab)
        return p / p.sum()

    def decode(self, tid):
        return f"<tok{int(tid)}>"


# ---------------------------------------------------------------- core step
def subset_step(probs, mass, scale_bits, engine, max_steps, seed):
    """Run the solver on one distribution. Returns dict with the chosen token ids."""
    scale = 1 << scale_bits
    values = [max(1, int(p * scale)) for p in probs]      # solver needs positive ints
    total = sum(values)
    target = min(total, max(1, round(mass * total)))
    solver = MatchSubsetSumSolver(
        0, 0, "Planted subset", engine, max_steps, seed, values=values, target=target
    )
    res = solver.run()
    chosen = [i for i, s in enumerate(res["assignment"]) if s]
    got = sum(values[i] for i in chosen)
    return {
        "chosen": chosen,
        "values": values,
        "total": total,
        "target": target,
        "got": got,
        "res": res,
        "got_mass": got / total,
    }


def sample_from(ids, weights, rng):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return int(ids[rng.choice(len(ids), p=w)])


# ---------------------------------------------------------------- reporting
def fmt_tok(s):
    return repr(s)[1:-1] if s.strip() else repr(s)[1:-1] or "∅"


def run(args):
    backend = MockBackend() if args.mock else GPT2Backend(args.model)
    ids = backend.encode(args.prompt)
    rng = np.random.default_rng(args.seed)

    # three parallel continuations: subset-sum, greedy, plain sampling
    ctx_sub = list(ids)
    ctx_greedy = list(ids)
    ctx_plain = list(ids)
    plain_rng = np.random.default_rng(args.seed + 1)

    print("=" * 78)
    print(f"GPT-2 subset-sum decoding | prompt: {args.prompt!r}")
    print(f"target mass {args.mass:.4f} | engine {args.engine} | scale 2^{args.scale_bits}"
          f"{' | MOCK DISTRIBUTION (not GPT-2)' if args.mock else ''}")
    print("=" * 78)

    log = []
    for step in range(1, args.steps + 1):
        probs = backend.probs(ctx_sub)
        order = np.argsort(-probs)
        rank_of = np.empty_like(order)
        rank_of[order] = np.arange(len(order))
        entropy = float(-(probs * np.log2(np.clip(probs, 1e-30, 1))).sum())

        out = subset_step(probs, args.mass, args.scale_bits, args.engine,
                          args.solver_steps, args.seed + step)
        chosen = out["chosen"]
        chosen_set = set(chosen)
        res = out["res"]

        # sample next token from the subset
        sub_w = [probs[i] for i in chosen]
        nxt = sample_from(chosen, sub_w, rng)
        sub_mass = float(sum(sub_w))

        head_n = 20
        head = order[:head_n]
        kept_head = [int(t) for t in head if int(t) in chosen_set]
        dropped_head = [int(t) for t in head if int(t) not in chosen_set]
        tail_ids = [i for i in chosen if rank_of[i] >= 100]
        tail_mass = float(sum(probs[i] for i in tail_ids))
        # how well subset resembles a plain top-p nucleus
        cum = np.cumsum(probs[order])
        nucleus_k = int(np.searchsorted(cum, args.mass) + 1)
        overlap = len(chosen_set & set(int(t) for t in order[:nucleus_k]))

        print(f"\n--- step {step} | context tail: {''.join(backend.decode(t) for t in ctx_sub[-6:])!r}")
        print(f"  full distribution: entropy {entropy:.2f} bits | top-5: " +
              ", ".join(f"{fmt_tok(backend.decode(t))}({probs[t]:.3f})" for t in order[:5]))
        print(f"  solver: target {args.mass:.4f} -> subset mass {out['got_mass']:.6f} "
              f"| |err| = {res['abs_error']:,} units ({res['abs_error'] / out['total']:.2e} of mass) "
              f"| {'EXACT' if res['solved'] else 'closest'} | {res['steps']} moves, {res['elapsed']:.2f}s")
        print(f"  subset: {len(chosen):,} of {len(probs):,} tokens "
              f"(plain top-p would use {nucleus_k:,}; overlap {overlap:,})")
        print(f"  head (top-{head_n}) kept {len(kept_head)}/{head_n}"
              + (f" | DROPPED: " + ", ".join(
                  f"{fmt_tok(backend.decode(t))}[#{rank_of[t] + 1}, {probs[t]:.3f}]" for t in dropped_head)
                 if dropped_head else ""))
        print(f"  tail (rank>=100) tokens in subset: {len(tail_ids):,} carrying {tail_mass:.4f} of mass")

        if step == 1:
            print("  decoded subset (first step), highest-prob first:")
            top_chosen = sorted(chosen, key=lambda i: -probs[i])[:args.show]
            for i in top_chosen:
                print(f"     id {i:>6} #{rank_of[i] + 1:<6} p={probs[i]:.5f}  {fmt_tok(backend.decode(i))!r}")
            if len(chosen) > args.show:
                print(f"     ... plus {len(chosen) - args.show:,} more lower-probability tokens")

        s_rank = int(rank_of[nxt]) + 1
        g = int(order[0])
        print(f"  sampled from subset: {fmt_tok(backend.decode(nxt))!r} "
              f"(id {nxt}, full-dist rank #{s_rank}, p={probs[nxt]:.4f}, "
              f"renormalised p={probs[nxt] / sub_mass:.4f}) | greedy would pick {fmt_tok(backend.decode(g))!r}")

        log.append({
            "step": step, "entropy": entropy, "subset_size": len(chosen), "nucleus_k": nucleus_k,
            "got_mass": out["got_mass"], "abs_err": res["abs_error"] / out["total"],
            "kept_head": len(kept_head), "head_n": head_n, "tail_mass": tail_mass,
            "sampled_rank": s_rank, "sampled_p": float(probs[nxt]),
            "top_prob": float(probs[g]),
        })

        ctx_sub.append(nxt)

        # baselines (their own contexts)
        pg = backend.probs(ctx_greedy)
        ctx_greedy.append(int(np.argmax(pg)))
        pp = backend.probs(ctx_plain)
        ctx_plain.append(sample_from(np.arange(len(pp)), pp, plain_rng))

    show = lambda c: "".join(backend.decode(t) for t in c)
    print("\n" + "=" * 78)
    print("OUTCOMES")
    print("=" * 78)
    print(f"subset-sum : {show(ctx_sub)!r}")
    print(f"greedy     : {show(ctx_greedy)!r}")
    print(f"plain samp.: {show(ctx_plain)!r}")
    print("\nsummary over steps:")
    print(f"  mean subset size {np.mean([l['subset_size'] for l in log]):,.0f} "
          f"vs plain top-p {np.mean([l['nucleus_k'] for l in log]):,.0f}")
    print(f"  mean |mass error| {np.mean([l['abs_err'] for l in log]):.2e}")
    print(f"  mean head-20 kept {np.mean([l['kept_head'] for l in log]):.1f}/20")
    print(f"  mean tail mass in subset {np.mean([l['tail_mass'] for l in log]):.4f}")
    print(f"  mean sampled rank {np.mean([l['sampled_rank'] for l in log]):.1f} "
          f"(1 = greedy choice); picked greedy token {sum(l['sampled_rank'] == 1 for l in log)}/{len(log)} steps")

    if args.plot:
        plot(log, args)


def plot(log, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [l["step"] for l in log]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(x, [l["entropy"] for l in log], marker="o")
    ax[0, 0].set(title="Entropy of next-token distribution (bits)", xlabel="step")
    ax[0, 1].plot(x, [l["subset_size"] for l in log], marker="o", label="subset-sum")
    ax[0, 1].plot(x, [l["nucleus_k"] for l in log], marker="s", label="plain top-p")
    ax[0, 1].set(title="Tokens allowed", xlabel="step", yscale="log")
    ax[0, 1].legend()
    ax[1, 0].bar(x, [l["tail_mass"] for l in log])
    ax[1, 0].set(title="Mass from tail tokens (rank>=100) in subset", xlabel="step")
    ax[1, 1].bar(x, [l["sampled_rank"] for l in log])
    ax[1, 1].set(title="Full-distribution rank of sampled token", xlabel="step", yscale="log")
    fig.suptitle(f"Subset-sum decoding, target mass {args.mass}" + (" (MOCK)" if args.mock else ""))
    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    print(f"\nplot saved to {args.plot}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="The meaning of life is")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--mass", type=float, default=0.90, help="target probability mass in (0,1)")
    ap.add_argument("--steps", type=int, default=120, help="tokens to generate")
    ap.add_argument("--engine", default="Swap search", choices=list(MatchSubsetSumSolver.ENGINES))
    ap.add_argument("--scale-bits", type=int, default=52)
    ap.add_argument("--solver-steps", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show", type=int, default=25, help="decoded subset tokens to list at step 1")
    ap.add_argument("--plot", default="")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    if not (0 < a.mass < 1):
        sys.exit("--mass must be between 0 and 1")
    run(a)
