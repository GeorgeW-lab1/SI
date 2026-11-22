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

instruction MODE (--instruction "text"): the target IS the instruction (or a sketch of what it looks like).
  1. FETCH      teacher-force GPT-2 along prompt + instruction; per step fetch the top-K candidate pool,
                the instruction token's probability/rank, and the distribution's entropy / top-1 prob.
  2. REGRESS    the smallest target mass at which the instruction token becomes reachable is
                logit(needed) ~ a + b*entropy + c*log(top1)   (least squares, margin so training steps are covered).
  3. SOLVE      at each step the regressed mass is the solver's target; the subset sum picks the token set.
  4. DECODE     subset ids -> token strings; check whether the instruction token is inside, then run free
                (no teacher) from the prompt with the regressed targets and compare with the instruction text.

Setup:   pip install torch transformers numpy
Run:     python gpt2_subset_decode.py --prompt "The meaning of life is" --masses 0.3,0.6,0.9,0.99 --steps 12
         (each target mass is run through the solver and the text it produces is shown, then
          all targets are compared in one TARGET -> TEXT table alongside greedy)
Options: --engine "Swap search" | "Anneal" | "Random queue"
         --scale-bits 52   (integer precision of the probabilities)
         --plot out.png    (needs matplotlib)
         --mock            (synthetic distribution, no GPT-2 download; pipeline test only)
"""

import argparse
import math
import time
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
        self.eos = self.tok.eos_token_id
        self.window = self.model.config.n_positions   # 1024

    def encode(self, text):
        return self.tok.encode(text)

    def probs(self, ids):
        with self.torch.no_grad():
            ids = ids[-self.window:]                        # sliding window past 1024 tokens
            logits = self.model(self.torch.tensor([ids])).logits[0, -1].double()
        return self.torch.softmax(logits, dim=-1).numpy()

    def decode(self, tid):
        return self.tok.decode([int(tid)])


class MockBackend:
    """Synthetic Zipf-like distribution that depends on the context. Pipeline test only."""

    vocab = 50257
    eos = None
    plant = None   # (prompt_ids, instruction_ids): boosts the instruction token so it has a realistic rank

    def encode(self, text):
        return [sum(map(ord, w)) % self.vocab for w in text.split()] or [0]

    def probs(self, ids):
        rng = np.random.default_rng(sum(ids) * 7919 + len(ids))
        ranks = np.arange(1, self.vocab + 1, dtype=np.float64)
        p = ranks ** -1.1
        p = p[rng.permutation(self.vocab)]
        p *= rng.lognormal(0, 0.4, self.vocab)
        p = p / p.sum()
        if self.plant is not None:
            pr, sol = self.plant
            i = len(ids) - len(pr)
            if 0 <= i < len(sol) and list(ids) == list(pr) + list(sol[:i]):
                target_rank = int(rng.geometric(0.02))            # instruction token lands at a plausible rank
                srt = np.sort(p)[::-1]
                p[sol[i]] = srt[min(target_rank, self.vocab - 1)]
                p = p / p.sum()
        return p

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


def generate(backend, args, mass, verbose=True):
    """Generate args.steps tokens with the allowed set chosen by the solver at this target mass."""
    ids = backend.encode(args.prompt)
    rng = np.random.default_rng(args.seed)
    ctx = list(ids)
    log = []
    first_subset = None

    for step in range(1, args.steps + 1):
        probs = backend.probs(ctx)
        order = np.argsort(-probs)
        rank_of = np.empty_like(order)
        rank_of[order] = np.arange(len(order))
        entropy = float(-(probs * np.log2(np.clip(probs, 1e-30, 1))).sum())

        out = subset_step(probs, mass, args.scale_bits, args.engine, args.solver_steps, args.seed + step)
        chosen = out["chosen"]
        chosen_set = set(chosen)
        res = out["res"]

        sub_w = [probs[i] for i in chosen]
        nxt = sample_from(chosen, sub_w, rng)
        sub_mass = float(sum(sub_w))

        head_n = 20
        head = order[:head_n]
        kept_head = [int(t) for t in head if int(t) in chosen_set]
        dropped_head = [int(t) for t in head if int(t) not in chosen_set]
        tail_ids = [i for i in chosen if rank_of[i] >= 100]
        tail_mass = float(sum(probs[i] for i in tail_ids))
        cum = np.cumsum(probs[order])
        nucleus_k = int(np.searchsorted(cum, mass) + 1)
        overlap = len(chosen_set & set(int(t) for t in order[:nucleus_k]))
        s_rank = int(rank_of[nxt]) + 1
        g = int(order[0])

        if step == 1:
            first_subset = sorted(chosen, key=lambda i: -probs[i])[:args.show]
            first_info = [(i, int(rank_of[i]) + 1, float(probs[i])) for i in first_subset]

        if verbose:
            print(f"\n  step {step} | text so far: {''.join(backend.decode(t) for t in ctx[len(ids):])!r}")
            print(f"    full dist: entropy {entropy:.2f} bits | top-5: " +
                  ", ".join(f"{fmt_tok(backend.decode(t))}({probs[t]:.3f})" for t in order[:5]))
            print(f"    solver: target {mass:.4f} -> subset mass {out['got_mass']:.6f} "
                  f"| err {res['abs_error'] / out['total']:.2e} of mass "
                  f"| {'EXACT' if res['solved'] else 'closest'} | {res['steps']} moves, {res['elapsed']:.2f}s")
            print(f"    subset {len(chosen):,}/{len(probs):,} tokens (top-p: {nucleus_k:,}, overlap {overlap:,}) "
                  f"| head-20 kept {len(kept_head)}/{head_n}"
                  + (" | DROPPED: " + ", ".join(
                      f"{fmt_tok(backend.decode(t))}[#{rank_of[t] + 1}]" for t in dropped_head[:6])
                     if dropped_head else "")
                  + f" | tail mass {tail_mass:.4f}")
            print(f"    -> sampled {fmt_tok(backend.decode(nxt))!r} (rank #{s_rank}, p={probs[nxt]:.4f}) "
                  f"| greedy would pick {fmt_tok(backend.decode(g))!r}")

        log.append({
            "step": step, "entropy": entropy, "subset_size": len(chosen), "nucleus_k": nucleus_k,
            "got_mass": out["got_mass"], "abs_err": res["abs_error"] / out["total"],
            "kept_head": len(kept_head), "head_n": head_n, "tail_mass": tail_mass,
            "sampled_rank": s_rank, "sampled_p": float(probs[nxt]), "top_prob": float(probs[g]),
            "token": backend.decode(nxt),
        })
        ctx.append(nxt)

    text = "".join(backend.decode(t) for t in ctx[len(ids):])
    return {"mass": mass, "text": text, "log": log, "first_info": first_info}


def greedy_text(backend, args):
    ids = backend.encode(args.prompt)
    ctx = list(ids)
    for _ in range(args.steps):
        ctx.append(int(np.argmax(backend.probs(ctx))))
    return "".join(backend.decode(t) for t in ctx[len(ids):])


def run(args):
    backend = MockBackend() if args.mock else GPT2Backend(args.model)
    if args.instruction:
        return run_instruction(backend, args)
    masses = [float(m) for m in args.masses.split(",")] if args.masses else [args.mass]

    print("=" * 78)
    print(f"GPT-2 subset-sum decoding | prompt: {args.prompt!r} | engine {args.engine}"
          f"{' | MOCK DISTRIBUTION (not GPT-2)' if args.mock else ''}")
    print(f"targets (probability mass): {', '.join(f'{m:g}' for m in masses)}")
    print("=" * 78)

    results = []
    for m in masses:
        print(f"\n##### TARGET {m:g} #####")
        r = generate(backend, args, m, verbose=not args.quiet)
        print(f"\n  decoded subset at step 1 (target {m:g}), highest-prob first:")
        for i, rk, p in r["first_info"]:
            print(f"     id {i:>6} #{rk:<6} p={p:.5f}  {fmt_tok(backend.decode(i))!r}")
        print(f"\n  TEXT for target {m:g}: {args.prompt}{r['text']!r}")
        results.append(r)

    g = greedy_text(backend, args)

    print("\n" + "=" * 78)
    print("TARGET -> TEXT")
    print("=" * 78)
    print(f"{'target':>8} | {'avg subset':>10} | {'tail mass':>9} | {'avg rank':>8} | text")
    for r in results:
        L = r["log"]
        print(f"{r['mass']:>8g} | {np.mean([l['subset_size'] for l in L]):>10,.0f} | "
              f"{np.mean([l['tail_mass'] for l in L]):>9.4f} | "
              f"{np.mean([l['sampled_rank'] for l in L]):>8.1f} | {args.prompt}{r['text']!r}")
    print(f"{'greedy':>8} | {'1':>10} | {'-':>9} | {'1.0':>8} | {args.prompt}{g!r}")
    if args.plot:
        plot(results[-1]["log"], args)


# ---------------------------------------------------------------- instruction mode
def pool_step(probs, pool_ids, frac, scale_bits, engine, max_steps, seed):
    """Solver on the top-K pool. frac = target as a fraction of the pool's total mass."""
    scale = 1 << scale_bits
    values = [max(1, int(probs[i] * scale)) for i in pool_ids]
    total = sum(values)
    target = min(total, max(1, round(frac * total)))
    solver = MatchSubsetSumSolver(0, 0, "Planted subset", engine, max_steps, seed, values=values, target=target)
    res = solver.run()
    chosen = [int(pool_ids[j]) for j, sel in enumerate(res["assignment"]) if sel]
    return {"chosen": chosen, "res": res, "total": total, "target": target,
            "got_frac": sum(values[j] for j, sel in enumerate(res["assignment"]) if sel) / total}


def feats(probs):
    ent = float(-(probs * np.log2(np.clip(probs, 1e-30, 1))).sum())
    return ent, float(math.log(max(probs.max(), 1e-12)))


def logit(x):
    x = min(max(x, 1e-4), 1 - 1e-4)
    return math.log(x / (1 - x))


def run_instruction(backend, args):
    K = args.pool
    prompt_ids = backend.encode(args.prompt)
    sol_ids = backend.encode(" " + args.instruction.strip()) if not args.mock else backend.encode(args.instruction)
    if args.mock:
        backend.plant = (prompt_ids, sol_ids)
    dec = lambda ids: "".join(backend.decode(t) for t in ids)

    print("=" * 78)
    print(f"instruction MODE | prompt: {args.prompt!r}")
    print(f"instruction (target): {args.instruction!r} -> {len(sol_ids)} tokens: "
          + " ".join(repr(backend.decode(t)) for t in sol_ids)
          + (" | MOCK DISTRIBUTION (not GPT-2)" if args.mock else ""))
    print("=" * 78)

    # 1. FETCH
    rows, ctx = [], list(prompt_ids)
    for i, t in enumerate(sol_ids):
        probs = backend.probs(ctx)
        order = np.argsort(-probs)
        rank = int(np.where(order == t)[0][0])
        pool = order[:K]
        pool_total = float(probs[pool].sum())
        needed = float(np.cumsum(probs[order])[rank] / pool_total) if rank < K else None
        ent, ltop = feats(probs)
        rows.append({"i": i, "tok": t, "rank": rank + 1, "p": float(probs[t]), "ent": ent,
                     "ltop": ltop, "needed": needed, "pool": pool, "probs": probs})
        ctx.append(t)
    inpool = [r for r in rows if r["needed"] is not None]
    print(f"\n[1] FETCH: {len(inpool)}/{len(rows)} instruction tokens lie inside the top-{K} pool")

    # 2. REGRESS
    if len(inpool) >= 4:
        X = np.array([[1.0, r["ent"], r["ltop"]] for r in inpool])
        y = np.array([logit(r["needed"]) for r in inpool])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        r2 = 1 - float(resid @ resid) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)
        margin = max(0.0, float(resid.max())) + args.margin
        coef = coef.copy()
        coef[0] += margin
        print(f"[2] REGRESS: logit(needed mass) = {coef[0]:+.3f} {coef[1]:+.3f}*entropy {coef[2]:+.3f}*log(top1)"
              f"   (R^2 {r2:.2f}, safety margin {margin:.2f})")
    else:
        mx = max([r["needed"] for r in inpool], default=0.99)
        coef = np.array([logit(min(mx + 0.01, 0.999)), 0.0, 0.0])
        print(f"[2] REGRESS: too few in-pool points for a fit; using constant target {mx:.3f}")
    predict = lambda ent, ltop: min(0.999, 1 / (1 + math.exp(-(coef[0] + coef[1] * ent + coef[2] * ltop))))

    # 3+4a. SOLVE + DECODE along the instruction (teacher-forced)
    print("\n[3] SOLVE + DECODE along the instruction (teacher-forced):")
    print(f"  {'step':>4} {'instruction tok':<16} {'rank':>6} {'p':>8} {'needed':>7} {'target':>7} "
          f"{'subset':>7} {'in subset?':>10}")
    hits = 0
    for r in rows:
        tgt = predict(r["ent"], r["ltop"])
        out = pool_step(r["probs"], r["pool"], tgt, args.scale_bits, args.engine, args.solver_steps, args.seed + r["i"])
        hit = r["tok"] in set(out["chosen"])
        hits += hit
        need = f"{r['needed']:.3f}" if r["needed"] is not None else "  n/a"
        print(f"  {r['i'] + 1:>4} {fmt_tok(backend.decode(r['tok'])):<16.16} {r['rank']:>6} {r['p']:>8.4f} "
              f"{need:>7} {tgt:>7.3f} {len(out['chosen']):>7,} {'YES' if hit else 'no':>10}")
        if r["i"] == 0:
            first = sorted(out["chosen"], key=lambda i: -r["probs"][i])[:args.show]
            first_rows = [(i, float(r["probs"][i])) for i in first]
    print(f"  instruction tokens recovered inside the solver's subset: {hits}/{len(rows)}")
    print(f"\n  decoded subset at step 1 (top {len(first_rows)} by prob):")
    for i, p in first_rows:
        print(f"     id {i:>6} p={p:.5f}  {fmt_tok(backend.decode(i))!r}")

    # 4b. FREE RUN with regressed targets
    n_free = args.max_tokens if args.max_tokens > 0 else len(sol_ids)
    print(f"\n[4] FREE RUN from the prompt with regressed targets (max {n_free} tokens, "
          f"stops at end-of-text; the instruction is not shown to the model):")
    rng = np.random.default_rng(args.seed)
    ctx = list(prompt_ids)
    every = 1 if n_free <= 40 else 25
    t0 = time.perf_counter()
    sizes = []
    for step in range(n_free):
        probs = backend.probs(ctx)
        order = np.argsort(-probs)
        pool = order[:K]
        ent, ltop = feats(probs)
        tgt = predict(ent, ltop)
        out = pool_step(probs, pool, tgt, args.scale_bits, args.engine, args.solver_steps, args.seed + 100 + step)
        nxt = sample_from(out["chosen"], [probs[i] for i in out["chosen"]], rng)
        sizes.append(len(out["chosen"]))
        if step % every == 0:
            want = sol_ids[step] if step < len(sol_ids) else None
            tail = ("" if want is None else
                    f" | instruction wanted {fmt_tok(backend.decode(want))!r} "
                    f"{'(in subset)' if want in set(out['chosen']) else '(outside subset)'}")
            print(f"  step {step + 1}: entropy {ent:.2f} -> target {tgt:.3f} | subset {len(out['chosen']):,} | "
                  f"sampled {fmt_tok(backend.decode(nxt))!r}{tail} | {time.perf_counter() - t0:.0f}s")
        if backend.eos is not None and nxt == backend.eos:
            print(f"  end-of-text sampled at step {step + 1}")
            break
        ctx.append(nxt)
    gen = ctx[len(prompt_ids):]
    match = sum(a == b for a, b in zip(gen, sol_ids))
    full_text = dec(gen)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(args.prompt + full_text)
        print(f"\nfull decoded text written to {args.out}")

    print("\n" + "=" * 78)
    print("TARGET (instruction) vs DECODED")
    print("=" * 78)
    print(f"instruction    : {args.prompt}{dec(sol_ids)!r}")
    print(f"decoded     : {args.prompt}{full_text!r}")
    print(f"\n--- decoded text (rendered, {len(gen)} tokens, mean subset {np.mean(sizes):,.0f}) ---")
    print(args.prompt + full_text)
    print("--- end ---\n")
    print(f"exact token matches (first {len(sol_ids)} positions) {match}/{len(sol_ids)} | instruction tokens inside solver subset {hits}/{len(rows)}")
    print("note: 'inside subset' measures whether the regressed target made the instruction reachable; "
          "'exact matches' also needs sampling to pick it.")



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
    ap.add_argument("--masses", default="", help="comma list of targets to compare, e.g. 0.3,0.6,0.9,0.99")
    ap.add_argument("--instruction", default="", help="target text (the instruction / what it could look like)")
    ap.add_argument("--max-tokens", type=int, default=240,
                    help="instruction mode: tokens to generate in the free run (0 = length of the instruction; e.g. 1000)")
    ap.add_argument("--out", default="", help="instruction mode: write the full decoded text to this file")
    ap.add_argument("--pool", type=int, default=200, help="top-K candidate tokens fetched per step")
    ap.add_argument("--margin", type=float, default=0.0, help="extra logit margin on the regressed target")
    ap.add_argument("--quiet", action="store_true", help="hide per-step detail, show only decoded subset + text")
    ap.add_argument("--steps", type=int, default=120, help="tokens to generate")
    ap.add_argument("--engine", default="Swap search", choices=list(MatchSubsetSumSolver.ENGINES))
    ap.add_argument("--scale-bits", type=int, default=52)
    ap.add_argument("--solver-steps", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show", type=int, default=25, help="decoded subset tokens to list at step 1")
    ap.add_argument("--plot", default="")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    for m in ([float(x) for x in a.masses.split(",")] if a.masses else [a.mass]):
        if not (0 < m < 1):
            sys.exit("targets must be between 0 and 1")
    run(a)
