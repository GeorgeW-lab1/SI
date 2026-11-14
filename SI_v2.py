#!/usr/bin/env python3
"""
Photonic GPT-2 Subset-Sum Decoder (Hugging Face Datasets & Local Transformers)

Pipeline:
1. Load a native Parquet dataset from Hugging Face (`datasets`).
2. Map features into a subset-sum instance with cryptographic commitment.
3. Send secret bits through a simulated photonic channel (optical pooling, confidence gating).
4. Prompt a local GPT-2 model via Hugging Face transformers to propose targeted bit repairs.
5. Verify candidates with exact NumPy matrix dot products and SHA-256 checks.

Requires: pip install transformers datasets torch numpy pandas huggingface-hub
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    sys.exit("Missing datasets dependency. Run: pip install datasets")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("Missing transformers/torch dependency. Run: pip install transformers torch")


# ------------------------------------------------------------------
# Photonic-channel parameters
# ------------------------------------------------------------------

ROWS, COLS = 8, 16
NBITS = ROWS * COLS
REVEAL_P = 0.45
FLIP_P = 0.02
LEVEL_BLACK = 0
LEVEL_GRAY = 127
LEVEL_WHITE = 255


# ------------------------------------------------------------------
# Local GPT-2 Wrapper
# ------------------------------------------------------------------

class LocalGPT2Decoder:
    """
    Local GPT-2 model wrapper for subset-sum repair proposals.
    Runs entirely on CPU or GPU without external API dependencies.
    """

    def __init__(
        self,
        model_name="gpt2",
        temperature=0.2,
        max_tokens=220,
        seed=1337
    ):
        self.model_name = model_name
        self.temperature = max(temperature, 1e-5) # Prevent zero division
        self.max_tokens = max_tokens
        self.seed = seed
        self.calls = 0
        self.prompt_chars = 0
        self.completion_chars = 0

        torch.manual_seed(seed)
        print(f"[+] Loading local GPT-2 model: '{self.model_name}'...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        print("[+] GPT-2 model loaded successfully.")

    def complete(self, prompt):
        self.calls += 1
        self.prompt_chars += len(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.90,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode only the generated tail
        prompt_len = input_ids.shape[1]
        generated_tokens = outputs[0][prompt_len:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        self.completion_chars += len(text)
        return text


# ------------------------------------------------------------------
# Hugging Face Dataset Ingestion & Subset-Sum Mapping
# ------------------------------------------------------------------

def load_or_fetch_dataset(
    dataset_name="zeroshot/twitter-financial-news-sentiment",
    subset_name=None,
    split="train",
    m=NBITS,
    seed=42,
    **kwargs
):
    """
    Fetches a native dataset from Hugging Face, avoiding legacy dataset scripts.
    """
    print(f"[+] Loading Hugging Face dataset: '{dataset_name}'")
    
    try:
        if subset_name:
            ds = hf_load_dataset(dataset_name, subset_name, split=split)
        else:
            ds = hf_load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"[-] Fallback loading default config due to: {e}")
        ds = hf_load_dataset(dataset_name, split=split)

    rng = np.random.default_rng(seed)
    
    weights = []
    for item in ds:
        raw_val = ""
        for v in item.values():
            if isinstance(v, (int, float)):
                raw_val = str(v)
                break
            elif isinstance(v, str):
                raw_val = v
                break
        
        digest = hashlib.sha256(raw_val.encode("utf-8")).digest()
        val = int.from_bytes(digest[:8], byteorder="big") % 900_000_000_000 + 100_000_000_000
        weights.append(val)
        
        if len(weights) >= m * 10:
            break

    weights = np.array(weights, dtype=np.int64)
    if len(weights) < m:
        weights = np.pad(weights, (0, m - len(weights)), mode="wrap")

    indices = rng.choice(len(weights), size=m, replace=False)
    weights = weights[indices]

    secret_b = rng.integers(0, 2, size=m, dtype=np.int8)
    if secret_b.sum() == 0:
        secret_b[0] = 1

    target_sum = int(np.dot(weights, secret_b))

    print(f"[+] Dataset mapped successfully")
    print(f"    Variables (m): {m}")
    print(f"    Target Sum:    {target_sum}")

    return weights, secret_b, target_sum


# ------------------------------------------------------------------
# Photonic Channel Simulation
# ------------------------------------------------------------------

class PhotonicChannel:
    def __init__(
        self,
        m,
        reveal_p=REVEAL_P,
        flip_p=FLIP_P,
        pixels_per_cell=144,
        pixel_noise=12.0,
        gain=0.80,
        offset=20.0,
        gamma=1.40,
        seed=7
    ):
        self.m = m
        self.reveal_p = reveal_p
        self.flip_p = flip_p
        self.rng = np.random.default_rng(seed)

        self.pixels_per_cell = pixels_per_cell
        self.sigma = pixel_noise / math.sqrt(pixels_per_cell)
        self.gain = gain
        self.offset = offset
        self.gamma = gamma

    def pooled(self, bits, reveal, flips):
        shown = np.bitwise_xor(bits, flips.astype(np.int8))
        levels = np.where(shown == 1, LEVEL_WHITE, LEVEL_BLACK)
        levels = np.where(reveal, levels, LEVEL_GRAY)

        optical = 255.0 * (levels.astype(np.float64) / 255.0) ** self.gamma
        optical = optical * self.gain + self.offset

        noise = self.rng.normal(0.0, self.sigma, size=self.m)
        return optical + noise

    def observe(self, secret_b):
        reveal = self.rng.random(self.m) < self.reveal_p
        flips = self.rng.random(self.m) < self.flip_p
        pooled_vals = self.pooled(secret_b, reveal, flips)
        return pooled_vals, reveal, flips


def calibrate_photonic_channel(channel, calibration_cycles=8):
    lows, grays, highs = [], [], []
    reveal = np.ones(channel.m, dtype=bool)
    no_flips = np.zeros(channel.m, dtype=np.int8)
    zeros = np.zeros(channel.m, dtype=np.int8)
    ones = np.ones(channel.m, dtype=np.int8)

    for _ in range(calibration_cycles):
        lows.append(channel.pooled(zeros, reveal, no_flips))
        grays.append(channel.pooled(zeros, reveal, no_flips))
        highs.append(channel.pooled(ones, reveal, no_flips))

    lo = np.mean(lows, axis=0)
    gray = np.mean(grays, axis=0)
    hi = np.mean(highs, axis=0)

    span = np.maximum(hi - lo, 1e-6)
    gray_z = (gray - lo) / span
    return lo, span, gray_z


def confidence_gate(pooled, lo, span, gray_z, gate_width=0.20):
    z = (pooled - lo) / span
    valid = np.abs(z - gray_z) >= gate_width
    bit = (z > gray_z).astype(np.uint8)
    return bit, valid


# ------------------------------------------------------------------
# Exact Cryptographic Verifier
# ------------------------------------------------------------------

def verify_candidate(weights, target_sum, candidate, salt, commit):
    candidate = np.asarray(candidate, dtype=np.uint8)
    if candidate.shape != weights.shape:
        return False, 0, target_sum, False

    candidate_sum = int(np.dot(weights.astype(np.int64), candidate.astype(np.int64)))
    packed = np.packbits(candidate).tobytes()
    digest = hashlib.sha256(salt + packed).digest()

    sum_match = candidate_sum == target_sum
    digest_match = digest == commit

    return (
        sum_match and digest_match,
        candidate_sum,
        target_sum - candidate_sum,
        digest_match
    )


# ------------------------------------------------------------------
# LLM Prompt Construction & Parsing
# ------------------------------------------------------------------

def build_llm_prompt(
    m, target_sum, current_sum, residual, candidate, margins, weights, uncertain, round_no, max_proposals, max_flips
):
    direction = "Current sum is TOO HIGH. Propose 1 -> 0 flips." if residual < 0 else "Current sum is TOO LOW. Propose 0 -> 1 flips."
    
    rows = [
        f"index={i}, weight={int(weights[i])}, current_bit={int(candidate[i])}, margin={int(margins[i])}"
        for i in uncertain
    ]

    return f"""Photonic subset-sum repair, round {round_no}.

Variables: {m}
Target sum: {target_sum}
Current sum: {current_sum}
Residual: {residual}
{direction}

Least reliable optical bits:
{"\n".join(rows)}

Propose up to {max_proposals} repair moves. Each move flips at most {max_flips} indices.
Return JSON strictly in this form:
{{"flips": [[index1, index2], [index3]]}}
JSON:"""


def parse_llm_proposals(text, m, max_proposals=8, max_flips=6):
    proposals = []
    try:
        # Try to find JSON block in output text
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        json_str = match.group(0) if match else text
        parsed = json.loads(json_str)
        raw = parsed.get("flips", parsed) if isinstance(parsed, dict) else parsed
        if isinstance(raw, list):
            proposals = raw if (raw and isinstance(raw[0], list)) else [raw]
    except Exception:
        numbers = [int(x) for x in re.findall(r"\b\d+\b", text)]
        if numbers:
            proposals = [numbers[:max_flips]]

    clean = []
    for proposal in proposals[:max_proposals]:
        if not isinstance(proposal, list):
            continue
        flips = []
        for val in proposal:
            try:
                idx = int(val)
            except Exception:
                continue
            if 0 <= idx < m and idx not in flips:
                flips.append(idx)
        if flips and len(flips) <= max_flips:
            clean.append(flips)
    return clean


# ------------------------------------------------------------------
# LLM-Guided Decoder Pipeline
# ------------------------------------------------------------------

def photonic_llm_decode(
    weights, target_sum, est_b, margins, salt, commit, llm,
    max_rounds=5, proposals_per_round=4, max_flips=6, uncertain_window=32, use_local_fallback=True
):
    m = len(weights)
    weights = weights.astype(np.int64)
    best = est_b.astype(np.uint8).copy()

    verified, best_sum, best_residual, _ = verify_candidate(weights, target_sum, best, salt, commit)
    candidate_tests = 1

    if verified:
        return best, {"verified": True, "source": "photonic_majority", "llm_calls": 0, "candidate_tests": candidate_tests, "rounds": 0, "residual": 0}

    best_abs_residual = abs(best_residual)
    uncertain = np.argsort(np.abs(margins), kind="stable")[:uncertain_window]

    for round_no in range(1, max_rounds + 1):
        prompt = build_llm_prompt(
            m, target_sum, best_sum, best_residual, best, margins, weights, uncertain, round_no, proposals_per_round, max_flips
        )

        response = llm.complete(prompt)
        proposals = parse_llm_proposals(response, m, max_proposals=proposals_per_round, max_flips=max_flips)

        if not proposals:
            continue

        round_best, round_best_sum, round_best_residual, round_best_abs_residual = None, best_sum, best_residual, best_abs_residual

        for flips in proposals:
            candidate = best.copy()
            candidate[np.asarray(flips, dtype=np.int64)] ^= 1

            verified, candidate_sum, residual, _ = verify_candidate(weights, target_sum, candidate, salt, commit)
            candidate_tests += 1

            if verified:
                return candidate, {"verified": True, "source": f"gpt2_round_{round_no}", "llm_calls": llm.calls, "candidate_tests": candidate_tests, "rounds": round_no, "residual": 0}

            if abs(residual) < round_best_abs_residual:
                round_best = candidate
                round_best_sum = candidate_sum
                round_best_residual = residual
                round_best_abs_residual = abs(residual)

        if round_best is not None:
            best, best_sum, best_residual, best_abs_residual = round_best, round_best_sum, round_best_residual, round_best_abs_residual
            print(f"[GPT-2] Round {round_no}: Residual improved to {best_residual}")

    if use_local_fallback:
        print("[+] Running deterministic residual fallback solver...")
        candidate, fallback_tests, verified = local_residual_repair(weights, target_sum, best, margins, salt, commit)
        candidate_tests += fallback_tests
        if verified:
            return candidate, {"verified": True, "source": "local_fallback", "llm_calls": llm.calls, "candidate_tests": candidate_tests, "rounds": max_rounds, "residual": 0}

    return best, {"verified": False, "source": "unverified", "llm_calls": llm.calls, "candidate_tests": candidate_tests, "rounds": max_rounds, "residual": int(best_residual)}


def local_residual_repair(weights, target_sum, start, margins, salt, commit, max_candidates=100_000):
    m = len(weights)
    candidate = start.astype(np.uint8).copy()
    current_sum = int(np.dot(weights, candidate))
    residual = target_sum - current_sum

    uncertain = np.argsort(np.abs(margins), kind="stable")
    ones_indices = uncertain[candidate[uncertain] == 1]
    zeros_indices = uncertain[candidate[uncertain] == 0]

    down = ones_indices[np.argsort(weights[ones_indices])[::-1]]
    up = zeros_indices[np.argsort(weights[zeros_indices])[::-1]]

    tests = 0
    while residual != 0 and tests < max_candidates:
        tests += 1
        improved = False
        if residual > 0:
            usable = up[weights[up] <= residual]
            if usable.size:
                idx = int(usable[0])
                candidate[idx] = 1
                current_sum += int(weights[idx])
                residual -= int(weights[idx])
                up = up[1:]
                improved = True
        else:
            usable = down[weights[down] <= -residual]
            if usable.size:
                idx = int(usable[0])
                candidate[idx] = 0
                current_sum -= int(weights[idx])
                residual += int(weights[idx])
                down = down[1:]
                improved = True

        if not improved:
            break

        packed = np.packbits(candidate).tobytes()
        if current_sum == target_sum and hashlib.sha256(salt + packed).digest() == commit:
            return candidate, tests, True

    packed = np.packbits(candidate).tobytes()
    return candidate, tests, (current_sum == target_sum and hashlib.sha256(salt + packed).digest() == commit)


# ------------------------------------------------------------------
# Main Execution Entry Point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Photonic GPT-2 Subset-Sum Decoder using Hugging Face datasets & transformers")
    parser.add_argument("--hf-dataset", default="zeroshot/twitter-financial-news-sentiment", help="Hugging Face dataset name")
    parser.add_argument("--hf-subset", default=None, help="Dataset subset config")
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--cols", type=int, default=COLS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cycles", type=int, default=49)
    parser.add_argument("--model-name", default="gpt2", help="Hugging Face model name (e.g. gpt2)")
    args = parser.parse_args()

    m = args.rows * args.cols

    # 1. Load real data from Hugging Face Datasets
    weights, secret_b, target_sum = load_or_fetch_dataset(
        dataset_name=args.hf_dataset,
        subset_name=args.hf_subset,
        m=m,
        seed=args.seed
    )

    salt = os.urandom(16)
    packed_secret = np.packbits(secret_b).tobytes()
    commit = hashlib.sha256(salt + packed_secret).digest()

    # 2. Simulate Photonic Channel
    channel = PhotonicChannel(m=m, seed=args.seed + 1)
    lo, span, gray_z = calibrate_photonic_channel(channel)

    ones_counts = np.zeros(m, dtype=np.int32)
    zeros_counts = np.zeros(m, dtype=np.int32)

    print(f"[+] Running {args.cycles} photonic observation cycles...")
    for cycle in range(1, args.cycles + 1):
        pooled, reveal, flips = channel.observe(secret_b)
        bit, valid = confidence_gate(pooled, lo, span, gray_z)
        ones_counts += (valid & (bit == 1)).astype(np.int32)
        zeros_counts += (valid & (bit == 0)).astype(np.int32)

    margins = ones_counts - zeros_counts
    est_b = (margins > 0).astype(np.uint8)

    # 3. Initialize Local GPT-2 Decoder
    llm = LocalGPT2Decoder(model_name=args.model_name, seed=args.seed)

    # 4. Run Iterative GPT-2 Repair Pipeline
    start_time = time.time()
    recovered, stats = photonic_llm_decode(
        weights=weights,
        target_sum=target_sum,
        est_b=est_b,
        margins=margins,
        salt=salt,
        commit=commit,
        llm=llm
    )
    elapsed = time.time() - start_time

    final_verified, final_sum, final_residual, digest_match = verify_candidate(
        weights, target_sum, recovered, salt, commit
    )

    print("\n================ PHOTONIC GPT-2 DECODER RESULTS ================")
    print(f"Verified exact recovery: {final_verified}")
    print(f"Final subset sum:        {final_sum}")
    print(f"Target subset sum:       {target_sum}")
    print(f"SHA-256 digest match:    {digest_match}")
    print(f"Bit errors vs secret:    {int(np.sum(secret_b != recovered))}")
    print(f"Model generation calls:  {stats['llm_calls']}")
    print(f"Candidate tests:         {stats['candidate_tests']}")
    print(f"Recovery source:         {stats['source']}")
    print(f"Elapsed time:            {elapsed:.3f} seconds")
    print("================================================================\n")


if __name__ == "__main__":
    main()
