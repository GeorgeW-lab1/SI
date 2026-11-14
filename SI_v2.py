#!/usr/bin/env python3
"""
Photonic-to-LLM Probability Bridge (Decoupled Reasoning & Bridged Steering)

Requires: pip install transformers datasets torch accelerate numpy pandas huggingface-hub
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import numpy as np
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
print("Environment variable TRANSFORMERS_VERBOSITY set to 'info'.")
try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    sys.exit("Missing datasets dependency. Run: pip install datasets")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
except ImportError:
    sys.exit("Missing transformers/torch dependency. Run: pip install transformers torch")


ROWS, COLS = 8, 16
NBITS = ROWS * COLS
REVEAL_P = 0.45
FLIP_P = 0.02
LEVEL_BLACK = 0
LEVEL_GRAY = 127
LEVEL_WHITE = 255


# ------------------------------------------------------------------
# Photonic-to-Token Probability Bridge (LogitsProcessor)
# ------------------------------------------------------------------

class PhotonicProbabilityBridgeProcessor(LogitsProcessor):
    """Transfers optical probability distributions into token logits during index selection."""
    def __init__(self, tokenizer, all_indices, optical_probabilities, boost_strength=4.0):
        self.boost_map = {}
        if len(all_indices) == 0:
            return

        num_indices = len(all_indices)
        for idx, prob in zip(all_indices, optical_probabilities):
            if prob <= 1e-6:
                continue
            token_ids = tokenizer.encode(str(int(idx)), add_special_tokens=False)
            bias = float(prob * boost_strength * num_indices)
            for tid in token_ids:
                self.boost_map[tid] = max(self.boost_map.get(tid, 0.0), bias)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.boost_map:
            return scores
        for tid, bias in self.boost_map.items():
            scores[:, tid] += bias
        return scores


# ------------------------------------------------------------------
# Local Qwen Model Wrapper with Decoupled Passes
# ------------------------------------------------------------------

class PhotonicBridgedDecoder:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.2,
        max_tokens=6400,
        seed=1337
    ):
        self.model_name = model_name
        self.temperature = max(temperature, 1e-5)
        self.max_tokens = max_tokens
        self.seed = seed
        self.calls = 0

        torch.manual_seed(seed)
        print(f"[+] Loading model: '{self.model_name}'...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )
        self.model.eval()
        print("[+] Model loaded successfully.")

    def reason(self, prompt):
        test = "init"
        while True:
            scopes = {
                    1: f"{test}",
                    2: "Write story about the future",
                    3: f"{test}",
                    4: "Design code about the following:",
                }
            for N, control_scope in scopes.items():
                
                import random; chunk = lambda path, n=10: (lambda w: " ".join(w[(i := random.randrange(max(1, len(w) - n + 1))):i+n]))(open(path).read().split());
                messages = [
                    {"role": "system", "content": f"Explain clearly as step {N} of the {control_scope}: "}, #{N}+{Control Scope} stepped lingual environment container/paradigm
                    {"role": "user", "content": chunk} #raw environment in
                ]
               
                    
                formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=6400,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.12,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    test = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    print(f"\n[Model Reasoning Output]:\n{test}\n")

# ------------------------------------------------------------------
# Dataset Ingestion & Photonic Channel Simulation
# ------------------------------------------------------------------

def load_or_fetch_dataset(dataset_name="zeroshot/twitter-financial-news-sentiment", m=NBITS, seed=42):
    print(f"[+] Loading dataset: '{dataset_name}'")
    ds = hf_load_dataset(dataset_name, split="train")

    rng = np.random.default_rng(seed)
    weights = []
    for item in ds:
        raw_val = next((str(v) for v in item.values() if isinstance(v, (int, float, str))), "")
        digest = hashlib.sha256(raw_val.encode("utf-8")).digest()
        val = int.from_bytes(digest[:8], byteorder="big") % 900_000_000_000 + 100_000_000_000
        weights.append(val)
        if len(weights) >= m * 10:
            break

    weights = np.array(weights, dtype=np.int64)
    if len(weights) < m:
        weights = np.pad(weights, (0, m - len(weights)), mode="wrap")

    weights = weights[rng.choice(len(weights), size=m, replace=False)]
    secret_b = rng.integers(0, 2, size=m, dtype=np.int8)
    if secret_b.sum() == 0:
        secret_b[0] = 1

    target_sum = int(np.dot(weights, secret_b))
    return weights, secret_b, target_sum


class PhotonicChannel:
    def __init__(self, m, seed=7):
        self.m = m
        self.rng = np.random.default_rng(seed)

    def observe(self, secret_b, reveal_p=REVEAL_P, flip_p=FLIP_P):
        reveal = self.rng.random(self.m) < reveal_p
        flips = self.rng.random(self.m) < flip_p
        shown = np.bitwise_xor(secret_b, flips.astype(np.int8))
        levels = np.where(shown == 1, LEVEL_WHITE, LEVEL_BLACK)
        levels = np.where(reveal, levels, LEVEL_GRAY)
        optical = 255.0 * (levels.astype(np.float64) / 255.0) ** 1.40 * 0.80 + 20.0
        return optical + self.rng.normal(0.0, 12.0 / math.sqrt(144), size=self.m)


def verify_candidate(weights, target_sum, candidate, salt, commit):
    candidate_sum = int(np.dot(weights, candidate))
    digest = hashlib.sha256(salt + np.packbits(candidate).tobytes()).digest()
    return candidate_sum == target_sum and digest == commit, candidate_sum, target_sum - candidate_sum


# ------------------------------------------------------------------
# Bridged Decoding Pipeline with Decoupled Reasoning
# ------------------------------------------------------------------

def photonic_bridged_decode(weights, target_sum, est_b, margins, salt, commit, llm, max_rounds=10):
    m = len(weights)
    best = est_b.astype(np.uint8).copy()
    verified, best_sum, best_residual = verify_candidate(weights, target_sum, best, salt, commit)
    

    all_indices = np.arange(m, dtype=np.int64)
    abs_margins = np.abs(margins).astype(np.float64)
    inv_conf = 1.0 / (abs_margins + 1.0)
    full_optical_probs = inv_conf / np.sum(inv_conf)
    sorted_by_prob = np.argsort(full_optical_probs)[::-1]

    while True:
        # Phase 1: Un-steered Natural Language Reasoning
        user_input = input("Prompt (or 'auto' / 'quit'): ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        reasoning_prompt = user_input
        reasoning_output = llm.reason(reasoning_prompt)
        



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    weights, secret_b, target_sum = load_or_fetch_dataset(m=NBITS)
    salt = os.urandom(16)
    commit = hashlib.sha256(salt + np.packbits(secret_b).tobytes()).digest()

    channel = PhotonicChannel(m=NBITS)
    pooled_accum = np.zeros(NBITS, dtype=np.float64)
    for _ in range(49):
        pooled_accum += channel.observe(secret_b)

    margins = pooled_accum - 49 * 127.0
    est_b = (margins > 0).astype(np.uint8)
    
    llm = PhotonicBridgedDecoder(model_name=args.model_name)
    
    t0 = time.time()
    recovered, stats = photonic_bridged_decode(weights, target_sum, est_b, margins, salt, commit, llm)
    elapsed = time.time() - t0

if __name__ == "__main__":
    main()
