import csv
import hashlib
import math
import os
import queue
import random
import re
import threading
import time
import tkinter as tk
import torch
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    HAS_TRANSFORMERS = False
from collections import Counter, deque
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    from PIL import Image, ImageTk
    HAS_WEBCAM_LIBS = True
except ImportError:
    HAS_WEBCAM_LIBS = False


"""
PHOTONIC + MODAL LOGIC PATTERN RECOGNISER (delay-line / autocorrelation style)

Photonic & logic chain
----------------------
Laser (P_peak mW, extinction ratio ER dB, RIN) or USB Webcam ROI A/B
  -> Beam splitter / coupler -> Photodetectors A & B
  -> TIA + RC node (tau = 50 ms, V > 0.7 V comparator -> b)
  -> Logic core:
       * Driver A -> a (optionally gated by SELECT / Q feedback)
       * pre_delay_and = a AND b, 150 ms delay -> delayed_and
       * Y = a OR b; D = delayed_and OR Y
       * N-bit FIFO queue (default 256 steps) between OR and 74HC74 -> d_queued
       * 74HC74 D-FF at 10 Hz: SELECT = Q, Observe = NOT Q

What is new in this version (making it an actual pattern recogniser)
--------------------------------------------------------------------
1. Structured inputs: "Repeating bits" (e.g. 1101001) and "Text stream"
   (ASCII -> bits, MSB first), one bit per 100 ms symbol, with optional bit-flip
   noise on the stream.
2. Like-with-like probe: D_queued is compared with D *now* (same signal, lagged
   by the queue length). On a stream that repeats at that lag this match rate
   climbs far above chance. The old "queue vs pre-delay AND" probe is kept, but it
   compares two different functions, so it only measures coincidence.
3. Lag sweep (autocorrelation): for every lag 1..N, match rate vs the chance
   baseline p1^2 + (1-p1)^2, with z-scores, a significance line, and automatic
   detection of the repeat period.

LLM link: encode generated tokens/text as bits, run the stream through, and the
lag sweep shows whether the output is looping (repetition detection) or contains
a copy-from-earlier-context structure (induction-head-style "seen this before").

Natural-text generation (mode "Generated text")
-----------------------------------------------
A word-level n-gram model is trained on a dataset (.txt/.csv or the built-in
sample). It generates natural text one token at a time. Each token is mapped to a
16-bit hash code (identical tokens -> identical bits, different tokens -> ~random
bits, so no ASCII byte-structure bias) and clocked through the photonic chain at one
bit per 100 ms. Before every token the recogniser runs a token-aligned lag sweep on
the recent bit stream. A significant match at a token lag means the text is looping.
With feedback ON the generator reacts: higher temperature, repetition penalty,
lower-order backoff, and a forced restart from a sentence start if the loop persists.
Turn feedback OFF at low temperature to watch the loops happen.

Trigram-index loader (dataset -> optical link -> trigram queue -> text model)
-----------------------------------------------------------------------------
The dataset is converted to overlapping trigrams (w1 w2 w3). Every unique trigram gets an
index; the dataset becomes a sequence of B-bit index words (+1 even-parity bit each),
framed as [16-bit SYNC][6-bit B][16-bit N][N words]. That bit stream crosses an optical link:
  * "Webcam optical link": a Tx window shows a DATA square and a CLOCK square (matching
    ROI B / ROI A). The webcam samples DATA on every CLOCK edge (self-clocked, data is set
    half a bit before the clock edge so camera latency/jitter is tolerated).
  * "Simulated photonic link": the same stream through the simulated laser -> coupler ->
    photodiode -> TIA -> RC -> comparator chain (no hardware needed).
The receiver hunts for SYNC, validates the header, and pushes the payload bits into the
Trigram Queue. The queue is then decoded (parity-checked), and the received trigram
counts become the n-gram text model. Generation (and loop feedback) then uses ONLY what
the queue received, so link errors really degrade the text model.
"""

STREAM_MODES = ("Repeating bits", "Text stream", "Generated text")
TOKEN_BITS = 16
PUNCT = set(".,!?;:")
SENT_END = set(".!?")
TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.,!?;:]")

SAMPLE_CORPUS = """
The morning train left the station slowly and the town woke up beneath a pale grey sky.
The baker opened the shop and the smell of warm bread drifted down the quiet street.
A small dog watched the people hurry past and the people watched the clock.
In the garden the roses were wet with rain and the bees were not yet awake.
The old man poured a cup of tea and sat by the window to read the news.
The news was the same as yesterday and the day before, but the tea was good and the chair was warm.
The teacher walked into the classroom and the children stopped talking at once.
She wrote a long question on the board and the children began to think.
Some of them looked at the window and some of them looked at the floor.
The river ran past the mill and the water was cold and clear.
A boy threw a stone into the river and the ripples spread across the water until they reached the bank.
The farmer drove the tractor across the field and the crows followed him down the long brown rows.
By noon the sky had cleared and the sun was bright over the hills.
The market was full of people buying apples, cheese and fresh bread.
The baker sold the last loaf and closed the shop for the afternoon.
In the evening the train came back and the town grew quiet again.
The old man closed his book and the dog went to sleep by the fire.
The rain returned in the night and the roses drank it slowly.
The next morning the train left the station and the town woke up beneath a pale grey sky.
The teacher looked at the children and the children looked at the board and the room grew quiet.
A woman carried a basket of apples across the bridge and the river sang below her.
The farmer sat by the fire and listened to the rain on the roof.
"""

_TOKEN_CODE_CACHE: dict = {}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def token_code(token: str, k: int = TOKEN_BITS) -> list[int]:
    """Deterministic k-bit hash code for a token (random-looking, identical tokens identical)."""
    key = (token, k)
    code = _TOKEN_CODE_CACHE.get(key)
    if code is None:
        n = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
        code = [(n >> i) & 1 for i in range(k)]
        _TOKEN_CODE_CACHE[key] = code
    return code


def mw_to_dbm(power_mw: float) -> float:
    return 10.0 * math.log10(max(power_mw, 1e-12))


def db_to_linear(db_val: float) -> float:
    return 10.0 ** (db_val / 10.0)


def text_to_bits(text: str) -> list[int]:
    """UTF-8 text -> list of bits, MSB first per byte."""
    bits: list[int] = []
    for byte in text.encode("utf-8"):
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits or [1]


def parse_bitstring(s: str) -> list[int]:
    bits = [int(ch) for ch in s if ch in "01"]
    return bits or [1]


def lag_sweep(bits: list[int], max_lag: int, z_threshold: float = 4.0, lags=None) -> dict:
    """
    Autocorrelation for binary streams.
    For each lag: match rate = P(bits[i] == bits[i+lag]); baseline = p1^2 + (1-p1)^2
    (what two independent streams with the same 1-density would give).
    Returns dict with per-lag (lag, rate, z), baseline, significance rate line,
    and the detected period (smallest significant lag near the peak rate).
    """
    n = len(bits)
    result = {"n": n, "rows": [], "baseline": 0.5, "sig_rate": 1.0, "period": None, "peak": None}
    if n < 6:
        return result
    p1 = sum(bits) / n
    base = p1 * p1 + (1.0 - p1) ** 2
    result["baseline"] = base

    rows = []
    lag_list = list(lags) if lags is not None else range(1, min(max_lag, n - 2) + 1)
    for lag in lag_list:
        if lag < 1 or lag > n - 2:
            continue
        m = n - lag
        matches = 0
        for i in range(m):
            if bits[i] == bits[i + lag]:
                matches += 1
        rate = matches / m
        sd = math.sqrt(base * (1.0 - base) / m) if 0.0 < base < 1.0 else 0.0
        z = (rate - base) / sd if sd > 0 else 0.0
        rows.append((lag, rate, z))
    result["rows"] = rows
    if not rows:
        return result

    sd_ref = math.sqrt(base * (1.0 - base) / n) if 0.0 < base < 1.0 else 0.0
    result["sig_rate"] = min(1.0, base + z_threshold * sd_ref)

    peak = max(rows, key=lambda r: r[1])
    result["peak"] = peak
    best_rate = peak[1]
    cands = [r for r in rows if r[2] >= z_threshold and r[1] >= best_rate - 0.02 and r[1] >= base + 0.10]
    if cands:
        result["period"] = min(cands, key=lambda r: r[0])
    return result


class ByteContextIndex:
    """In-memory byte-referenced context index for per-Q retrieval."""

    TERM_RE = re.compile(r"[\w'-]+", re.UNICODE)

    def __init__(self, chunk_bytes=384, overlap_bytes=64):
        self.chunk_bytes = max(128, int(chunk_bytes))
        self.overlap_bytes = min(max(0, int(overlap_bytes)), self.chunk_bytes - 1)
        self.chunks = []
        self.postings = {}
        self.doc_freq = {}

    @staticmethod
    def terms(text):
        return [term.lower() for term in ByteContextIndex.TERM_RE.findall(text)]



    def build(self, text):
        self.chunks.clear()
        self.postings.clear()
        self.doc_freq.clear()
        raw = text.encode("utf-8")
        step = max(1, self.chunk_bytes - self.overlap_bytes)
        start = 0
        chunk_id = 0
        while start < len(raw):
            end = min(len(raw), start + self.chunk_bytes)
            # Decode byte slices safely while preserving the byte reference.
            chunk_text = raw[start:end].decode("utf-8", errors="ignore").strip()
            if chunk_text:
                terms = self.terms(chunk_text)
                counts = {}
                for term in terms:
                    counts[term] = counts.get(term, 0) + 1
                self.chunks.append({
                    "id": chunk_id,
                    "byte_start": start,
                    "byte_end": end,
                    "text": chunk_text,
                    "terms": counts,
                    "length": max(1, len(terms))
                })
                for term in counts:
                    self.postings.setdefault(term, []).append(chunk_id)
                    self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
                chunk_id += 1
            if end >= len(raw):
                break
            start += step
        return len(self.chunks)



    def retrieve(self, query, limit=3):
        qterms = set(self.terms(query))
        if not qterms or not self.chunks:
            return []
        scores = {}
        total_chunks = max(1, len(self.chunks))
        for term in qterms:
            ids = self.postings.get(term, ())
            if not ids:
                continue
            # IDF-weighted term overlap; short byte chunks make this fast.
            idf = math.log(1.0 + total_chunks / (1.0 + self.doc_freq.get(term, 0)))
            for cid in ids:
                tf = self.chunks[cid]["terms"].get(term, 0)
                norm = math.sqrt(self.chunks[cid]["length"])
                scores[cid] = scores.get(cid, 0.0) + idf * tf / norm
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            {**self.chunks[cid], "score": score}
            for cid, score in ranked[:max(1, int(limit))]
            if score > 0
        ]



def detect_token_loop(token_ids, max_period=12, min_repeats=3):
    """Detect an exact repeated suffix in generated token IDs.

    Returns the shortest repeating suffix pattern repeated ``min_repeats``
    times, or None. Approximate matches are intentionally ignored.
    """
    tokens = list(token_ids)
    min_repeats = max(2, int(min_repeats))
    max_period = max(1, int(max_period))
    max_test_period = min(max_period, len(tokens) // min_repeats)
    for period in range(1, max_test_period + 1):
        repeated = tokens[-period * min_repeats:]
        pattern = tokens[-period:]
        if repeated == pattern * min_repeats:
            return {"period": period, "repeats": min_repeats, "pattern": tuple(pattern)}
    return None


class DecoderWorker:
    def __init__(self):
        self.commands = queue.Queue()
        self.events = queue.Queue()

        self.busy = threading.Event()
        self.shutdown = threading.Event()

        self.model = None
        self.tokenizer = None
        self.device = None

        self.active = False
        self.session_id = None

        self.cache = None
        self.next_input = None
        # Full decoder attention mask: cached prefix + current input token.
        # This is extended once per accepted token, keeping cache positions coherent.
        self.attention_mask = None
        self.last_accepted_q_edge = 0
        self.generated = []
        self.max_tokens = 128
        self.last_q_latency_ms = 0.0
        self.total_q_latency_ms = 0.0
        self.context_index = ByteContextIndex()
        self.context_index_chunks = 0
        self.last_retrieval_refs = []
        self.retrieval_token_budget = 48
        self.context_limit = None

        self.thread = threading.Thread(
            target=self.run,
            daemon=True
        )
        self.thread.start()

    def submit(self, kind, data):
        if self.busy.is_set() or self.shutdown.is_set():
            return False

        self.busy.set()
        self.commands.put((kind, data))
        return True

    def run(self):
        while not self.shutdown.is_set():
            try:
                kind, data = self.commands.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if kind == "load":
                    self.load_model(data)
                elif kind == "start":
                    self.start_session(data)
                elif kind == "token":
                    self.generate_one(data)

            except Exception as error:
                self.active = False

                self.events.put((
                    "error",
                    {
                        "operation": kind,
                        "session_id": self.session_id,
                        "message": str(error)
                    }
                ))

            finally:
                self.busy.clear()

    def load_model(self, name):
        if not HAS_TRANSFORMERS:
            raise RuntimeError(
                "Hugging Face Transformers is not installed. Install it with "
                "`pip install transformers` to enable the LLM; the photonic simulator "
                "can still run without it."
            )
        self.active = False
        self.session_id = None

        self.cache = None
        self.next_input = None
        self.attention_mask = None
        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Faster attention + reduced precision on supported hardware.
        load_kwargs = {"trust_remote_code": False}
        if device.type == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["attn_implementation"] = "sdpa"

        tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=False
        )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                name, **load_kwargs
            )
        except (TypeError, ValueError):
            # Older architectures may not support SDPA selection.
            load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                name, **load_kwargs
            )

        # Explicitly place every model parameter and buffer on the selected device.
        # Do not rely on from_pretrained() defaults or partial device maps.
        model.to(device)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.events.put((
            "loaded",
            {"message": f"Model loaded on {device}."}
        ))

    def start_session(self, data):
        if self.model is None:
            raise RuntimeError("Load a model first.")

        self.active = False
        self.session_id = data["session_id"]
        requested_max_tokens = max(1, int(data["max_tokens"]))

        prompt = data["prompt"]
        # Retrieval is performed once at session start against the selected corpus,
        # not against the already-cached prompt on every Q event.
        retrieval_corpus = data.get("retrieval_corpus", "") or ""
        self.context_index_chunks = self.context_index.build(retrieval_corpus)
        self.seed_context = prompt
        self.last_retrieval_refs = []
        retrieved = self.context_index.retrieve(prompt, limit=4)
        retrieval_blocks = []
        remaining_budget = max(0, int(self.retrieval_token_budget))
        for chunk in retrieved:
            if remaining_budget <= 0:
                break
            chunk_ids = self.tokenizer(
                chunk["text"], return_tensors=None, add_special_tokens=False
            )["input_ids"]
            # Tokenizers return a flat list for non-tensor calls; normalize it.
            if chunk_ids and isinstance(chunk_ids[0], list):
                chunk_ids = chunk_ids[0]
            take = min(len(chunk_ids), remaining_budget)
            if take <= 0:
                continue
            snippet = self.tokenizer.decode(chunk_ids[:take], skip_special_tokens=True).strip()
            if not snippet:
                continue
            retrieval_blocks.append(snippet)
            self.last_retrieval_refs.append({
                "id": chunk["id"],
                "byte_start": chunk["byte_start"],
                "byte_end": chunk["byte_end"],
                "score": round(float(chunk["score"]), 6),
                "tokens": take,
            })
            remaining_budget -= take


        model_prompt = prompt
        if retrieval_blocks:
            model_prompt = (
                "Retrieved context (added once at session start; do not repeat it verbatim):\n"
                + "\n---\n".join(retrieval_blocks)
                + "\n\nUser prompt / task:\n"
                + prompt
            )

        if self.tokenizer.chat_template:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": model_prompt}],
                tokenize=False,
                add_generation_prompt=True
            )

            encoded = self.tokenizer(
                formatted,
                return_tensors="pt",
                add_special_tokens=False
            )
        else:
            encoded = self.tokenizer(
                model_prompt,
                return_tensors="pt"
            )

        # Move every tensor returned by the tokenizer to the model device.
        encoded = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in encoded.items()
        }
        ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(ids, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(device=self.device, dtype=torch.long)

        if ids.shape[1] == 0:
            raise RuntimeError("The prompt contains no tokens.")

        context_limit = getattr(
            self.model.config,
            "max_position_embeddings",
            None
        )

        self.context_limit = context_limit if isinstance(context_limit, int) and context_limit > 0 else None
        # Clamp requested output to the actual remaining model context instead
        # of rejecting the session when prompt + requested tokens is too large.
        if self.context_limit is not None:
            remaining = self.context_limit - int(ids.shape[1])
            if remaining < 1:
                raise RuntimeError(
                    f"Prompt uses {ids.shape[1]} tokens, reaching the model context "
                    f"limit ({self.context_limit}); shorten the prompt."
                )
            self.max_tokens = min(requested_max_tokens, remaining)
        else:
            self.max_tokens = requested_max_tokens



        self.cache = None
        self.next_input = ids
        self.attention_mask = attention_mask
        self.last_accepted_q_edge = 0
        self.generated = []
        self.last_q_latency_ms = 0.0
        self.total_q_latency_ms = 0.0
        self.active = True

        self.events.put((
            "ready",
            {
                "session_id": self.session_id,
                "message": "Armed; waiting for Q HIGH."
            }
        ))

    def eos_ids(self):
        eos = self.model.generation_config.eos_token_id

        if eos is None:
            eos = self.tokenizer.eos_token_id

        if eos is None:
            return set()
        if isinstance(eos, int):
            return {eos}

        return set(eos)

    def generate_one(self, data):
        if (
            not self.active
            or data["session_id"] != self.session_id
        ):
            return

        control = data["control"]
        # A model step is legal only for a token-acceptance event produced by a
        # Q-high rising edge. No control signal is ever inserted into token IDs.
        if not bool(data.get("q_accept", False)):
            return
        try:
            q_edge = int(data.get("q_edge"))
        except (TypeError, ValueError):
            return
        # Each queued photonic acceptance event is uniquely numbered. Ignore
        # duplicate or stale credits so a replay cannot advance the LLM twice.
        if q_edge <= self.last_accepted_q_edge:
            return

        q_started = time.perf_counter()
        with torch.inference_mode():
            # Keep token inputs and attention mask on the model device. The
            # attention mask spans the cached prefix plus the current input.
            self.next_input = self.next_input.to(self.device)
            if self.attention_mask is None:
                raise RuntimeError("Decoder attention mask was not initialized.")
            self.attention_mask = self.attention_mask.to(self.device)

            # The prompt is encoded once in start_session and then advanced
            # autoregressively through next_input + KV cache. Never splice
            # retrieved tokens into a cached sequence on each Q event: doing so
            # corrupts positional continuity and causes incoherent output.
            model_input = self.next_input
            retrieval_refs = self.last_retrieval_refs

            result = self.model(
                input_ids=model_input,
                attention_mask=self.attention_mask,
                past_key_values=self.cache,
                use_cache=True
            )

            logits = result.logits[0, -1].to(self.device).float()
            new_cache = getattr(result, "past_key_values", None)
            if new_cache is None:
                raise RuntimeError(
                    "The selected causal language model did not return a KV cache. "
                    "Enable use_cache in the model configuration or choose a cache-capable model."
                )
            self.cache = new_cache

            # Repetition control changes token-selection logits only; the prompt,
            # token input, positions, and KV cache remain untouched.
            adjusted = logits.clone()
            recent_ids = self.generated[-64:]
            if recent_ids:
                counts = Counter(recent_ids)
                unique_ids = torch.tensor(list(counts.keys()), dtype=torch.long, device=self.device)
                selected = adjusted[unique_ids]
                repetition_penalty = 1.16
                adjusted[unique_ids] = torch.where(
                    selected > 0, selected / repetition_penalty, selected * repetition_penalty
                )
                count_values = torch.tensor(list(counts.values()), dtype=torch.float32, device=self.device)
                adjusted[unique_ids] -= 0.035 * count_values

            # Conservative exact n-gram loop detector. On three repeated cycles,
            # penalize loop tokens and block the token that would continue the
            # cycle, if at least one finite alternative remains. Sampling only;
            # model cache and autoregressive token history are not rewritten.
            loop_info = detect_token_loop(self.generated, max_period=12, min_repeats=3)
            loop_escape_token = None
            if loop_info:
                pattern_ids = [int(x) for x in loop_info["pattern"]
                              if 0 <= int(x) < adjusted.numel()]
                if pattern_ids:
                    pattern_tensor = torch.tensor(sorted(set(pattern_ids)), dtype=torch.long, device=self.device)
                    loop_logits = adjusted[pattern_tensor]
                    adjusted[pattern_tensor] = torch.where(
                        loop_logits > 0, loop_logits / 1.22, loop_logits * 1.22
                    )
                    loop_escape_token = int(loop_info["pattern"][0])
                    if 0 <= loop_escape_token < adjusted.numel():
                        candidate = adjusted.clone()
                        candidate[loop_escape_token] = -float("inf")
                        if torch.isfinite(candidate).any():
                            adjusted = candidate

            # Q clocks acceptance. Circuit control remains bounded and only
            # modestly influences temperature/nucleus width; loop recovery adds
            # a small sampling margin without changing the cache.
            base_temperature = min(0.50, max(0.25, float(control["temperature"])))
            temperature = min(0.54, base_temperature + (0.035 if loop_info else 0.0))
            top_p = 0.90 + (0.02 if bool(control.get("delayed_AND", False)) else 0.0)
            top_p += 0.03 if loop_info else 0.0
            top_p = min(0.95, max(0.88, top_p))
            scaled = adjusted / temperature
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            sorted_logits[remove] = -float("inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            position = torch.multinomial(probs, num_samples=1).item()
            token_id = int(sorted_indices[position].item())
            mode = ("loop-escape top-p sampling" if loop_info
                    else "Q-accepted top-p sampling")

            self.generated.append(token_id)
            self.last_accepted_q_edge = q_edge

            self.next_input = torch.tensor(
                [[token_id]],
                dtype=torch.long,
                device=self.device
            )
            # The sampled token has not yet been consumed by the model. Extend
            # the mask now so the next Q-accepted call sees [cached prefix + token].
            next_mask_bit = torch.ones(
                (self.attention_mask.shape[0], 1),
                dtype=self.attention_mask.dtype,
                device=self.device
            )
            self.attention_mask = torch.cat((self.attention_mask, next_mask_bit), dim=-1)

            text = self.tokenizer.decode(
                self.generated,
                skip_special_tokens=True
            )

            self.last_q_latency_ms = (time.perf_counter() - q_started) * 1000.0
            self.total_q_latency_ms += self.last_q_latency_ms

            finished = (
                token_id in self.eos_ids()
                or len(self.generated) >= self.max_tokens
            )

            if finished:
                self.active = False

            self.events.put((
                "token",
                {
                    "session_id": self.session_id,
                    "text": text,
                    "token_id": token_id,
                    "count": len(self.generated),
                    "q_latency_ms": self.last_q_latency_ms,
                    "avg_q_latency_ms": self.total_q_latency_ms / max(1, len(self.generated)),
                    "mode": mode,
                    "loop_detected": bool(loop_info),
                    "loop_period": loop_info["period"] if loop_info else None,
                    "loop_repeats": loop_info["repeats"] if loop_info else None,
                    "loop_pattern": list(loop_info["pattern"]) if loop_info else [],
                    "loop_escape_token": loop_escape_token,
                    "temperature": temperature,
                    "top_p": top_p,
                    "finished": finished,
                    "control": control,
                    "q_edge": q_edge,
                    "retrieval_refs": retrieval_refs,
                    "retrieval_chunks": len(retrieval_refs),
                    "context_index_chunks": self.context_index_chunks,
                    "attention_mask_tokens": int(self.attention_mask.shape[-1])
                }
            ))


class NGramModel:
    """Word-level n-gram language model trained on a text dataset, with backoff."""

    def __init__(self, text: str, order: int = 2):
        self.order = max(1, min(4, int(order)))
        self.tokens = tokenize(text)
        if len(self.tokens) < self.order + 8:
            raise ValueError("Dataset is too small (need at least a few dozen words).")
        toks = self.tokens
        self.unigram = Counter(toks)
        self.tables: list = [None] + [{} for _ in range(self.order)]
        for k in range(1, self.order + 1):
            tbl = self.tables[k]
            for i in range(len(toks) - k):
                key = tuple(toks[i:i + k])
                nxt = toks[i + k]
                bucket = tbl.get(key)
                if bucket is None:
                    bucket = tbl[key] = Counter()
                bucket[nxt] += 1
        self.starts = [i for i in range(len(toks) - self.order) if i == 0 or toks[i - 1] in SENT_END] or [0]
        self.start_contexts = [self.tokens[i:i + self.order] for i in self.starts]

    @classmethod
    def from_trigrams(cls, seq):
        """Build an order-2 model from a received trigram sequence (None entries = rejected words)."""
        valid = [t for t in seq if t is not None]
        if len(valid) < 8:
            raise ValueError("Too few valid trigrams were received to build a text model.")
        self = cls.__new__(cls)
        self.order = 2
        t1: dict = {}
        t2: dict = {}
        self.tables = [None, t1, t2]
        self.unigram = Counter()
        self.tokens = []
        starts: list[list[str]] = []
        prev = None
        first = True
        for t in seq:
            if t is None:
                prev = None
                continue
            a, b, c = t
            t2.setdefault((a, b), Counter())[c] += 1
            t1.setdefault((b,), Counter())[c] += 1
            self.unigram[c] += 1
            if first or (prev is not None and prev[0] in SENT_END):
                starts.append([a, b])
            if prev is not None and prev[1:] == (a, b):
                self.tokens.append(c)
            else:
                self.tokens.extend([a, b, c])
            prev = t
            first = False
        self.start_contexts = starts or [list(valid[0][:2])]
        self.starts = list(range(len(self.start_contexts)))
        return self

    def random_start(self, rng: random.Random) -> list[str]:
        return list(rng.choice(self.start_contexts))

    def sample(self, context, rng, temperature, recent: Counter, penalty: float, max_order: int) -> str:
        cand = None
        for k in range(min(max_order, len(context), self.order), 0, -1):
            cand = self.tables[k].get(tuple(context[-k:]))
            if cand:
                break
        if not cand:
            cand = self.unigram
        inv_t = 1.0 / max(0.05, temperature)
        words = list(cand.keys())
        weights = []
        for w in words:
            wt = cand[w] ** inv_t
            if penalty > 1.0:
                wt /= penalty ** recent.get(w, 0)
            weights.append(wt)
        return rng.choices(words, weights=weights, k=1)[0]


class TextStreamSource:
    """
    Generates natural text token by token and exposes it as a bit stream.
    Before each token the recogniser (token-aligned lag sweep over the recent bits)
    checks for a loop; with feedback on, the generator is steered out of it.
    """

    def __init__(self, ngram: NGramModel, seed: int, k: int = TOKEN_BITS,
                 temperature: float = 0.5, feedback: bool = True):
        self.ngram = ngram
        self.rng = random.Random(seed)
        self.K = k
        self.temperature = temperature
        self.feedback = feedback

        self.window_tokens = 32
        self.max_loop_tokens = 16
        self.recent_window = 16

        self.tokens: list[str] = []
        self.codes: list[list[int]] = []
        self.pieces: list[tuple[str, str]] = []
        self.bit_buffer: deque[int] = deque()
        self.context: list[str] = []
        self.pending: list[str] = list(ngram.random_start(self.rng))
        self.window_start = 0

        self.loop_active = False
        self.loop_period: int | None = None
        self.loop_tokens = 0
        self.loop_episodes = 0
        self.consecutive = 0
        self.kicks = 0
        self.bits_emitted = 0

    def next_bit(self) -> int:
        if not self.bit_buffer:
            self._generate_token()
        self.bits_emitted += 1
        return self.bit_buffer.popleft()

    def _detect_loop(self) -> int | None:
        codes = self.codes[self.window_start:][-self.window_tokens:]
        n_tok = len(codes)
        if n_tok < 6:
            return None
        bits = [b for code in codes for b in code]
        max_j = min(self.max_loop_tokens, n_tok // 2)
        lags = [j * self.K for j in range(1, max_j + 1)]
        res = lag_sweep(bits, max_j * self.K, z_threshold=6.0, lags=lags)
        cands = [r for r in res["rows"] if r[1] >= 0.9 and r[2] >= 6.0]
        if not cands:
            return None
        return min(cands, key=lambda r: r[0])[0] // self.K

    def _emit(self, tok: str, tag: str):
        first = not self.tokens
        piece = tok if (first or tok in PUNCT) else " " + tok
        self.tokens.append(tok)
        code = token_code(tok, self.K)
        self.codes.append(code)
        self.bit_buffer.extend(code)
        self.pieces.append((piece, tag))
        self.context = (self.context + [tok])[-self.ngram.order:]

    def _generate_token(self):
        period = self._detect_loop()
        if period is not None:
            if not self.loop_active:
                self.loop_episodes += 1
            self.loop_active = True
            self.loop_tokens += 1
            self.loop_period = period
            self.consecutive += 1
        else:
            self.loop_active = False
            self.consecutive = 0

        acting = self.feedback and self.loop_active
        tag = "loop" if self.loop_active else ""

        if acting and self.consecutive >= 2:
            self.kicks += 1
            self.consecutive = 0
            self.window_start = len(self.tokens)
            self.context = []
            self.pending = list(self.ngram.random_start(self.rng))
            tag = "kick"

        if self.pending:
            tok = self.pending.pop(0)
        else:
            recent = Counter(self.tokens[-self.recent_window:])
            tok = self.ngram.sample(
                self.context,
                self.rng,
                self.temperature + (0.5 if acting else 0.0),
                recent,
                3.0 if acting else 1.0,
                max(1, self.ngram.order - 1) if acting else self.ngram.order,
            )
        self._emit(tok, tag)

    def text(self) -> str:
        return "".join(p for p, _ in self.pieces)

    def distinct3(self) -> float:
        t = self.tokens
        if len(t) < 3:
            return 1.0
        tri = [tuple(t[i:i + 3]) for i in range(len(t) - 2)]
        return len(set(tri)) / len(tri)


SYNC_WORD = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
HEADER_BITS = 6 + 16


def int_to_bits(value: int, n: int) -> list[int]:
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def bits_to_int(bits) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


class TrigramCodec:
    """Dataset -> overlapping trigrams -> unique-trigram index words -> framed bit stream."""

    def __init__(self, text: str, max_tokens: int = 400):
        toks = tokenize(text)[:max_tokens]
        if len(toks) < 12:
            raise ValueError("Dataset is too small to build trigram index bits (need >= 12 tokens).")
        self.tokens = toks
        seq = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
        self.tri_list: list[tuple] = []
        self.tri_index: dict = {}
        for t in seq:
            if t not in self.tri_index:
                self.tri_index[t] = len(self.tri_list)
                self.tri_list.append(t)
        self.sequence = [self.tri_index[t] for t in seq]
        if len(self.sequence) > 65535:
            raise ValueError("Too many trigram words for the 16-bit length header (max 65535).")
        self.index_bits = max(1, math.ceil(math.log2(max(2, len(self.tri_list)))))
        self.word_bits = self.index_bits + 1  # + even parity

    def ground_truth(self) -> list[tuple]:
        return [self.tri_list[i] for i in self.sequence]

    def encode_frame(self) -> list[int]:
        bits = list(SYNC_WORD)
        bits += int_to_bits(self.index_bits, 6)
        bits += int_to_bits(len(self.sequence), 16)
        for idx in self.sequence:
            w = int_to_bits(idx, self.index_bits)
            bits += w + [sum(w) % 2]
        return bits


class TrigramQueue:
    """FIFO bit queue that receives the dataset's trigram index bits."""

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self.bits: deque[int] = deque(maxlen=self.capacity)

    def push(self, bit: int):
        self.bits.append(int(bit))

    def clear(self):
        self.bits.clear()

    def tail(self, n: int) -> list[int]:
        if n <= 0:
            return []
        data = list(self.bits)
        return data[-n:]


def decode_queue(codec: TrigramCodec, queue: TrigramQueue) -> list:
    """Queue bits -> trigram tuples (None where parity failed or the index is out of range)."""
    bits = list(queue.bits)
    wb = codec.word_bits
    out: list = []
    for i in range(0, len(bits) - wb + 1, wb):
        w = bits[i:i + wb]
        if sum(w) % 2:
            out.append(None)
            continue
        idx = bits_to_int(w[:-1])
        out.append(codec.tri_list[idx] if idx < len(codec.tri_list) else None)
    return out


class FrameReceiver:
    """Hunts for SYNC, validates the header, then pushes payload bits into the trigram queue."""

    HUNT, HEADER, PAYLOAD, DONE = range(4)
    STATE_NAMES = ["hunting for SYNC", "reading header", "receiving trigram bits", "complete"]

    def __init__(self, codec: TrigramCodec, queue: TrigramQueue):
        self.codec = codec
        self.queue = queue
        self.state = self.HUNT
        self.shift: deque[int] = deque(maxlen=len(SYNC_WORD))
        self.header: list[int] = []
        self.expected = len(codec.sequence) * codec.word_bits
        self.payload = 0
        self.bits_seen = 0
        self.sync_hits = 0
        self.false_syncs = 0

    @property
    def done(self) -> bool:
        return self.state == self.DONE

    @property
    def progress(self) -> float:
        return self.payload / self.expected if self.expected else 0.0

    def feed(self, bit: int):
        self.bits_seen += 1
        if self.state == self.HUNT:
            self.shift.append(int(bit))
            if len(self.shift) == len(SYNC_WORD) and list(self.shift) == SYNC_WORD:
                self.state = self.HEADER
                self.header = []
                self.sync_hits += 1
        elif self.state == self.HEADER:
            self.header.append(int(bit))
            if len(self.header) == HEADER_BITS:
                b = bits_to_int(self.header[:6])
                n = bits_to_int(self.header[6:])
                if b == self.codec.index_bits and n == len(self.codec.sequence):
                    self.state = self.PAYLOAD
                    self.queue.clear()
                    self.payload = 0
                else:
                    self.false_syncs += 1
                    self.state = self.HUNT
                    self.shift.clear()
        elif self.state == self.PAYLOAD:
            self.queue.push(bit)
            self.payload += 1
            if self.payload >= self.expected:
                self.state = self.DONE


class ClockedSampler:
    """Camera-side bit recovery: DATA is sampled on every CLOCK edge (self-clocked link)."""

    def __init__(self):
        self.last_clock: int | None = None
        self.edges = 0

    def update(self, clock: int, data: int):
        if self.last_clock is None:
            self.last_clock = int(clock)
            return None
        if int(clock) != self.last_clock:
            self.last_clock = int(clock)
            self.edges += 1
            return int(data)
        return None


class OpticalLinkSim:
    """Simulated photonic link: laser(ER, RIN) -> coupler -> PD B -> TIA -> RC -> comparator."""

    def __init__(self, laser_mw, er_db, split_a, loss_db, resp, tia_gain, rin_std, seed,
                 symbol_steps: int = 40, inject_ber: float = 0.0):
        self.laser_mw = laser_mw
        self.low_mw = laser_mw / max(db_to_linear(er_db), 1.0)
        self.split_b = 1.0 - min(max(split_a, 0.0), 1.0)
        self.loss = db_to_linear(-loss_db)
        self.resp = resp
        self.tia_gain = tia_gain
        self.rin_std = rin_std
        self.symbol_steps = max(1, int(symbol_steps))
        self.inject_ber = inject_ber
        self.rng = random.Random(seed + 11)
        self.rc = 0.0
        self.decay = math.exp(-PhotonicBenchModel.DT / PhotonicBenchModel.RC_TAU)

    def send_bit(self, bit: int) -> int:
        base = self.laser_mw if bit else self.low_mw
        sup = PhotonicBenchModel.SUPPLY
        for _ in range(self.symbol_steps):
            tx = max(0.0, base * (1.0 + self.rng.gauss(0.0, self.rin_std)))
            i_ma = tx * self.split_b * self.loss * self.resp
            v_t = min(sup, i_ma * self.tia_gain)
            self.rc = v_t + (self.rc - v_t) * self.decay
        rx = int(self.rc > PhotonicBenchModel.THRESHOLD)
        if self.inject_ber > 0.0 and self.rng.random() < self.inject_ber:
            rx ^= 1
        return rx


class WebcamWorker:
    """Captures latest frames from a USB webcam outside the Tkinter UI thread."""

    def __init__(self, camera_index: int):
        self.camera_index = camera_index
        self.frames: queue.Queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self, timeout: float = 1.5):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def _put_latest(self, item):
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            pass

    def _capture_loop(self):
        capture = None
        try:
            capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                self.error = (
                    f"Cannot open USB camera {self.camera_index}. "
                    "Check connection/permissions or close other camera apps."
                )
                return

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.error = "USB camera stopped returning frames."
                    break
                self._put_latest((time.perf_counter(), frame))
        except Exception as exc:
            self.error = str(exc)
        finally:
            if capture is not None:
                capture.release()

    def latest_frame(self):
        latest = None
        while True:
            try:
                latest = self.frames.get_nowait()
            except queue.Empty:
                return latest


class PhotonicBenchModel:
    """End-to-end photonic + logic discrete-step simulation engine."""

    DT = 0.005
    RC_TAU = 0.050
    SUPPLY = 5.0
    THRESHOLD = 0.7
    AND_DELAY = 0.150
    CLOCK_PERIOD = 0.100
    INPUT_PERIOD = 0.100

    def __init__(
        self,
        probe_duration: float = 2.0,
        seed: int = 7,
        mode: str = "Random",
        ignore_startup: bool = False,
        manual_x: bool = False,
        feed_select_to_driver_a: bool = True,
        d_queue_length: int = 256,
        laser_power_mw: float = 10.0,
        extinction_db: float = 20.0,
        split_ratio_a: float = 0.5,
        insertion_loss_db: float = 1.5,
        responsivity_a_per_w: float = 0.85,
        tia_gain_v_per_ma: float = 2.5,
        delay_loss_db: float = 0.5,
        optical_noise_std: float = 0.01,
        pattern_bits: str = "1101001",
        stream_text: str = "the cat sat on the mat. ",
        stream_noise: float = 0.0,
        ngram_model: NGramModel | None = None,
        gen_temperature: float = 0.5,
        gen_feedback: bool = True,
    ):
        self.probe_steps = max(1, int(round(probe_duration / self.DT)))
        self.total_steps = 2 * self.probe_steps
        self.probe_duration = self.probe_steps * self.DT
        self.total_duration = self.total_steps * self.DT

        self.flip_probability = 0.0
        self.seed = seed
        self.mode = mode
        self.ignore_startup = ignore_startup
        self.manual_x = manual_x
        self.feed_select_to_driver_a = feed_select_to_driver_a
        self.d_queue_length = max(1, int(d_queue_length))

        self.laser_power_mw = laser_power_mw
        self.extinction_db = extinction_db
        self.split_ratio_a = split_ratio_a
        self.insertion_loss_db = insertion_loss_db
        self.responsivity_a_per_w = responsivity_a_per_w
        self.tia_gain_v_per_ma = tia_gain_v_per_ma
        self.delay_loss_db = delay_loss_db
        self.optical_noise_std = optical_noise_std

        self.prompt_bits = [0]
        self.prompt_bit_index = 0
        self.pattern_stream = parse_bitstring(pattern_bits)
        self.text_stream = text_to_bits(stream_text)
        self.stream_noise = min(max(stream_noise, 0.0), 1.0)
        self.ngram_model = ngram_model
        self.gen_temperature = gen_temperature
        self.gen_feedback = gen_feedback

        self.delay_steps = max(1, int(round(self.AND_DELAY / self.DT)))
        self.clock_steps = max(1, int(round(self.CLOCK_PERIOD / self.DT)))
        self.input_steps = max(1, int(round(self.INPUT_PERIOD / self.DT)))

        self.reset()

    def set_queue_length(self, new_length: int):
        new_length = max(1, int(new_length))
        self.d_queue_length = new_length
        while len(self.d_bit_queue) > new_length:
            self.d_bit_queue.popleft()
        while len(self.d_bit_queue) < new_length:
            self.d_bit_queue.appendleft(self.queue_rng.randrange(2))
        self.d_queued = self.d_bit_queue[0]

    def reset(self):
        self.input_rng = random.Random(self.seed)
        self.photonic_rng = random.Random(self.seed + 2)
        self.queue_rng = random.Random(self.seed + 3)
        self.stream_rng = random.Random(self.seed + 4)
        self.gen: TextStreamSource | None = (
            TextStreamSource(self.ngram_model, self.seed + 5, TOKEN_BITS, self.gen_temperature, self.gen_feedback)
            if self.ngram_model is not None else None
        )

        self.step_index = 0
        self.time = 0.0

        self.x = 0
        self.random_a_bit = 1
        self.optical_tx_mw = 0.0
        self.optical_a_mw = 0.0
        self.optical_b_mw = 0.0
        self.photocurrent_a_ua = 0.0
        self.photocurrent_b_ua = 0.0
        self.optical_delayed_mw = 0.0
        self.optical_phase_rad = 0.0

        self.driver_a_raw = 0
        self.driver_a_select_gate = 1
        self.a = 0
        self.b = 0
        self.rc_voltage = 0.0

        self.pre_delay_and = 0
        self.truth = 0
        self.delayed_and = 0
        self.y = 0
        self.d = 0

        self.d_bit_queue: deque[int] = deque(
            self.queue_rng.randrange(2) for _ in range(self.d_queue_length)
        )
        self.d_queued = self.d_bit_queue[0]

        self.post_queue_and = 0

        # Legacy probe: queue out vs AND before delay (different functions)
        self.probe_count = 0
        self.probe_matches = 0
        self.probe_mismatches = 0
        self.last_match: int | None = None
        self.last_lag_steps = max(0, self.d_queue_length - 1)
        self.lag_steps_sum = 0

        # Like-with-like probe: D_queued vs D now (valid once queue holds real data)
        self.self_lag_count = 0
        self.self_lag_matches = 0
        self.last_self_match: int | None = None

        self.clock = 0
        self.q = 0
        self.not_q = 1
        self.q_history: deque[tuple[float, int]] = deque()
        self.q_high_fraction = 0.0
        self.q_temperature = 0.2

        self.accumulator = 1
        self.accumulator_samples = 0

        self.and_history: list[int] = []
        self.optical_and_history: list[float] = []
        self.records: list[dict] = []

    @property
    def finished(self) -> bool:
        return self.step_index >= self.total_steps

    def _update_inputs(self, cam_a: int | None = None, cam_b: int | None = None):
        if self.mode == "USB Webcam":
            # Webcam mode must not silently fall back to Manual input when the
            # camera frame becomes stale. Fail closed and hold the prompt cursor.
            if cam_a is None or cam_b is None:
                self.x = 0
                self.driver_a_raw = 0
                return
            if self.step_index % self.input_steps == 0:
                self.random_a_bit = self.input_rng.randrange(2)
            self.x = int(cam_b)
            prompt_bit = self.prompt_bits[self.prompt_bit_index % len(self.prompt_bits)] if self.prompt_bits else 0
            self.driver_a_raw = int(cam_a) & int(prompt_bit)
            if self.step_index % self.input_steps == 0 and self.prompt_bits:
                self.prompt_bit_index = (self.prompt_bit_index + 1) % len(self.prompt_bits)
            return

        if self.step_index % self.input_steps != 0:
            return
        update_num = self.step_index // self.input_steps

        if self.mode in STREAM_MODES:
            if self.mode == "Generated text" and self.gen is not None:
                bit = self.gen.next_bit()
            else:
                stream = self.pattern_stream if self.mode == "Repeating bits" else self.text_stream
                bit = stream[update_num % len(stream)]
            if self.stream_noise > 0.0 and self.stream_rng.random() < self.stream_noise:
                bit ^= 1
            self.x = bit
            self.driver_a_raw = bit
            return

        if self.mode == "Random":
            self.x = self.input_rng.randrange(2)
            self.driver_a_raw = self.input_rng.randrange(2)
        elif self.mode == "Square wave":
            self.x = update_num % 2
            self.driver_a_raw = self.input_rng.randrange(2)
        elif self.mode == "Manual":
            self.x = int(bool(self.manual_x))
            self.driver_a_raw = self.input_rng.randrange(2)
        else:  # "Direct (x -> a, b)" and "Photonic TIA"
            self.x = int(bool(self.manual_x))
            self.driver_a_raw = self.x

    def step(
        self,
        cam_a: int | None = None,
        cam_b: int | None = None,
        brightness_a: float = 0.0,
        brightness_b: float = 0.0,
        threshold_px: float = 130.0,
    ) -> bool:
        if self.finished:
            return False

        self._update_inputs(cam_a=cam_a, cam_b=cam_b)

        self.driver_a_select_gate = self.q if self.feed_select_to_driver_a else 1
        self.a = self.driver_a_raw & self.driver_a_select_gate

        er_linear = db_to_linear(self.extinction_db)
        low_mw = self.laser_power_mw / max(er_linear, 1.0)
        base_tx_mw = self.laser_power_mw if self.x else low_mw
        rin_factor = 1.0 + self.photonic_rng.gauss(0.0, self.optical_noise_std)
        self.optical_tx_mw = max(0.0, base_tx_mw * rin_factor)

        self.optical_phase_rad = (
            self.optical_phase_rad + 2.0 * math.pi * 1.5 * self.DT
        ) % (2.0 * math.pi)

        loss_lin = db_to_linear(-self.insertion_loss_db)
        split_a = min(max(self.split_ratio_a, 0.0), 1.0)
        split_b = 1.0 - split_a

        self.optical_a_mw = self.optical_tx_mw * split_a * loss_lin
        self.optical_b_mw = self.optical_tx_mw * split_b * loss_lin

        self.photocurrent_a_ua = max(0.0, self.optical_a_mw * self.responsivity_a_per_w * 1000.0)
        self.photocurrent_b_ua = max(0.0, self.optical_b_mw * self.responsivity_a_per_w * 1000.0)

        if self.mode == "Photonic TIA":
            drive_v_target = min(
                self.SUPPLY,
                (self.photocurrent_b_ua / 1000.0) * self.tia_gain_v_per_ma,
            )
        else:
            drive_v_target = self.SUPPLY * self.x

        decay = math.exp(-self.DT / self.RC_TAU)
        self.rc_voltage = drive_v_target + (self.rc_voltage - drive_v_target) * decay
        self.b = int(self.rc_voltage > self.THRESHOLD)

        self.pre_delay_and = self.a & self.b
        self.truth = self.pre_delay_and
        self.and_history.append(self.pre_delay_and)

        opt_and_mw = self.optical_tx_mw if self.pre_delay_and else 0.0
        self.optical_and_history.append(opt_and_mw)

        delayed_idx = self.step_index - self.delay_steps
        if delayed_idx >= 0:
            self.delayed_and = self.and_history[delayed_idx]
            self.optical_delayed_mw = (
                self.optical_and_history[delayed_idx] * db_to_linear(-self.delay_loss_db)
            )
        else:
            self.delayed_and = 0
            self.optical_delayed_mw = 0.0

        self.y = self.a | self.b
        self.d = self.delayed_and | self.y

        self.d_bit_queue.append(int(self.d))
        if len(self.d_bit_queue) > self.d_queue_length:
            self.d_queued = self.d_bit_queue.popleft()
        else:
            self.d_queued = self.d_bit_queue[0]
        effective_queue_steps = max(0, len(self.d_bit_queue) - 1)

        self.post_queue_and = self.d_queued & self.b
        self.last_lag_steps = effective_queue_steps

        # Like-with-like probe: lagged D vs D now, only once real data has propagated
        self_cell: int | str = ""
        if self.step_index >= self.d_queue_length:
            same = int(self.d_queued == self.d)
            self.last_self_match = same
            self.self_lag_count += 1
            self.self_lag_matches += same
            self_cell = same

        completed = self.step_index + 1
        self.time = completed * self.DT

        phase = completed % self.clock_steps
        self.clock = int(phase < self.clock_steps // 2)
        rising_edge = phase == 0

        if rising_edge:
            self.q = self.d_queued
            self.not_q = 1 - self.q
            self.q_history.append((self.time, self.q))
            while self.q_history and self.q_history[0][0] < self.time - 2.0:
                self.q_history.popleft()
            self.q_high_fraction = (sum(v for _, v in self.q_history) / len(self.q_history)) if self.q_history else 0.0
            self.q_temperature = 0.30 + 0.15 * self.q_high_fraction
            if not self.ignore_startup or delayed_idx >= 0:
                self.accumulator &= self.delayed_and
                self.accumulator_samples += 1

        probe_active = completed <= self.probe_steps
        match_cell: int | str = ""
        if probe_active:
            matched = int(self.d_queued == self.pre_delay_and)
            self.last_match = matched
            self.probe_count += 1
            self.probe_matches += matched
            self.probe_mismatches += 1 - matched
            self.lag_steps_sum += self.last_lag_steps
            match_cell = matched

        self.records.append({
            "time_s": round(self.time, 6),
            "mode": self.mode,
            "optical_x": self.x,
            "tx_power_mw": round(self.optical_tx_mw, 5),
            "tx_power_dbm": round(mw_to_dbm(self.optical_tx_mw), 3),
            "rx_a_mw": round(self.optical_a_mw, 5),
            "rx_b_mw": round(self.optical_b_mw, 5),
            "iph_a_ua": round(self.photocurrent_a_ua, 3),
            "iph_b_ua": round(self.photocurrent_b_ua, 3),
            "delayed_opt_mw": round(self.optical_delayed_mw, 5),
            "brightness_A": round(brightness_a, 2),
            "brightness_B": round(brightness_b, 2),
            "brightness_threshold": round(threshold_px, 2),
            "driver_a_raw": self.driver_a_raw,
            "driver_a_select_feedback": int(self.feed_select_to_driver_a),
            "driver_a_select_gate": self.driver_a_select_gate,
            "a": self.a,
            "rc_voltage_V": round(self.rc_voltage, 6),
            "b": self.b,
            "pre_delay_and": self.pre_delay_and,
            "delayed_and": self.delayed_and,
            "Y_or": self.y,
            "D_raw_or": self.d,
            "d_queue_len": len(self.d_bit_queue),
            "queue_lag_steps": self.last_lag_steps,
            "queue_lag_ms": round(self.last_lag_steps * self.DT * 1000.0, 2),
            "D_queued": self.d_queued,
            "post_queue_and": self.post_queue_and,
            "probe_active": int(probe_active),
            "queue_vs_pre_delay_and_match": match_cell,
            "queue_vs_D_now_match": self_cell,
            "gen_loop_active": int(self.gen.loop_active) if self.gen is not None else 0,
            "clock": self.clock,
            "rising_edge": int(rising_edge),
            "Q_SELECT": self.q,
            "not_Q_Observe": self.not_q,
            "Q_probe_high_fraction": self.q_high_fraction,
            "Q_probe_temperature": self.q_temperature,
            "running_and": self.accumulator,
            "accumulator_samples": self.accumulator_samples,
        })

        self.step_index = completed
        return True

    def symbol_samples(self, key: str) -> list[int]:
        """One sample per input symbol, taken at the last step of each symbol."""
        return [int(r[key]) for r in self.records[self.input_steps - 1::self.input_steps]]


class PhotonicCircuitSimulator(tk.Tk):
    ROI_A = (0.58, 0.25, 0.92, 0.75)  # Right ROI
    ROI_B = (0.08, 0.25, 0.42, 0.75)  # Left ROI

    LAG_SIGNALS = {
        "x (input stream)": "optical_x",
        "D (OR out)": "D_raw_or",
        "pre-delay AND": "pre_delay_and",
        "Q / SELECT": "Q_SELECT",
    }

    def __init__(self):
        super().__init__()
        self.title("Photonic & Modal Logic Pattern Recogniser — trigram queue via webcam -> text generation")
        self.geometry("1380x1060")
        self.minsize(960, 620)

        self.after_id: str | None = None
        self.cam_after_id: str | None = None
        self.resize_after_id: str | None = None
        self.running = False
        self.model: PhotonicBenchModel | None = None
        self.lag_result: dict | None = None
        self.prompt_bits = [0]
        self.prompt_bit_index = 0
        self.llm_decoder = DecoderWorker()
        self.llm_model_loaded = False
        self.llm_session_id = 0
        self.llm_session_active = False
        self.llm_pending_start = False
        self.llm_pending_start_payload = None
        self.llm_pending_tokens = deque()
        self.llm_q_edge_counter = 0
        self.llm_q_dropped = 0
        self.llm_token_records = []
        self.llm_model_var = tk.StringVar(value="gpt2")
        self.llm_max_tokens_var = tk.StringVar(value="128")
        self.llm_status_var = tk.StringVar(
            value=("LLM not loaded" if HAS_TRANSFORMERS else "LLM unavailable: install transformers")
        )
        self.llm_token_status_var = tk.StringVar(value="Tokens: 0")
        self.llm_prompt_text = None
        self.llm_output_text = None
        self.corpus_text = SAMPLE_CORPUS
        self.corpus_name = "built-in sample"
        self.ngram_cache: tuple | None = None
        self.shown_pieces = 0
        self.corpus_info_var = tk.StringVar()
        self.gen_stats_var = tk.StringVar(value="")
        self.order_var = tk.StringVar(value="2")
        self.temp_var = tk.StringVar(value="0.4")
        self.gen_fb_var = tk.BooleanVar(value=True)

        self.codec: TrigramCodec | None = None
        self.tqueue: TrigramQueue | None = None
        self.receiver: FrameReceiver | None = None
        self.queue_ngram: NGramModel | None = None
        self.loader_active = False
        self.loader_after_id: str | None = None
        self.link_sim: OpticalLinkSim | None = None
        self.sim_tx_bits: list[int] = []
        self.sim_tx_pos = 0
        self.sampler = ClockedSampler()
        self.tx_window = None
        self.tx_canvas = None
        self.tx_after_id: str | None = None
        self.tx_bits: list[int] = []
        self.tx_k = 0
        self.tx_t0 = 0.0
        self.tx_period = 0.12
        self.tx_data_on = 0
        self.tx_clock_on = 0
        self.link_source_var = tk.StringVar(value="Simulated photonic link")
        self.max_tokens_var = tk.StringVar(value="400")
        self.bit_ms_var = tk.StringVar(value="120")
        self.inject_ber_var = tk.StringVar(value="0.0")
        self.use_queue_var = tk.BooleanVar(value=True)
        self.load_status_var = tk.StringVar(value="Trigram queue: empty (press Build / Load into queue)")
        self.codec_info_var = tk.StringVar(value="")
        self.lag_info_var = tk.StringVar(value="Lag sweep: run the simulation, then press Analyse lags.")

        self.webcam: WebcamWorker | None = None
        self.latest_capture_time: float | None = None
        self.preview_image = None
        self.brightness_a = 0.0
        self.brightness_b = 0.0
        self.detector_a = 0
        self.detector_b = 0

        self.duration_var = tk.StringVar(value="60.0")
        self.speed_var = tk.StringVar(value="10.0")
        self.seed_var = tk.StringVar(value="7")
        self.queue_len_var = tk.StringVar(value="320")
        self.mode_var = tk.StringVar(value="Generated text")
        self.manual_x_var = tk.BooleanVar(value=False)
        self.ignore_startup_var = tk.BooleanVar(value=False)
        self.feedback_driver_a_var = tk.BooleanVar(value=False)

        self.pattern_var = tk.StringVar(value="1101001")
        self.text_var = tk.StringVar(value="the cat sat on the mat. ")
        self.stream_noise_var = tk.StringVar(value="0.0")

        self.lag_signal_var = tk.StringVar(value="x (input stream)")
        self.max_lag_var = tk.StringVar(value="256")

        self.cam_index_var = tk.StringVar(value="0")
        self.cam_thresh_var = tk.DoubleVar(value=130.0)

        self.laser_mw_var = tk.StringVar(value="10.0")
        self.er_db_var = tk.StringVar(value="20.0")
        self.split_a_var = tk.StringVar(value="0.5")
        self.loss_db_var = tk.StringVar(value="1.5")
        self.resp_var = tk.StringVar(value="0.85")
        self.tia_gain_var = tk.StringVar(value="2.5")
        self.rin_std_var = tk.StringVar(value="0.01")

        self.status_var = tk.StringVar()
        self.photonic_var = tk.StringVar()
        self.stats_var = tk.StringVar()
        self.cam_status_var = tk.StringVar(
            value="USB Cam: off" if HAS_WEBCAM_LIBS else "USB Cam: cv2/PIL unavailable"
        )

        self._build_ui()
        self.reset_simulation()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.llm_poll_after = self.after(100, self._llm_timer_poll)
        if HAS_WEBCAM_LIBS:
            self._poll_webcam()

    # ---------------------------------------------------------
    # Scrollable shell
    # ---------------------------------------------------------

    def _build_scrollable_shell(self) -> ttk.Frame:
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        self.scroll_canvas = tk.Canvas(outer, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(outer, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.v_scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        content_frame = ttk.Frame(self.scroll_canvas)
        self._scroll_window_id = self.scroll_canvas.create_window(
            (0, 0), window=content_frame, anchor="nw"
        )

        content_frame.bind("<Configure>", self._on_content_configure)
        self.scroll_canvas.bind("<Configure>", self._on_scroll_canvas_configure)
        self.scroll_canvas.bind("<Enter>", self._bind_mousewheel)
        self.scroll_canvas.bind("<Leave>", self._unbind_mousewheel)

        return content_frame

    def _on_content_configure(self, _event=None):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _on_scroll_canvas_configure(self, event):
        self.scroll_canvas.itemconfigure(self._scroll_window_id, width=event.width)

    def _on_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            self.scroll_canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            self.scroll_canvas.yview_scroll(3, "units")
        elif event.delta:
            steps = int(-event.delta / 120) or (-1 if event.delta > 0 else 1)
            self.scroll_canvas.yview_scroll(steps * 3, "units")

    def _bind_mousewheel(self, _event=None):
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel)
        self.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None):
        self.unbind_all("<MouseWheel>")
        self.unbind_all("<Button-4>")
        self.unbind_all("<Button-5>")

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def _build_ui(self):
        root = self._build_scrollable_shell()

        logic_frame = ttk.LabelFrame(root, text="Logic, Timing & Stream Controls", padding=7)
        logic_frame.pack(fill="x", padx=10, pady=(8, 4))

        top_fields = [
            ("Probe dur (s)", self.duration_var, 7),
            ("Speed", self.speed_var, 7),
            ("Seed", self.seed_var, 7),
            ("Queue len (steps)", self.queue_len_var, 7),
        ]
        for col, (lbl, var, w) in enumerate(top_fields):
            ttk.Label(logic_frame, text=lbl).grid(row=0, column=col * 2, padx=4, sticky="w")
            entry = ttk.Entry(logic_frame, textvariable=var, width=w)
            entry.grid(row=0, column=col * 2 + 1, padx=4)
            if var is self.queue_len_var:
                entry.bind("<FocusOut>", self._sync_live_vars)
                entry.bind("<Return>", self._sync_live_vars)

        ttk.Label(logic_frame, text="Input mode").grid(row=1, column=0, padx=4, pady=5, sticky="w")
        mode_box = ttk.Combobox(
            logic_frame,
            textvariable=self.mode_var,
            values=[
                "Generated text", "Repeating bits", "Text stream", "Random", "Square wave", "Manual",
                "Direct (x -> a, b)", "Photonic TIA", "USB Webcam",
            ],
            state="readonly",
            width=16,
        )
        mode_box.grid(row=1, column=1, padx=4, pady=5, sticky="w")
        mode_box.bind("<<ComboboxSelected>>", self._sync_live_vars)

        ttk.Checkbutton(
            logic_frame, text="Manual x HIGH", variable=self.manual_x_var, command=self._sync_live_vars
        ).grid(row=1, column=2, columnspan=2, padx=6, sticky="w")

        ttk.Checkbutton(
            logic_frame,
            text="Ignore startup in accum",
            variable=self.ignore_startup_var,
            command=self._sync_live_vars,
        ).grid(row=1, column=4, columnspan=2, padx=6, sticky="w")

        ttk.Checkbutton(
            logic_frame,
            text="Feed SELECT back into Driver A",
            variable=self.feedback_driver_a_var,
            command=self._sync_live_vars,
        ).grid(row=1, column=6, columnspan=4, padx=6, sticky="w")

        stream_frame = ttk.LabelFrame(
            logic_frame,
            text="Structured stream (Repeating bits / Text stream; one bit per 100 ms symbol; Reset to apply)",
            padding=5,
        )
        stream_frame.grid(row=2, column=0, columnspan=10, sticky="ew", pady=(4, 0))
        ttk.Label(stream_frame, text="Pattern bits").grid(row=0, column=0, padx=4)
        ttk.Entry(stream_frame, textvariable=self.pattern_var, width=22).grid(row=0, column=1, padx=4)
        ttk.Label(stream_frame, text="Text (ASCII->bits)").grid(row=0, column=2, padx=(12, 4))
        ttk.Entry(stream_frame, textvariable=self.text_var, width=34).grid(row=0, column=3, padx=4)
        ttk.Label(stream_frame, text="Stream flip p").grid(row=0, column=4, padx=(12, 4))
        ttk.Entry(stream_frame, textvariable=self.stream_noise_var, width=6).grid(row=0, column=5, padx=4)
        ttk.Label(
            stream_frame,
            text="Tip: queue steps = period_in_bits x 20 to align the hardware probe with the repeat.",
        ).grid(row=1, column=0, columnspan=6, sticky="w", padx=4, pady=(3, 0))

        load_frame = ttk.LabelFrame(
            logic_frame,
            text="Trigram-index loader: dataset -> trigram index bits -> optical link -> trigram queue -> text model",
            padding=5,
        )
        load_frame.grid(row=3, column=0, columnspan=10, sticky="ew", pady=(4, 0))
        ttk.Label(load_frame, text="Link").grid(row=0, column=0, padx=4, sticky="w")
        ttk.Combobox(
            load_frame, textvariable=self.link_source_var,
            values=["Simulated photonic link", "Webcam optical link"], state="readonly", width=22,
        ).grid(row=0, column=1, padx=4)
        ttk.Label(load_frame, text="Max tokens").grid(row=0, column=2, padx=(10, 4))
        ttk.Entry(load_frame, textvariable=self.max_tokens_var, width=6).grid(row=0, column=3, padx=4)
        ttk.Label(load_frame, text="Webcam bit (ms)").grid(row=0, column=4, padx=(10, 4))
        ttk.Entry(load_frame, textvariable=self.bit_ms_var, width=6).grid(row=0, column=5, padx=4)
        ttk.Label(load_frame, text="Inject BER (sim)").grid(row=0, column=6, padx=(10, 4))
        ttk.Entry(load_frame, textvariable=self.inject_ber_var, width=7).grid(row=0, column=7, padx=4)

        lbtn = ttk.Frame(load_frame)
        lbtn.grid(row=1, column=0, columnspan=10, sticky="w", pady=(4, 0))
        ttk.Button(lbtn, text="Build trigram bits", command=self.build_trigram_bits).pack(side="left", padx=4)
        ttk.Button(lbtn, text="Load into queue", command=self.start_load).pack(side="left", padx=4)
        ttk.Button(lbtn, text="Open Tx window", command=self.open_tx_window).pack(side="left", padx=4)
        ttk.Button(lbtn, text="Abort load", command=self._abort_load).pack(side="left", padx=4)
        ttk.Checkbutton(
            lbtn, text="Generate from trigram queue (applies on load/Reset)", variable=self.use_queue_var
        ).pack(side="left", padx=12)

        ttk.Label(load_frame, textvariable=self.codec_info_var).grid(
            row=2, column=0, columnspan=10, sticky="w", padx=4, pady=(3, 0)
        )
        prog = ttk.Frame(load_frame)
        prog.grid(row=3, column=0, columnspan=10, sticky="ew", pady=(3, 0))
        self.load_progress = ttk.Progressbar(prog, maximum=100, length=260)
        self.load_progress.pack(side="left", padx=4)
        ttk.Label(prog, textvariable=self.load_status_var).pack(side="left", padx=8)
        self.queue_canvas = tk.Canvas(
            load_frame, background="white", height=80, highlightthickness=1, highlightbackground="#d9e2e8"
        )
        self.queue_canvas.grid(row=4, column=0, columnspan=10, sticky="ew", pady=(4, 0))
        self.queue_canvas.bind("<Configure>", lambda _e: self._draw_queue_canvas())
        load_frame.columnconfigure(9, weight=1)

        gen_frame = ttk.LabelFrame(
            logic_frame,
            text="Natural-text generator (mode: Generated text) — n-gram model + recogniser loop feedback (Reset to apply dataset/order)",
            padding=5,
        )
        gen_frame.grid(row=4, column=0, columnspan=10, sticky="ew", pady=(4, 0))
        ttk.Button(gen_frame, text="Load dataset...", command=self.load_dataset).grid(row=0, column=0, padx=4)
        ttk.Button(gen_frame, text="Use sample", command=self.use_sample_corpus).grid(row=0, column=1, padx=4)
        ttk.Label(gen_frame, textvariable=self.corpus_info_var).grid(
            row=0, column=2, columnspan=6, padx=8, sticky="w"
        )
        ttk.Label(gen_frame, text="N-gram order (1-4)").grid(row=1, column=0, padx=4, pady=(4, 0), sticky="w")
        ttk.Entry(gen_frame, textvariable=self.order_var, width=4).grid(row=1, column=1, padx=4, pady=(4, 0), sticky="w")
        ttk.Label(gen_frame, text="Temperature").grid(row=1, column=2, padx=(12, 4), pady=(4, 0))
        temp_entry = ttk.Entry(gen_frame, textvariable=self.temp_var, width=6)
        temp_entry.grid(row=1, column=3, padx=4, pady=(4, 0))
        temp_entry.bind("<FocusOut>", self._sync_live_vars)
        temp_entry.bind("<Return>", self._sync_live_vars)
        ttk.Checkbutton(
            gen_frame, text="Recogniser anti-loop feedback", variable=self.gen_fb_var, command=self._sync_live_vars
        ).grid(row=1, column=4, columnspan=3, padx=12, pady=(4, 0), sticky="w")

        cam_frame = ttk.LabelFrame(
            logic_frame, text="USB Webcam ROI Input (Left = B -> x, Right = A -> Driver A)", padding=5
        )
        cam_frame.grid(row=5, column=0, columnspan=10, sticky="ew", pady=(4, 0))

        ttk.Label(cam_frame, text="USB Cam idx").grid(row=0, column=0, padx=4)
        ttk.Entry(cam_frame, textvariable=self.cam_index_var, width=5).grid(row=0, column=1, padx=4)

        self.start_cam_btn = ttk.Button(
            cam_frame, text="Start USB Cam", command=self.start_webcam,
            state="normal" if HAS_WEBCAM_LIBS else "disabled",
        )
        self.start_cam_btn.grid(row=0, column=2, padx=4)

        self.stop_cam_btn = ttk.Button(
            cam_frame, text="Stop USB Cam", command=self.stop_webcam,
            state="normal" if HAS_WEBCAM_LIBS else "disabled",
        )
        self.stop_cam_btn.grid(row=0, column=3, padx=4)

        ttk.Label(cam_frame, text="ROI Thresh").grid(row=0, column=4, padx=(10, 4))
        ttk.Scale(
            cam_frame, from_=0, to=255, variable=self.cam_thresh_var, orient="horizontal", length=160
        ).grid(row=0, column=5, padx=4)

        ttk.Label(cam_frame, textvariable=self.cam_status_var).grid(row=0, column=6, padx=10, sticky="w")

        phot_frame = ttk.LabelFrame(root, text="Photonic Front-End Parameters (Reset to apply)", padding=7)
        phot_frame.pack(fill="x", padx=10, pady=(0, 4))

        phot_fields = [
            ("Laser P (mW)", self.laser_mw_var, 7),
            ("ER (dB)", self.er_db_var, 7),
            ("Split A", self.split_a_var, 7),
            ("Coupler loss (dB)", self.loss_db_var, 7),
            ("Resp (A/W)", self.resp_var, 7),
            ("TIA (V/mA)", self.tia_gain_var, 7),
            ("RIN sigma", self.rin_std_var, 7),
        ]
        for col, (lbl, var, w) in enumerate(phot_fields):
            r = 0 if col < 4 else 1
            c_idx = (col % 4) * 2
            ttk.Label(phot_frame, text=lbl).grid(row=r, column=c_idx, padx=4, pady=2, sticky="w")
            ttk.Entry(phot_frame, textvariable=var, width=w).grid(row=r, column=c_idx + 1, padx=4, pady=2)

        btns = ttk.Frame(root, padding=(10, 2))
        btns.pack(fill="x")

        self.run_btn = ttk.Button(btns, text="Run", command=self.toggle_running)
        self.run_btn.pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Step 5 ms", command=self.single_step).pack(side="left", padx=4)
        ttk.Button(btns, text="Reset", command=self.reset_simulation).pack(side="left", padx=4)
        ttk.Button(btns, text="Run to end", command=self.run_to_end).pack(side="left", padx=4)
        ttk.Button(btns, text="Export CSV", command=self.export_csv).pack(side="left", padx=4)
        ttk.Button(btns, text="Export text", command=self.export_text).pack(side="left", padx=4)

        ttk.Label(root, textvariable=self.status_var, padding=(12, 2)).pack(fill="x")
        ttk.Label(root, textvariable=self.photonic_var, padding=(12, 2)).pack(fill="x")

        middle = ttk.Frame(root)
        middle.pack(fill="both", expand=True, padx=10, pady=4)

        self.canvas = tk.Canvas(
            middle, background="white", height=520, highlightthickness=1, highlightbackground="#d9e2e8"
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._schedule_redraw)

        self.cam_panel = ttk.LabelFrame(middle, text="USB Webcam Feed", padding=4, width=320)
        self.cam_preview = ttk.Label(self.cam_panel, text="USB cam stopped", anchor="center")
        self.cam_preview.pack(fill="both", expand=True)
        if HAS_WEBCAM_LIBS:
            self.cam_panel.pack(side="right", fill="y", padx=(8, 0))

        gen_text_frame = ttk.LabelFrame(
            root,
            text="Generated text (red = recogniser flagged a loop, orange = forced restart)",
            padding=4,
        )
        gen_text_frame.pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(gen_text_frame, textvariable=self.gen_stats_var).pack(fill="x", pady=(0, 3))
        text_row = ttk.Frame(gen_text_frame)
        text_row.pack(fill="x")
        self.gen_text = tk.Text(text_row, height=7, wrap="word", font=("Georgia", 11))
        gen_scroll = ttk.Scrollbar(text_row, orient="vertical", command=self.gen_text.yview)
        self.gen_text.configure(yscrollcommand=gen_scroll.set, state="disabled")
        self.gen_text.tag_configure("loop", background="#f8d7da")
        self.gen_text.tag_configure("kick", background="#ffd9a8")
        gen_scroll.pack(side="right", fill="y")
        self.gen_text.pack(side="left", fill="x", expand=True)

        ttk.Label(root, textvariable=self.stats_var, padding=(12, 2)).pack(fill="x")

        self.waveform = tk.Canvas(
            root, background="#f7fafc", height=180, highlightthickness=1, highlightbackground="#d9e2e8"
        )
        self.waveform.pack(fill="x", padx=10, pady=(0, 6))
        self.waveform.bind("<Configure>", self._schedule_redraw)

        # LLM controls are additive: retain the original circuit, optical link,
        # trigram queue, lag analysis, n-gram tools and waveform UI.
        llm_frame = ttk.LabelFrame(root, text="Prompt-driven Driver A + Q-clocked LLM", padding=6)
        llm_frame.pack(fill="x", padx=10, pady=(0, 8))
        model_row = ttk.Frame(llm_frame)
        model_row.pack(fill="x")
        ttk.Label(model_row, text="HF model").pack(side="left", padx=4)
        ttk.Entry(model_row, textvariable=self.llm_model_var, width=36).pack(side="left", padx=4)
        ttk.Button(model_row, text="Load LLM", command=self.llm_load_model).pack(side="left", padx=4)
        ttk.Label(model_row, text="Max tokens").pack(side="left", padx=(10, 4))
        ttk.Entry(model_row, textvariable=self.llm_max_tokens_var, width=7).pack(side="left", padx=4)
        ttk.Button(model_row, text="Arm LLM", command=self.llm_arm).pack(side="left", padx=4)
        ttk.Button(model_row, text="Stop LLM", command=self.llm_stop).pack(side="left", padx=4)
        ttk.Label(model_row, textvariable=self.llm_status_var).pack(side="left", padx=8)
        prompt_row = ttk.Frame(llm_frame)
        prompt_row.pack(fill="x", pady=(5, 3))
        self.llm_prompt_text = tk.Text(prompt_row, height=4, wrap="word")
        self.llm_prompt_text.pack(fill="x", expand=True)
        self.llm_prompt_text.insert("1.0", "Write a continuation based on the context.\n")
        self.llm_prompt_text.bind("<KeyRelease>", self.llm_update_prompt_bits)
        self.llm_output_text = tk.Text(llm_frame, height=6, wrap="word", state="disabled")
        self.llm_output_text.pack(fill="x", pady=(3, 0))
        ttk.Label(llm_frame, textvariable=self.llm_token_status_var).pack(fill="x")
        self.llm_update_prompt_bits()

        lag_frame = ttk.LabelFrame(
            root, text="Lag sweep (autocorrelation): match rate vs lag, in input symbols", padding=6
        )
        lag_frame.pack(fill="x", padx=10, pady=(0, 10))

        ctl = ttk.Frame(lag_frame)
        ctl.pack(fill="x")
        ttk.Label(ctl, text="Signal").pack(side="left", padx=4)
        ttk.Combobox(
            ctl, textvariable=self.lag_signal_var, values=list(self.LAG_SIGNALS.keys()),
            state="readonly", width=18,
        ).pack(side="left", padx=4)
        ttk.Label(ctl, text="Max lag").pack(side="left", padx=(12, 4))
        ttk.Entry(ctl, textvariable=self.max_lag_var, width=6).pack(side="left", padx=4)
        ttk.Button(ctl, text="Analyse lags", command=self.analyse_lags).pack(side="left", padx=10)
        ttk.Label(lag_frame, textvariable=self.lag_info_var, wraplength=1200, justify="left").pack(
            fill="x", pady=(4, 2)
        )

        self.lag_canvas = tk.Canvas(
            lag_frame, background="white", height=200, highlightthickness=1, highlightbackground="#d9e2e8"
        )
        self.lag_canvas.pack(fill="x", pady=(2, 0))
        self.lag_canvas.bind("<Configure>", self._schedule_redraw)

    # ---------------------------------------------------------
    # Prompt bitstream + Q-clocked LLM
    # ---------------------------------------------------------
    def llm_update_prompt_bits(self, _event=None):
        if self.llm_prompt_text is None:
            return
        prompt = self.llm_prompt_text.get("1.0", "end-1c")
        bits = text_to_bits(prompt)
        self.prompt_bits = bits or [0]
        self.prompt_bit_index %= len(self.prompt_bits)
        if self.model is not None:
            self.model.prompt_bits = list(self.prompt_bits)
            self.model.prompt_bit_index = self.prompt_bit_index

    def llm_load_model(self):
        if not HAS_TRANSFORMERS:
            messagebox.showerror(
                "Transformers not installed",
                "The photonic simulator can run, but LLM loading requires:\n\n"
                "pip install transformers\n\n"
                "Then restart the application."
            )
            return
        name = self.llm_model_var.get().strip()
        if not name:
            return
        if self.llm_decoder.busy.is_set():
            messagebox.showinfo("LLM", "Wait for the current model operation to finish.")
            return
        self.llm_model_loaded = False
        if self.llm_decoder.submit("load", name):
            self.llm_status_var.set("Loading model...")

    def llm_arm(self):
        if not self.llm_model_loaded:
            messagebox.showwarning("LLM", "Load the model first.")
            return
        prompt = self.llm_prompt_text.get("1.0", "end-1c").strip()
        if not prompt:
            messagebox.showwarning("Prompt", "Enter prompt/context text.")
            return
        try:
            max_tokens = int(self.llm_max_tokens_var.get())
            if not 1 <= max_tokens <= 2048:
                raise ValueError
        except ValueError:
            messagebox.showerror("Max tokens", "Enter a value from 1 to 2048.")
            return
        self.llm_session_id += 1
        self.llm_session_active = False
        self.llm_pending_start = True
        self.llm_pending_tokens.clear()
        self.llm_q_edge_counter = 0
        self.llm_q_dropped = 0
        self.llm_token_records.clear()
        self.llm_output_text.configure(state="normal")
        self.llm_output_text.delete("1.0", "end")
        self.llm_output_text.configure(state="disabled")
        self.llm_update_prompt_bits()
        self.prompt_bit_index = 0
        # Reset the live Driver-A prompt bit cursor in the circuit as well,
        # so webcam ROI-A starts gating against the first prompt bit each arm.
        if self.model is not None:
            self.model.prompt_bits = list(self.prompt_bits)
            self.model.prompt_bit_index = 0
        payload = {
            "session_id": self.llm_session_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "retrieval_corpus": getattr(self, "corpus_text", "") or "",
        }
        # Keep the start request if a previous inference call is still in flight.
        # The UI poller will submit it as soon as the decoder worker becomes idle.
        self.llm_pending_start_payload = payload
        self.llm_pending_start = True
        self.llm_dispatch_start()

    def llm_dispatch_start(self):
        payload = self.llm_pending_start_payload
        if payload is None or self.llm_decoder.busy.is_set():
            return
        if self.llm_decoder.submit("start", payload):
            self.llm_pending_start_payload = None
            self.llm_status_var.set("Preparing prompt...")

    def llm_stop(self):
        self.llm_session_active = False
        self.llm_pending_start = False
        self.llm_pending_start_payload = None
        self.llm_pending_tokens.clear()
        self.llm_status_var.set("Stopped; in-flight tokens will be ignored.")

    def _llm_timer_poll(self):
        self.llm_poll_after = None
        self.llm_poll()
        if not getattr(self, "closing", False):
            self.llm_poll_after = self.after(100, self._llm_timer_poll)

    def llm_poll(self):
        while True:
            try:
                kind, data = self.llm_decoder.events.get_nowait()
            except queue.Empty:
                break
            if kind == "loaded":
                self.llm_model_loaded = True
                self.llm_status_var.set(data.get("message", "Loaded"))
            elif kind == "ready" and data.get("session_id") == self.llm_session_id and self.llm_pending_start:
                self.llm_pending_start = False
                self.llm_session_active = True
                self.llm_status_var.set("Armed; waiting for Q HIGH")
            elif kind == "error":
                # Ignore failures from an older session after the user has
                # armed a newer one. Model-load errors have no session id.
                error_session = data.get("session_id")
                if error_session is None or error_session == self.llm_session_id:
                    self.llm_session_active = False
                    self.llm_pending_start = False
                    self.llm_pending_start_payload = None
                    self.llm_pending_tokens.clear()
                    self.llm_status_var.set("LLM error: " + data.get("message", "unknown"))
            elif (kind == "token" and data.get("session_id") == self.llm_session_id
                  and self.llm_session_active):
                self.llm_output_text.configure(state="normal")
                self.llm_output_text.delete("1.0", "end")
                self.llm_output_text.insert("1.0", data.get("text", ""))
                self.llm_output_text.configure(state="disabled")
                self.llm_token_records.append(data)
                self.llm_token_status_var.set(
                    f"Tokens: {data.get('count', 0)} | {data.get('mode')} | "
                    f"Q edge: {data.get('q_edge', '?')} | "
                    f"T: {data.get('control', {}).get('temperature', 0):.3f} | "
                    f"retrieved chunks: {data.get('retrieval_chunks', 0)} | "
                    f"loop period: {data.get('loop_period') or '-'} | "
                    f"queued Q: {len(self.llm_pending_tokens)} | dropped: {self.llm_q_dropped}"
                )
                if data.get("finished"):
                    self.llm_session_active = False
                    self.llm_pending_tokens.clear()
                    self.llm_status_var.set("Generation finished")
        self.llm_dispatch_start()
        self.llm_dispatch_pending()

    def llm_request_from_q(self):
        if not self.llm_session_active or self.model is None:
            return
        # Q-high rising edges are explicit token-acceptance credits. Keep a
        # bounded FIFO so inference latency does not silently overwrite events;
        # excess edges are rejected visibly instead of growing without bound.
        queue_limit = 8
        if len(self.llm_pending_tokens) >= queue_limit:
            self.llm_q_dropped += 1
            self.llm_token_status_var.set(
                f"Q queue full ({queue_limit}); rejected edge | dropped: {self.llm_q_dropped}"
            )
            return
        self.llm_q_edge_counter += 1
        m = self.model
        q_fraction = float(getattr(m, "q_high_fraction", 0.0))
        q_fraction = min(1.0, max(0.0, q_fraction))
        control = {
            "temperature": 0.30 + 0.15 * q_fraction,
            "delayed_AND": bool(m.delayed_and),
        }
        self.llm_pending_tokens.append({
            "session_id": self.llm_session_id,
            "q_accept": True,
            "q_edge": self.llm_q_edge_counter,
            "control": control,
        })
        self.llm_dispatch_pending()



    def llm_dispatch_pending(self):
        if (self.llm_pending_tokens and self.llm_session_active
                and not self.llm_decoder.busy.is_set()):
            payload = self.llm_pending_tokens[0]
            if self.llm_decoder.submit("token", payload):
                self.llm_pending_tokens.popleft()

    # ---------------------------------------------------------
    # USB Webcam Controls
    # ---------------------------------------------------------

    def start_webcam(self):
        if not HAS_WEBCAM_LIBS:
            return
        if self.webcam is not None and self.webcam.is_alive:
            messagebox.showinfo("USB Webcam", "Stop the active USB camera first.")
            return
        try:
            idx = int(self.cam_index_var.get())
            if idx < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid camera index", "Enter a non-negative integer (e.g., 0 or 1).")
            return

        self.latest_capture_time = None
        self.detector_a = 0
        self.detector_b = 0
        self.webcam = WebcamWorker(idx)
        self.webcam.start()
        self.cam_status_var.set(f"Opening USB cam {idx}...")

    def stop_webcam(self):
        if self.webcam is not None:
            self.webcam.stop()
            self.webcam = None
        self.latest_capture_time = None
        self.detector_a = 0
        self.detector_b = 0
        self.cam_status_var.set("USB Cam: stopped")
        self.cam_preview.configure(image="", text="USB cam stopped")
        self.redraw()

    def _webcam_fresh(self) -> bool:
        return (
            self.webcam is not None
            and self.webcam.is_alive
            and self.latest_capture_time is not None
            and (time.perf_counter() - self.latest_capture_time) < 1.0
        )

    def _process_webcam_frame(self, capture_time: float, frame):
        self.latest_capture_time = capture_time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        def slice_roi(box):
            x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
            return x1, y1, x2, y2, gray[y1:y2, x1:x2]

        ax1, ay1, ax2, ay2, reg_a = slice_roi(self.ROI_A)
        bx1, by1, bx2, by2, reg_b = slice_roi(self.ROI_B)

        self.brightness_a = float(reg_a.mean()) if reg_a.size else 0.0
        self.brightness_b = float(reg_b.mean()) if reg_b.size else 0.0

        thresh = float(self.cam_thresh_var.get())
        self.detector_a = int(self.brightness_a >= thresh)
        self.detector_b = int(self.brightness_b >= thresh)

        vis = frame.copy()
        for label, (x1, y1, x2, y2), state, bright in [
            ("A", (ax1, ay1, ax2, ay2), self.detector_a, self.brightness_a),
            ("B", (bx1, by1, bx2, by2), self.detector_b, self.brightness_b),
        ]:
            col = (40, 210, 40) if state else (50, 80, 230)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(
                vis, f"{label}: {bright:.0f}->{state}", (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2
            )

        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pw = max(200, self.cam_preview.winfo_width())
        ph = max(160, self.cam_preview.winfo_height())
        pil_img.thumbnail((pw, ph))
        self.preview_image = ImageTk.PhotoImage(pil_img)
        self.cam_preview.configure(image=self.preview_image, text="")
        self.cam_status_var.set(f"USB Cam live (thresh {thresh:.0f})")

    def _poll_webcam(self):
        if self.webcam is not None:
            latest = self.webcam.latest_frame()
            if latest is not None:
                self._process_webcam_frame(*latest)
                self._loader_webcam_sample()
            if not self.webcam.is_alive:
                err = self.webcam.error
                self.webcam = None
                self.latest_capture_time = None
                self.detector_a = 0
                self.detector_b = 0
                self.cam_status_var.set(f"USB Cam error: {err}" if err else "USB Cam: stopped")
        self.cam_after_id = self.after(30, self._poll_webcam)

    # ---------------------------------------------------------
    # Simulation Controls
    # ---------------------------------------------------------

    def _parse_config(self) -> dict:
        cfg = {
            "duration": float(self.duration_var.get()),
            "speed": float(self.speed_var.get()),
            "seed": int(self.seed_var.get()),
            "q_len": int(self.queue_len_var.get()),
            "laser_mw": float(self.laser_mw_var.get()),
            "er_db": float(self.er_db_var.get()),
            "split_a": float(self.split_a_var.get()),
            "loss_db": float(self.loss_db_var.get()),
            "resp": float(self.resp_var.get()),
            "tia_gain": float(self.tia_gain_var.get()),
            "rin_std": float(self.rin_std_var.get()),
            "stream_noise": float(self.stream_noise_var.get()),
        }

        if not math.isfinite(cfg["duration"]) or not (0 < cfg["duration"] <= 120):
            raise ValueError("Probe duration must be > 0 and <= 120 s.")
        if not math.isfinite(cfg["speed"]) or not (0.01 <= cfg["speed"] <= 10000):
            raise ValueError("Playback speed must be in [0.01, 100].")
        if not (1 <= cfg["q_len"] <= 4096):
            raise ValueError("Queue length must be between 1 and 4096 steps.")
        if not math.isfinite(cfg["laser_mw"]) or cfg["laser_mw"] < 0:
            raise ValueError("Laser power must be >= 0 mW.")
        if not (0 <= cfg["split_a"] <= 1):
            raise ValueError("Split ratio A must be in [0, 1].")
        if not math.isfinite(cfg["resp"]) or cfg["resp"] <= 0:
            raise ValueError("Responsivity must be > 0 A/W.")
        if not math.isfinite(cfg["tia_gain"]) or cfg["tia_gain"] < 0:
            raise ValueError("TIA gain must be >= 0 V/mA.")
        if not math.isfinite(cfg["rin_std"]) or cfg["rin_std"] < 0:
            raise ValueError("RIN sigma must be >= 0.")
        if not (0 <= cfg["stream_noise"] <= 1):
            raise ValueError("Stream flip probability must be in [0, 1].")
        return cfg

    def _sync_live_vars(self, _event=None):
        if self.model is not None:
            self.model.mode = self.mode_var.get()
            self.model.manual_x = bool(self.manual_x_var.get())
            self.model.ignore_startup = bool(self.ignore_startup_var.get())
            self.model.feed_select_to_driver_a = bool(self.feedback_driver_a_var.get())
            if self.model.gen is not None:
                self.model.gen.feedback = bool(self.gen_fb_var.get())
                try:
                    t = float(self.temp_var.get())
                    if t > 0:
                        self.model.gen.temperature = t
                except (ValueError, tk.TclError):
                    pass
            try:
                q_len = int(self.queue_len_var.get())
                if 1 <= q_len <= 4096 and q_len != self.model.d_queue_length:
                    self.model.set_queue_length(q_len)
            except (ValueError, tk.TclError):
                pass
        self.redraw()

    def _step_once(self) -> bool:
        assert self.model is not None
        self._sync_live_vars()
        fresh = self._webcam_fresh()
        advanced = self.model.step(
            cam_a=self.detector_a if fresh else None,
            cam_b=self.detector_b if fresh else None,
            brightness_a=self.brightness_a,
            brightness_b=self.brightness_b,
            threshold_px=float(self.cam_thresh_var.get()),
        )
        if advanced and self.model.records and self.model.records[-1].get("rising_edge") and self.model.q:
            self.llm_request_from_q()
        return advanced

    def pause(self):
        self.running = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None
        self.run_btn.configure(text="Run")

    def reset_simulation(self):
        self.pause()
        try:
            cfg = self._parse_config()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        try:
            ngram = self._ensure_ngram()
            gen_temp = float(self.temp_var.get())
            if gen_temp <= 0:
                raise ValueError("Temperature must be > 0.")
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid generator settings", str(exc))
            return

        self.playback_speed = cfg["speed"]
        self.lag_result = None
        self.lag_info_var.set("Lag sweep: run the simulation, then press Analyse lags.")
        self.model = PhotonicBenchModel(
            probe_duration=cfg["duration"],
            seed=cfg["seed"],
            mode=self.mode_var.get(),
            ignore_startup=self.ignore_startup_var.get(),
            manual_x=self.manual_x_var.get(),
            feed_select_to_driver_a=self.feedback_driver_a_var.get(),
            d_queue_length=cfg["q_len"],
            laser_power_mw=cfg["laser_mw"],
            extinction_db=cfg["er_db"],
            split_ratio_a=cfg["split_a"],
            insertion_loss_db=cfg["loss_db"],
            responsivity_a_per_w=cfg["resp"],
            tia_gain_v_per_ma=cfg["tia_gain"],
            optical_noise_std=cfg["rin_std"],
            pattern_bits=self.pattern_var.get(),
            stream_text=self.text_var.get(),
            stream_noise=cfg["stream_noise"],
            ngram_model=ngram,
            gen_temperature=gen_temp,
            gen_feedback=self.gen_fb_var.get(),
        )
        self.llm_update_prompt_bits()
        self.model.prompt_bits = list(self.prompt_bits)
        self.model.prompt_bit_index = self.prompt_bit_index
        self.redraw()

    def toggle_running(self):
        if self.model is None:
            return
        if self.running:
            self.pause()
            self.redraw()
            return
        if self.model.finished:
            self.status_var.set("Simulation finished. Press Reset to run again.")
            return
        if self.mode_var.get() == "USB Webcam" and not self._webcam_fresh():
            messagebox.showwarning("USB Camera not ready", "Start the USB webcam first or choose another input mode.")
            return
        try:
            self.playback_speed = self._parse_config()["speed"]
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.running = True
        self.run_btn.configure(text="Pause")
        self._tick()

    def _tick(self):
        self.after_id = None
        if not self.running or self.model is None:
            return
        try:
            self.playback_speed = float(self.speed_var.get())
        except (ValueError, tk.TclError):
            pass

        self.llm_poll()
        self._step_once()
        self.redraw()

        if self.model.finished:
            self.pause()
            self.analyse_lags()
            return

        delay_ms = max(1, round(1000 * self.model.DT / max(0.01, self.playback_speed)))
        self.after_id = self.after(delay_ms, self._tick)

    def single_step(self):
        if self.model is None:
            return
        self.pause()
        self.llm_poll()
        self._step_once()
        self.redraw()

    def run_to_end(self):
        if self.model is None:
            return
        self.pause()
        while not self.model.finished:
            self._step_once()
        self.redraw()
        self.analyse_lags()

    # ---------------------------------------------------------
    # Lag sweep
    # ---------------------------------------------------------

    def analyse_lags(self):
        m = self.model
        if m is None or not m.records:
            self.lag_info_var.set("Lag sweep: no data yet. Run or step the simulation first.")
            return
        try:
            max_lag = int(self.max_lag_var.get())
            if max_lag < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid max lag", "Max lag must be a positive integer.")
            return

        key = self.LAG_SIGNALS.get(self.lag_signal_var.get(), "optical_x")
        bits = m.symbol_samples(key)
        res = lag_sweep(bits, max_lag)
        res["key"] = key
        self.lag_result = res

        step_per_sym = m.input_steps
        if not res["rows"]:
            self.lag_info_var.set(f"Lag sweep: only {res['n']} symbols so far, need at least 6.")
        else:
            pk = res["peak"]
            msg = (
                f"{self.lag_signal_var.get()}: n={res['n']} symbols | chance baseline={res['baseline']:.3f} | "
                f"significance line (z=4)={res['sig_rate']:.3f} | "
                f"peak lag {pk[0]} ({pk[0] * step_per_sym} steps, {pk[0] * m.INPUT_PERIOD * 1000:.0f} ms) "
                f"rate={pk[1]:.3f} z={pk[2]:.1f}"
            )
            per = res["period"]
            if per is not None:
                msg += (
                    f" | DETECTED REPEAT PERIOD = {per[0]} symbols "
                    f"(set Queue len to {per[0] * step_per_sym} steps to align the hardware probe)"
                )
                if m.mode == "Generated text":
                    msg += f" = {per[0] / TOKEN_BITS:.2f} tokens of {TOKEN_BITS} bits each"
            else:
                msg += " | no significant repeat detected"
            self.lag_info_var.set(msg)
        self.redraw()

    def _draw_lag_chart(self):
        c = self.lag_canvas
        c.delete("all")
        res = self.lag_result
        w = max(300, c.winfo_width())
        h = max(120, c.winfo_height())
        left, right, top, bottom = 50, w - 15, 14, h - 26
        plot_w = max(10, right - left)
        plot_h = max(10, bottom - top)

        c.create_rectangle(left, top, right, bottom, outline="#d4dde5")
        for frac in (0.0, 0.5, 1.0):
            y = bottom - plot_h * frac
            c.create_text(left - 6, y, text=f"{frac:.1f}", anchor="e", font=("Arial", 8), fill="#34495e")

        if not res or not res["rows"]:
            c.create_text(w / 2, h / 2, text="No lag sweep yet", fill="#82919e")
            return

        rows = res["rows"]
        n_bars = len(rows)
        bar_w = plot_w / n_bars
        per_lag = res["period"][0] if res["period"] else None

        for i, (lag, rate, z) in enumerate(rows):
            x0 = left + i * bar_w
            y0 = bottom - plot_h * rate
            sig = z >= 4.0
            color = "#bd4258" if sig else "#9fb3c2"
            if per_lag is not None and lag == per_lag:
                color = "#12805c"
            c.create_rectangle(x0 + 0.5, y0, x0 + max(1.0, bar_w - 0.5), bottom, fill=color, outline="")

        base_y = bottom - plot_h * res["baseline"]
        sig_y = bottom - plot_h * res["sig_rate"]
        c.create_line(left, base_y, right, base_y, fill="#34495e", dash=(4, 3))
        c.create_text(right - 2, base_y - 7, text="chance", anchor="e", font=("Arial", 8), fill="#34495e")
        c.create_line(left, sig_y, right, sig_y, fill="#b36a20", dash=(2, 3))
        c.create_text(left + 4, sig_y - 7, text="z = 4", anchor="w", font=("Arial", 8), fill="#b36a20")

        tick_every = max(1, n_bars // 16)
        for i, (lag, _r, _z) in enumerate(rows):
            if lag == 1 or lag % tick_every == 0:
                x = left + (i + 0.5) * bar_w
                c.create_text(x, bottom + 11, text=str(lag), font=("Arial", 8), fill="#34495e")
        c.create_text(right, h - 3, text="lag (symbols)", anchor="e", font=("Arial", 8), fill="#34495e")

    # ---------------------------------------------------------
    # Drawing
    # ---------------------------------------------------------

    def _schedule_redraw(self, _event=None):
        if self.resize_after_id is not None:
            self.after_cancel(self.resize_after_id)
        self.resize_after_id = self.after(25, self.redraw)

    def redraw(self):
        self.resize_after_id = None
        if self.model is None:
            return
        self._draw_circuit()
        self._draw_waveforms()
        self._draw_lag_chart()
        self._draw_queue_canvas()
        self._update_text_panel()
        self._update_status()

    def _section(self, x, y, w, h, title):
        c = self.canvas
        c.create_rectangle(x, y, x + w, y + h, outline="#d4dde5", dash=(4, 3))
        c.create_text(x + 12, y + 13, text=title, anchor="w", font=("Arial", 10, "bold"), fill="#34495e")

    def _box(self, x, y, w, h, text, value=None):
        fill = "#f5f7fa" if value is None else ("#d8f3e6" if value else "#edf0f4")
        self.canvas.create_rectangle(x, y, x + w, y + h, fill=fill, outline="#aebdc9")
        self.canvas.create_text(x + w / 2, y + h / 2, text=text, font=("Arial", 10), fill="#203040", justify="center")

    def _wire(self, points, value=None, dash=None):
        color = "#12805c" if value else "#82919e"
        kw = {"arrow": tk.LAST, "fill": color, "width": 2}
        if dash is not None:
            kw["dash"] = dash
        self.canvas.create_line(*points, **kw)

    def _gate(self, x, y, name, value):
        c = self.canvas
        fill = "#d8f3e6" if value else "#edf0f4"
        c.create_polygon(
            x, y, x + 62, y, x + 85, y + 23, x + 62, y + 46, x, y + 46,
            fill=fill, outline="#aebdc9"
        )
        c.create_text(x + 37, y + 23, text=name, font=("Arial", 10, "bold"), fill="#203040")

    def _format_queue_preview(self, m: PhotonicBenchModel) -> str:
        bits = "".join(str(bit) for bit in m.d_bit_queue)
        if len(bits) > 12:
            bits = bits[:6] + ".." + bits[-4:]
        return f"[{bits}] ({len(m.d_bit_queue)}/{m.d_queue_length})"

    def _draw_circuit(self):
        m = self.model
        assert m is not None
        c = self.canvas
        c.delete("all")

        self._section(10, 10, 1140, 125, f"PHOTONIC BENCH — mode: {m.mode}")
        self._section(10, 145, 1140, 115, "PARALLEL DRIVERS (SELECT -> Driver A) / TIA + RC THRESHOLD")
        self._section(10, 270, 1140, 200, "MAIN CIRCUIT — pre-delay AND -> 150 ms delay + queue -> 74HC74")
        self._section(10, 480, 1140, 100, "PROBES — legacy: queue vs pre-delay AND | pattern: queue D_q vs D now (same signal, lagged)")

        self._box(880, 40, 220, 45, f"Laser ({m.laser_power_mw:g} mW)\nTx = {m.optical_tx_mw:.2f} mW", m.x)
        self._box(590, 40, 220, 45, f"Coupler / x={m.x}\nphi={m.optical_phase_rad:.2f} rad", m.x)
        self._wire([880, 62, 810, 62], m.x)

        self._box(
            40, 80, 220, 40,
            f"PD A / Cam A ({self.brightness_a:.0f})\n{m.optical_a_mw:.2f} mW | {m.photocurrent_a_ua:.1f} uA",
            m.driver_a_raw,
        )
        self._box(
            310, 80, 220, 40,
            f"PD B / Cam B ({self.brightness_b:.0f})\n{m.optical_b_mw:.2f} mW | {m.photocurrent_b_ua:.1f} uA",
            m.x,
        )
        self._wire([700, 85, 700, 100, 530, 100], m.x)
        self._wire([590, 62, 150, 62, 150, 80], m.x)

        fb_str = f"Q={m.q}" if m.feed_select_to_driver_a else "OFF"
        self._box(
            40, 182, 225, 56,
            f"Driver A (SEL fb: {fb_str})\nraw={m.driver_a_raw}, sel_gate={m.driver_a_select_gate} -> a={m.a}",
            m.a,
        )
        self._box(295, 182, 190, 56, f"Driver B / x\nx = {m.x}", m.x)
        self._box(515, 182, 210, 56, f"RC (tau={m.RC_TAU*1000:.0f}ms)\nVc = {m.rc_voltage:.3f} V")
        self._box(760, 182, 260, 56, f"Comparator > {m.THRESHOLD} V\nb = {m.b}", m.b)

        self._wire([150, 120, 150, 182], m.driver_a_raw)
        self._wire([420, 120, 420, 182], m.x)
        self._wire([485, 210, 515, 210], m.x)
        self._wire([725, 210, 760, 210], m.b)

        self._wire(
            [1065, 370, 1065, 276, 22, 276, 22, 210, 40, 210],
            m.q if m.feed_select_to_driver_a else 0,
            dash=None if m.feed_select_to_driver_a else (5, 4),
        )

        c.create_text(35, 328, text=f"a={m.a}\nb={m.b}", anchor="w", font=("Arial", 11))
        self._gate(100, 308, "AND", m.pre_delay_and)
        self._wire([68, 323, 100, 323], m.a)
        self._wire([68, 340, 100, 340], m.b)

        self._box(
            225, 308, 185, 46,
            f"Delay {m.AND_DELAY*1000:.0f} ms ({m.optical_delayed_mw:.2f}mW)\nout = {m.delayed_and}",
            m.delayed_and,
        )
        self._wire([185, 331, 225, 331], m.pre_delay_and)

        self._box(450, 308, 180, 46, f"Running AND\nval = {m.accumulator}", m.accumulator)
        self._wire([410, 323, 450, 323], m.delayed_and)

        c.create_text(35, 412, text=f"a={m.a}\nb={m.b}", anchor="w", font=("Arial", 11))
        self._gate(100, 390, "OR", m.y)
        self._wire([68, 405, 100, 405], m.a)
        self._wire([68, 422, 100, 422], m.b)

        self._gate(245, 390, "OR", m.d)
        self._wire([185, 413, 245, 413], m.y)
        self._wire([410, 339, 430, 339, 430, 400, 245, 400], m.delayed_and)

        self._box(
            355, 382, 195, 62,
            f"Queue (in D={m.d})\n{self._format_queue_preview(m)}\nout={m.d_queued} (lag {m.last_lag_steps}st)",
            m.d_queued,
        )
        self._wire([330, 413, 355, 413], m.d)

        self._box(585, 378, 175, 70, f"74HC74 D-FF\nDq={m.d_queued}  Q={m.q}\nClk: {1/m.CLOCK_PERIOD:g} Hz", m.q)
        self._wire([550, 413, 585, 413], m.d_queued)

        self._box(1000, 370, 130, 30, f"SELECT = {m.q}", m.q)
        self._box(1000, 420, 130, 30, f"Observe = {m.not_q}", m.not_q)
        self._wire([760, 395, 1000, 385], m.q)
        self._wire([760, 425, 1000, 435], m.not_q)

        active = m.step_index < m.probe_steps
        p_state = "ACTIVE" if active else "STOPPED"
        self._box(
            55, 515, 225, 45,
            f"Pre-Delay AND = {m.pre_delay_and}\nQueue out D_q = {m.d_queued}",
            m.pre_delay_and,
        )
        self._box(
            310, 515, 170, 45,
            f"D now = {m.d}\nlag = {m.last_lag_steps} steps",
            m.d,
        )
        self._wire([280, 538, 310, 538], m.pre_delay_and)

        match_str = "-" if m.last_match is None else ("MATCH" if m.last_match else "DIFF")
        self._box(
            510, 515, 310, 45,
            f"Legacy ({p_state}): D_q={m.d_queued} vs pre_AND={m.pre_delay_and}\n"
            f"state = {match_str} | post_Q_AND={m.post_queue_and}",
        )
        self._wire([480, 538, 510, 538])

        self_pct = (m.self_lag_matches / m.self_lag_count) if m.self_lag_count else 0.0
        self._box(
            850, 515, 255, 45,
            f"AND-probe: {m.probe_matches}/{m.probe_count}\n"
            f"Self-lag D_q==D: {m.self_lag_matches}/{m.self_lag_count} ({self_pct:.1%})",
            m.last_self_match,
        )
        self._wire([820, 538, 850, 538])

        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())
        c.scale("all", 0, 0, w / 1160, h / 590)

    def _draw_waveforms(self):
        m = self.model
        assert m is not None
        p = self.waveform
        p.delete("all")

        w = max(600, p.winfo_width())
        left = 115
        right = w - 25
        plot_w = max(10, right - left)

        signals = [
            ("x (input)", "optical_x", "#3b7d3a"),
            ("pre_del AND", "pre_delay_and", "#1976a3"),
            ("D (OR out)", "D_raw_or", "#b36a20"),
            ("D_queued", "D_queued", "#7753a6"),
            ("Q / SELECT", "Q_SELECT", "#bd4258"),
        ]

        stride = max(1, len(m.records) // 1800)
        visible = m.records[::stride]
        if m.records and (not visible or visible[-1] is not m.records[-1]):
            visible.append(m.records[-1])

        total_dur = max(m.DT, m.total_duration)

        for idx, (label, key, color) in enumerate(signals):
            base_y = 22 + idx * 28
            p.create_text(8, base_y, text=label, anchor="w", fill="#34495e")
            p.create_line(left, base_y + 7, right, base_y + 7, fill="#dce4eb")

            points = [left, base_y + 7]
            prev_y = base_y + 7
            for rec in visible:
                x = left + plot_w * (rec["time_s"] / total_dur)
                y = base_y + 7 - 14 * rec[key]
                points.extend([x, prev_y, x, y])
                prev_y = y

            if len(points) >= 4:
                p.create_line(*points, fill=color, width=2)

        probe_x = left + plot_w * (m.probe_duration / total_dur)
        p.create_line(probe_x, 8, probe_x, 160, fill="#888888", dash=(4, 3))
        p.create_text(left, 172, text="0 s", anchor="w")
        p.create_text(probe_x, 172, text=f"Probe stops: {m.probe_duration:g} s")
        p.create_text(right, 172, text=f"{m.total_duration:g} s", anchor="e")

    def _update_status(self):
        m = self.model
        assert m is not None

        state = "Finished" if m.finished else ("Running" if self.running else "Paused")
        fb_str = f"ON ({m.driver_a_select_gate})" if m.feed_select_to_driver_a else "OFF"
        self.status_var.set(
            f"{state}  |  Time: {m.time:.3f} / {m.total_duration:g} s  |  "
            f"dt: {m.DT * 1000:g} ms  |  Mode: {m.mode}  |  SELECT->DriverA: {fb_str}  |  "
            f"Stream flip p: {m.stream_noise:g}"
        )
        self.photonic_var.set(
            f"Tx: {m.optical_tx_mw:.2f} mW ({mw_to_dbm(m.optical_tx_mw):.1f} dBm)  |  "
            f"PD-A: {m.photocurrent_a_ua:.1f} uA (CamA={self.brightness_a:.0f})  |  "
            f"PD-B: {m.photocurrent_b_ua:.1f} uA (CamB={self.brightness_b:.0f})"
        )

        match_pct = (m.probe_matches / m.probe_count) if m.probe_count else 0.0
        self_pct = (m.self_lag_matches / m.self_lag_count) if m.self_lag_count else 0.0
        self.stats_var.set(
            f"D_queued={m.d_queued}, D={m.d}, Q={m.q}  |  "
            f"Queue lag {m.last_lag_steps} steps ({m.last_lag_steps * m.DT * 1000:.0f} ms)  |  "
            f"Self-lag D_q==D: {m.self_lag_matches}/{m.self_lag_count} ({self_pct:.1%})  |  "
            f"Legacy queue==pre-AND: {m.probe_matches}/{m.probe_count} ({match_pct:.1%})"
        )

    # ---------------------------------------------------------
    # Trigram-index loader (optical link -> trigram queue)
    # ---------------------------------------------------------

    def _bit_period_ms(self) -> int:
        try:
            return min(2000, max(60, int(self.bit_ms_var.get())))
        except (ValueError, tk.TclError):
            return 120

    def _invalidate_queue(self):
        self._abort_load(silent=True)
        self.queue_ngram = None
        self.codec = None
        self.tqueue = None
        self.receiver = None
        self.codec_info_var.set("")
        self.load_status_var.set("Trigram queue: empty (dataset changed - press Build / Load into queue)")
        if hasattr(self, "load_progress"):
            self.load_progress["value"] = 0
        self._draw_queue_canvas()

    def build_trigram_bits(self) -> bool:
        try:
            max_tokens = int(self.max_tokens_var.get())
            if not (12 <= max_tokens <= 20000):
                raise ValueError("Max tokens must be between 12 and 20000.")
            self.codec = TrigramCodec(self.corpus_text, max_tokens)
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Trigram bits", str(exc))
            return False
        c = self.codec
        frame = len(c.encode_frame())
        ms = self._bit_period_ms()
        self.codec_info_var.set(
            f"{len(c.tokens)} tokens -> {len(c.sequence)} trigram words ({len(c.tri_list)} unique) | "
            f"index {c.index_bits} bits + 1 parity | frame {frame} bits | "
            f"webcam time ~ {frame * ms / 1000:.0f} s at {ms} ms/bit"
        )
        return True

    def start_load(self):
        if self.loader_active:
            return
        self.pause()
        if not self.build_trigram_bits():
            return
        c = self.codec
        assert c is not None
        frame = c.encode_frame()
        self.tqueue = TrigramQueue(len(c.sequence) * c.word_bits)
        self.receiver = FrameReceiver(c, self.tqueue)
        self.sampler = ClockedSampler()
        self.queue_ngram = None
        try:
            seed = int(self.seed_var.get())
        except (ValueError, tk.TclError):
            seed = 7
        rng = random.Random(seed)

        if self.link_source_var.get().startswith("Simulated"):
            try:
                cfg = self._parse_config()
                ber = float(self.inject_ber_var.get())
                if not (0.0 <= ber <= 0.5):
                    raise ValueError("Inject BER must be in [0, 0.5].")
            except (ValueError, tk.TclError) as exc:
                messagebox.showerror("Invalid settings", str(exc))
                return
            self.link_sim = OpticalLinkSim(
                cfg["laser_mw"], cfg["er_db"], cfg["split_a"], cfg["loss_db"], cfg["resp"],
                cfg["tia_gain"], cfg["rin_std"], seed, symbol_steps=40, inject_ber=ber,
            )
            junk = [rng.randrange(2) for _ in range(rng.randrange(8, 40))]
            self.sim_tx_bits = junk + frame
            self.sim_tx_pos = 0
            self.loader_active = True
            self.load_status_var.set("Sending over simulated photonic link...")
            self._sim_load_tick()
        else:
            if not self._webcam_fresh():
                messagebox.showwarning(
                    "USB Camera not ready",
                    "Start the USB webcam first. Aim it so the Tx window's DATA square is in the left ROI (B) "
                    "and its CLOCK square is in the right ROI (A).",
                )
                return
            self.open_tx_window()
            self.tx_bits = [1, 0] * 12 + frame
            self.loader_active = True
            self.load_status_var.set("Webcam link: transmitting (hunting for SYNC)...")
            self._tx_start()
        self._update_loader_ui()

    def _sim_load_tick(self):
        self.loader_after_id = None
        rx = self.receiver
        if not self.loader_active or rx is None or self.link_sim is None:
            return
        for _ in range(100):
            if rx.done or self.sim_tx_pos >= len(self.sim_tx_bits):
                break
            bit = self.sim_tx_bits[self.sim_tx_pos]
            self.sim_tx_pos += 1
            rx.feed(self.link_sim.send_bit(bit))
        self._update_loader_ui()
        if rx.done:
            self._finish_load()
        elif self.sim_tx_pos >= len(self.sim_tx_bits):
            self._abort_load(msg="Transmission ended before a full frame was received (sync lost or header corrupted).")
        else:
            self.loader_after_id = self.after(1, self._sim_load_tick)

    def _loader_webcam_sample(self):
        if (
            not self.loader_active
            or self.receiver is None
            or not self.link_source_var.get().startswith("Webcam")
            or not self._webcam_fresh()
        ):
            return
        bit = self.sampler.update(self.detector_a, self.detector_b)  # A = CLOCK, B = DATA
        if bit is None:
            return
        self.receiver.feed(bit)
        self._update_loader_ui()
        if self.receiver.done:
            self._finish_load()

    def _update_loader_ui(self):
        rx = self.receiver
        if rx is None:
            return
        self.load_progress["value"] = 100.0 * rx.progress
        self.load_status_var.set(
            f"{FrameReceiver.STATE_NAMES[rx.state]} | bits seen {rx.bits_seen} | queue "
            f"{rx.payload}/{rx.expected} bits | sync hits {rx.sync_hits} (false {rx.false_syncs})"
        )
        self._draw_queue_canvas()

    def _finish_load(self):
        self._stop_transmit()
        self.loader_active = False
        c, q = self.codec, self.tqueue
        if c is None or q is None:
            return
        seq = decode_queue(c, q)
        truth = c.ground_truth()
        rejected = sum(1 for s in seq if s is None)
        wrong = sum(1 for got, exp in zip(seq, truth) if got is not None and got != exp)
        try:
            model = NGramModel.from_trigrams(seq)
        except ValueError as exc:
            self.load_status_var.set(f"Queue received but model build failed: {exc}")
            messagebox.showerror("Trigram queue", str(exc))
            return
        self.queue_ngram = model
        self.load_progress["value"] = 100
        self.load_status_var.set(
            f"Queue loaded: {len(seq)} words | {rejected} rejected by parity | {wrong} undetected wrong | "
            f"model: {len(model.unigram)} words, {len(model.tokens)} tokens"
        )
        self._draw_queue_canvas()
        if self.use_queue_var.get():
            self.mode_var.set("Generated text")
            self.reset_simulation()

    def _abort_load(self, silent: bool = False, msg: str | None = None):
        was_active = self.loader_active
        self.loader_active = False
        if self.loader_after_id is not None:
            self.after_cancel(self.loader_after_id)
            self.loader_after_id = None
        self._stop_transmit()
        if not silent and (was_active or msg):
            self.load_status_var.set(msg or "Load aborted.")

    # ---- optical Tx window (webcam link) ----

    def open_tx_window(self):
        if self.tx_window is not None:
            try:
                if self.tx_window.winfo_exists():
                    self.tx_window.lift()
                    return
            except tk.TclError:
                pass
        win = tk.Toplevel(self)
        win.title("Optical Tx - aim the webcam here (DATA left = ROI B, CLOCK right = ROI A)")
        win.geometry("820x440")
        win.configure(background="black")
        cv = tk.Canvas(win, background="black", highlightthickness=0)
        cv.pack(fill="both", expand=True)
        cv.bind("<Configure>", lambda _e: self._tx_layout())
        win.protocol("WM_DELETE_WINDOW", lambda: (self._abort_load(), win.destroy()))
        self.tx_window = win
        self.tx_canvas = cv
        self.tx_data_on = 0
        self.tx_clock_on = 0
        self._tx_layout()

    def _tx_layout(self):
        cv = self.tx_canvas
        if cv is None:
            return
        try:
            cv.delete("all")
            w = max(100, cv.winfo_width())
            h = max(100, cv.winfo_height())
        except tk.TclError:
            return
        for label, box, on in (
            ("DATA", self.ROI_B, self.tx_data_on),
            ("CLOCK", self.ROI_A, self.tx_clock_on),
        ):
            x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
            cv.create_rectangle(x1, y1, x2, y2, fill="white" if on else "black", outline="#555555", width=2)
            cv.create_text((x1 + x2) / 2, y1 - 14, text=label, fill="#888888", font=("Arial", 11, "bold"))

    def _tx_schedule(self, when: float, fn):
        delay = max(1, int((when - time.perf_counter()) * 1000))
        self.tx_after_id = self.after(delay, fn)

    def _tx_start(self):
        self.tx_k = 0
        self.tx_period = self._bit_period_ms() / 1000.0
        self.tx_t0 = time.perf_counter() + 1.0
        self.tx_data_on = 0
        self.tx_clock_on = 0
        self._tx_layout()
        self._tx_schedule(self.tx_t0, self._tx_data_event)

    def _tx_data_event(self):
        self.tx_after_id = None
        if not self.loader_active or not self.tx_bits:
            return
        self.tx_data_on = self.tx_bits[self.tx_k % len(self.tx_bits)]
        self._tx_layout()
        # CLOCK edge half a bit later: DATA has been stable for half a bit period when it is sampled
        self._tx_schedule(self.tx_t0 + (self.tx_k + 0.5) * self.tx_period, self._tx_clock_event)

    def _tx_clock_event(self):
        self.tx_after_id = None
        if not self.loader_active:
            return
        self.tx_clock_on ^= 1
        self._tx_layout()
        self.tx_k += 1
        self._tx_schedule(self.tx_t0 + self.tx_k * self.tx_period, self._tx_data_event)

    def _stop_transmit(self):
        if self.tx_after_id is not None:
            try:
                self.after_cancel(self.tx_after_id)
            except tk.TclError:
                pass
            self.tx_after_id = None
        self.tx_data_on = 0
        self.tx_clock_on = 0
        self._tx_layout()

    def _draw_queue_canvas(self):
        c = getattr(self, "queue_canvas", None)
        if c is None:
            return
        c.delete("all")
        w = max(300, c.winfo_width())
        q = self.tqueue
        if q is None:
            c.create_text(8, 40, text="Trigram queue: empty", anchor="w", fill="#82919e")
            return
        n = len(q.bits)
        cap = q.capacity
        c.create_rectangle(8, 8, w - 8, 24, outline="#aebdc9")
        c.create_rectangle(8, 8, 8 + (w - 16) * n / cap, 24, fill="#12805c", outline="")
        c.create_text(w / 2, 16, text=f"Trigram queue {n}/{cap} bits ({n / cap:.0%})", font=("Arial", 9))
        cells = max(1, min(n, (w - 16) // 9))
        for i, b in enumerate(q.tail(cells)):
            x = 8 + i * 9
            c.create_rectangle(x, 32, x + 7, 50, fill="#203040" if b else "#edf0f4", outline="#aebdc9")
        wb = self.codec.word_bits if self.codec else 0
        c.create_text(
            8, 64, anchor="w", font=("Arial", 8), fill="#34495e",
            text=f"newest queue bits (dark = 1) | word = {wb} bits: [{wb - 1}-bit trigram index | even parity]",
        )

    # ---------------------------------------------------------
    # Dataset + generated text
    # ---------------------------------------------------------

    def _read_corpus(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            if path.lower().endswith(".csv"):
                parts = []
                for row in csv.reader(fh):
                    for cell in row:
                        cell = cell.strip()
                        if cell:
                            parts.append(cell if cell[-1] in ".!?" else cell + ".")
                text = "\n".join(parts)
            else:
                text = fh.read(2_000_000)
        return text[:2_000_000]

    def _ensure_ngram(self) -> NGramModel:
        if self.use_queue_var.get() and self.queue_ngram is not None:
            qm = self.queue_ngram
            self.corpus_info_var.set(
                f"TRIGRAM QUEUE model (received over the optical link) | {len(qm.tokens)} tokens | "
                f"vocab {len(qm.unigram)} | order 2"
            )
            return qm
        order = int(self.order_var.get())
        if not (1 <= order <= 4):
            raise ValueError("N-gram order must be between 1 and 4.")
        key = (self.corpus_name, len(self.corpus_text), order)
        if self.ngram_cache is None or self.ngram_cache[0] != key:
            model = NGramModel(self.corpus_text, order)
            self.ngram_cache = (key, model)
        model = self.ngram_cache[1]
        self.corpus_info_var.set(
            f"Dataset: {self.corpus_name} | {len(model.tokens)} tokens | vocab {len(model.unigram)} | order {model.order}"
        )
        return model

    def load_dataset(self):
        path = filedialog.askopenfilename(
            title="Load text dataset",
            filetypes=[("Text / CSV / Markdown", "*.txt *.csv *.md"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            text = self._read_corpus(path)
            old = (self.corpus_text, self.corpus_name)
            self.corpus_text, self.corpus_name = text, os.path.basename(path)
            try:
                self._ensure_ngram()
            except ValueError:
                self.corpus_text, self.corpus_name = old
                raise
        except (OSError, ValueError, csv.Error) as exc:
            messagebox.showerror("Dataset error", str(exc))
            return
        self._invalidate_queue()
        self.reset_simulation()

    def use_sample_corpus(self):
        self.corpus_text, self.corpus_name = SAMPLE_CORPUS, "built-in sample"
        self._invalidate_queue()
        self.reset_simulation()

    def _update_text_panel(self):
        m = self.model
        gen = m.gen if m is not None else None
        if gen is None:
            return
        n = len(gen.pieces)
        if n < self.shown_pieces:
            self.gen_text.configure(state="normal")
            self.gen_text.delete("1.0", "end")
            self.gen_text.configure(state="disabled")
            self.shown_pieces = 0
        if n > self.shown_pieces:
            self.gen_text.configure(state="normal")
            for piece, tag in gen.pieces[self.shown_pieces:]:
                self.gen_text.insert("end", piece, (tag,) if tag else ())
            self.gen_text.see("end")
            self.gen_text.configure(state="disabled")
            self.shown_pieces = n
        loop = f"ACTIVE (period {gen.loop_period} tok)" if gen.loop_active else "none"
        self.gen_stats_var.set(
            f"{n} tokens ({gen.bits_emitted} bits sent) | loop now: {loop} | flagged tokens: {gen.loop_tokens} | "
            f"loop episodes: {gen.loop_episodes} | forced restarts: {gen.kicks} | "
            f"distinct trigrams: {gen.distinct3():.0%} | feedback {'ON' if gen.feedback else 'OFF'} | T={gen.temperature:.2f}"
        )

    def export_text(self):
        gen = self.model.gen if self.model is not None else None
        if gen is None or not gen.pieces:
            messagebox.showinfo("No text", "Run the simulation in Generated text mode first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export generated text", defaultextension=".txt", filetypes=[("Text files", "*.txt")]
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(gen.text() + "\n")
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))

    def export_csv(self):
        if self.model is None or not self.model.records:
            messagebox.showinfo("No data", "Run the simulation or step first.")
            return

        path = filedialog.asksaveasfilename(
            title="Export photonic simulation data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(self.model.records[0].keys()))
                writer.writeheader()
                writer.writerows(self.model.records)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return

        messagebox.showinfo("Export complete", f"Exported {len(self.model.records)} samples.")

    def close(self):
        self.closing = True
        if getattr(self, "llm_poll_after", None) is not None:
            try:
                self.after_cancel(self.llm_poll_after)
            except tk.TclError:
                pass
        try:
            self.llm_decoder.shutdown.set()
        except Exception:
            pass
        self._abort_load(silent=True)
        self.pause()
        self._unbind_mousewheel()
        if self.resize_after_id is not None:
            self.after_cancel(self.resize_after_id)
        if self.cam_after_id is not None:
            self.after_cancel(self.cam_after_id)
        if self.webcam is not None:
            self.webcam.stop()
        self.destroy()


if __name__ == "__main__":
    app = PhotonicCircuitSimulator()
    app.mainloop()