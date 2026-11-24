import contextlib
import csv
import math
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
from collections import deque
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

LLM link: the prompt is encoded as bits and gates Driver A; each Q-high rising
edge is one token-acceptance credit for a Hugging Face causal LM (KV-cached
decoding).

"""

STREAM_MODES = ("Repeating bits", "Text stream")

# None -> the photonic circuit runs on the same device as the LLM (GPU if present).
# Set to "cpu" only to compare against the GPU path.
CIRCUIT_DEVICE = None


def pick_device() -> "torch.device":
    """Best available accelerator: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(dev) -> str:
    if dev.type == "cuda":
        return f"{dev} ({torch.cuda.get_device_name(dev)})"
    return str(dev)


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
        self.generated_t = None  # same tokens, resident on the model device
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

        device = pick_device()

        # Faster attention + reduced precision on supported hardware.
        load_kwargs = {"trust_remote_code": False}
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            load_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
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
            {"message": f"Model loaded on {describe_device(device)}."
                        + ("" if device.type != "cpu" else " WARNING: no GPU found, running on CPU.")}
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
        self.generated_t = torch.empty(0, dtype=torch.long, device=self.device)
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
            if self.generated_t.numel() > 0:
                unique_ids, count_values = torch.unique(self.generated_t[-64:], return_counts=True)
                selected = adjusted[unique_ids]
                repetition_penalty = 1.16
                adjusted[unique_ids] = torch.where(
                    selected > 0, selected / repetition_penalty, selected * repetition_penalty
                )
                adjusted[unique_ids] -= 0.035 * count_values.to(torch.float32)

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
            position_t = torch.multinomial(probs, num_samples=1)
            token_t = sorted_indices[position_t]  # stays on the device
            token_id = int(token_t.item())        # one sync: Python needs the id for decode/EOS
            mode = ("loop-escape top-p sampling" if loop_info
                    else "Q-accepted top-p sampling")

            self.generated.append(token_id)
            self.generated_t = torch.cat((self.generated_t, token_t.to(torch.long).view(1)))
            self.last_accepted_q_edge = q_edge

            self.next_input = token_t.to(torch.long).view(1, 1)
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
    """
    End-to-end photonic + logic discrete-step simulation engine.

    The circuit state and every per-step computation live on the compute device
    (GPU when available): laser Tx + RIN noise, coupler/photodiodes, TIA, RC node,
    comparator, AND/OR gates, the 150 ms delay line, the D-queue (a ring buffer)
    and the 74HC74 latch (Q). Python only supplies the input symbol, keeps the
    bookkeeping (probes, history, CSV records) and reads one packed result vector
    back per step.
    """

    DT = 0.005
    RC_TAU = 0.050
    SUPPLY = 5.0
    THRESHOLD = 0.7
    AND_DELAY = 0.050
    CLOCK_PERIOD = 0.0000100
    INPUT_PERIOD = 0.0000100

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
        device=None,
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

        self.delay_steps = max(1, int(round(self.AND_DELAY / self.DT)))
        self.clock_steps = max(1, int(round(self.CLOCK_PERIOD / self.DT)))
        self.input_steps = max(1, int(round(self.INPUT_PERIOD / self.DT)))

        self.device = torch.device(device) if device is not None else pick_device()
        # MPS has no float64; everywhere else the analog path is float64 so the
        # RC / comparator threshold behaves exactly like the scalar reference.
        self.fdtype = torch.float32 if self.device.type == "mps" else torch.float64
        # Own CUDA stream: the per-step readback must not wait for the LLM's
        # forward pass running on the default stream in the decoder thread.
        self.stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None

        self.reset()

    def _ctx(self):
        return torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()

    def _scalar(self, value: float):
        return torch.tensor(float(value), dtype=self.fdtype, device=self.device)

    def set_queue_length(self, new_length: int):
        new_length = max(1, int(new_length))
        with self._ctx():
            ordered = torch.roll(self.qbuf, -self.qpos)  # oldest -> newest
            old = ordered.numel()
            if new_length < old:
                ordered = ordered[old - new_length:]  # drop the oldest bits
            elif new_length > old:
                fill = [self.queue_rng.randrange(2) for _ in range(new_length - old)]
                pad = torch.tensor(fill[::-1], dtype=self.fdtype, device=self.device)
                ordered = torch.cat((pad, ordered))   # prepend (oldest side)
            self.qbuf = ordered.contiguous().clone()
            self.qpos = 0
            self.d_queue_length = new_length
            self.d_queued = int(self.qbuf[0].item())

    def reset(self):
        self.input_rng = random.Random(self.seed)
        self.queue_rng = random.Random(self.seed + 3)
        self.stream_rng = random.Random(self.seed + 4)

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

        # ---- device-resident circuit state ----
        f, dev = self.fdtype, self.device
        with self._ctx():
            self.rng_gen = torch.Generator(device=dev)
            self.rng_gen.manual_seed(self.seed + 2)

            er_linear = db_to_linear(self.extinction_db)
            self.low_mw = self.laser_power_mw / max(er_linear, 1.0)
            loss_lin = db_to_linear(-self.insertion_loss_db)
            split_a = min(max(self.split_ratio_a, 0.0), 1.0)
            split_b = 1.0 - split_a
            ua = self.responsivity_a_per_w * 1000.0
            # tx -> [Tx mW, PD-A mW, PD-B mW, PD-A uA, PD-B uA]
            self.chain_coeff = torch.tensor(
                [1.0, split_a * loss_lin, split_b * loss_lin,
                 split_a * loss_lin * ua, split_b * loss_lin * ua],
                dtype=f, device=dev,
            )
            self.rc_decay = math.exp(-self.DT / self.RC_TAU)
            self.delay_gain = db_to_linear(-self.delay_loss_db)

            self.t_zero = self._scalar(0.0)
            self.t_one = self._scalar(1.0)
            self.t_rc = self._scalar(0.0)   # RC capacitor voltage
            self.t_q = self._scalar(0.0)    # 74HC74 Q (SELECT)
            self.and_hist = torch.zeros(self.total_steps, dtype=f, device=dev)
            self.opt_hist = torch.zeros(self.total_steps, dtype=f, device=dev)

            init = [self.queue_rng.randrange(2) for _ in range(self.d_queue_length)]
            self.qbuf = torch.tensor(init, dtype=f, device=dev)  # ring buffer, oldest at qpos
            self.qpos = 0
            self.d_queued = init[0]

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

        self.records: list[dict] = []

    def queue_preview(self) -> str:
        """First 6 / last 4 queue bits (oldest -> newest), gathered on the device."""
        n = self.qbuf.numel()
        if n > 12:
            idx = [(self.qpos + i) % n for i in range(6)] + [(self.qpos + n - 4 + i) % n for i in range(4)]
        else:
            idx = [(self.qpos + i) % n for i in range(n)]
        with self._ctx():
            vals = self.qbuf[torch.tensor(idx, device=self.device)].tolist()
        bits = "".join(str(int(round(v))) for v in vals)
        if n > 12:
            bits = bits[:6] + ".." + bits[6:]
        return f"[{bits}] ({n}/{self.d_queue_length})"

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

        i = self.step_index
        delayed_idx = i - self.delay_steps
        completed = i + 1
        phase = completed % self.clock_steps
        rising_edge = phase == 0
        f, dev = self.fdtype, self.device

        # ---------------- circuit step, entirely on the device ----------------
        with self._ctx():
            inp = torch.tensor([float(self.x), float(self.driver_a_raw)], dtype=f, device=dev)
            x_t, raw_t = inp[0], inp[1]

            gate = self.t_q if self.feed_select_to_driver_a else self.t_one
            a_t = raw_t * gate                                   # Driver A (SELECT-gated)

            base = self.low_mw + (self.laser_power_mw - self.low_mw) * x_t   # laser: ER high/low
            if self.optical_noise_std > 0.0:                                 # RIN
                noise = torch.randn((), generator=self.rng_gen, device=dev, dtype=f)
                base = base * (1.0 + self.optical_noise_std * noise)
            tx_t = torch.clamp(base, min=0.0)

            chain = tx_t * self.chain_coeff                      # coupler + photodiodes
            if self.mode == "Photonic TIA":
                drive = torch.clamp(chain[4] * (self.tia_gain_v_per_ma / 1000.0), max=self.SUPPLY)
            else:
                drive = x_t * self.SUPPLY
            rc_t = drive + (self.t_rc - drive) * self.rc_decay   # RC node
            b_t = (rc_t > self.THRESHOLD).to(f)                  # comparator

            pre_t = a_t * b_t                                    # AND (pre-delay)
            if delayed_idx >= 0:                                 # 150 ms delay line
                delayed_t = self.and_hist[delayed_idx]
                opt_del_t = self.opt_hist[delayed_idx] * self.delay_gain
            else:
                delayed_t = self.t_zero
                opt_del_t = self.t_zero
            self.and_hist[i] = pre_t
            self.opt_hist[i] = tx_t * pre_t

            y_t = torch.maximum(a_t, b_t)                        # OR
            d_t = torch.maximum(delayed_t, y_t)                  # OR with delayed AND

            dq_t = self.qbuf[self.qpos].clone()                  # FIFO queue (ring buffer)
            self.qbuf[self.qpos] = d_t
            post_t = dq_t * b_t
            same_t = (dq_t == d_t).to(f)

            self.t_rc = rc_t
            if rising_edge:                                      # 74HC74: Q <- Dq on clock edge
                self.t_q = dq_t

            out = torch.stack([
                tx_t, chain[1], chain[2], chain[3], chain[4], opt_del_t,
                a_t, rc_t, b_t, pre_t, delayed_t, y_t, d_t, dq_t, post_t, same_t,
            ]).tolist()                                          # single device -> host read
        self.qpos = (self.qpos + 1) % self.qbuf.numel()

        (tx, a_mw, b_mw, iph_a, iph_b, opt_del, a, rc, b, pre, delayed, y, d, dq, post, same_v) = out
        self.optical_tx_mw = tx
        self.optical_a_mw = a_mw
        self.optical_b_mw = b_mw
        self.photocurrent_a_ua = max(0.0, iph_a)
        self.photocurrent_b_ua = max(0.0, iph_b)
        self.optical_delayed_mw = opt_del
        self.optical_phase_rad = (
            self.optical_phase_rad + 2.0 * math.pi * 1.5 * self.DT
        ) % (2.0 * math.pi)
        self.a = int(round(a))
        self.rc_voltage = rc
        self.b = int(round(b))
        self.pre_delay_and = int(round(pre))
        self.truth = self.pre_delay_and
        self.delayed_and = int(round(delayed))
        self.y = int(round(y))
        self.d = int(round(d))
        self.d_queued = int(round(dq))
        self.post_queue_and = int(round(post))
        self.last_lag_steps = max(0, self.qbuf.numel() - 1)

        # Like-with-like probe: lagged D vs D now, only once real data has propagated
        self_cell: int | str = ""
        if self.step_index >= self.d_queue_length:
            same = int(round(same_v))
            self.last_self_match = same
            self.self_lag_count += 1
            self.self_lag_matches += same
            self_cell = same

        self.time = completed * self.DT

        self.clock = int(phase < self.clock_steps // 2)

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
            "d_queue_len": self.qbuf.numel(),
            "queue_lag_steps": self.last_lag_steps,
            "queue_lag_ms": round(self.last_lag_steps * self.DT * 1000.0, 2),
            "D_queued": self.d_queued,
            "post_queue_and": self.post_queue_and,
            "probe_active": int(probe_active),
            "queue_vs_pre_delay_and_match": match_cell,
            "queue_vs_D_now_match": self_cell,
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
    # Below these sizes the canvases stop shrinking and scroll instead.
    CIRCUIT_MIN_W, CIRCUIT_MIN_H = 1160, 520
    WAVE_MIN_W = 1000

    ROI_A = (0.58, 0.25, 0.92, 0.75)  # Right ROI
    ROI_B = (0.08, 0.25, 0.42, 0.75)  # Left ROI

    def __init__(self):
        super().__init__()
        self.title("Photonic & Modal Logic Pattern Recogniser — Q-clocked LLM")
        self.geometry("1380x1060")
        self.minsize(960, 620)

        self.device = torch.device(CIRCUIT_DEVICE) if CIRCUIT_DEVICE else pick_device()
        self.device_var = tk.StringVar(value=(
            f"Compute device: {describe_device(self.device)} — photonic circuit + LLM"
            if self.device.type != "cpu"
            else "Compute device: CPU — WARNING: no GPU found (photonic circuit + LLM on CPU)"
        ))
        self.after_id: str | None = None
        self.cam_after_id: str | None = None
        self.resize_after_id: str | None = None
        self.running = False
        self.model: PhotonicBenchModel | None = None
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
        self.llm_model_var = tk.StringVar(value="Qwen/Qwen2.5-0.5B-Instruct")
        self.llm_max_tokens_var = tk.StringVar(value="128")
        self.llm_status_var = tk.StringVar(
            value=("LLM not loaded" if HAS_TRANSFORMERS else "LLM unavailable: install transformers")
        )
        self.llm_token_status_var = tk.StringVar(value="Tokens: 0")
        self.llm_prompt_text = None
        self.llm_output_text = None

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
        self.mode_var = tk.StringVar(value="Repeating bits")
        self.manual_x_var = tk.BooleanVar(value=False)
        self.ignore_startup_var = tk.BooleanVar(value=False)
        self.feedback_driver_a_var = tk.BooleanVar(value=False)

        self.pattern_var = tk.StringVar(value="1101001")
        self.text_var = tk.StringVar(value="the cat sat on the mat. ")
        self.stream_noise_var = tk.StringVar(value="0.0")

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
        if isinstance(event.widget, tk.Text):
            return  # the Text widget scrolls itself; don't also scroll the page
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

        logic_frame = ttk.LabelFrame(root, text="Circuit Controls", padding=7)
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
                "Repeating bits", "Text stream", "Random", "Square wave", "Manual",
                "Direct (x -> a, b)", "Photonic TIA",
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

        btns = ttk.Frame(root, padding=(10, 2))
        btns.pack(fill="x")

        self.run_btn = ttk.Button(btns, text="Run", command=self.toggle_running)
        self.run_btn.pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Step 5 ms", command=self.single_step).pack(side="left", padx=4)
        ttk.Button(btns, text="Reset", command=self.reset_simulation).pack(side="left", padx=4)
        ttk.Button(btns, text="Run to end", command=self.run_to_end).pack(side="left", padx=4)
        ttk.Button(btns, text="Export CSV", command=self.export_csv).pack(side="left", padx=4)

        ttk.Label(root, textvariable=self.device_var, padding=(12, 2)).pack(fill="x")
        ttk.Label(root, textvariable=self.status_var, padding=(12, 2)).pack(fill="x")
        ttk.Label(root, textvariable=self.photonic_var, padding=(12, 2)).pack(fill="x")

        middle = ttk.Frame(root)
        middle.pack(fill="both", expand=True, padx=10, pady=4)

        circuit_box = ttk.Frame(middle)
        circuit_box.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(
            circuit_box, background="white", height=520, highlightthickness=1, highlightbackground="#d9e2e8"
        )
        circuit_v = ttk.Scrollbar(circuit_box, orient="vertical", command=self.canvas.yview)
        circuit_h = ttk.Scrollbar(circuit_box, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=circuit_h.set, yscrollcommand=circuit_v.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        circuit_v.grid(row=0, column=1, sticky="ns")
        circuit_h.grid(row=1, column=0, sticky="ew")
        circuit_box.rowconfigure(0, weight=1)
        circuit_box.columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", self._schedule_redraw)

        self.cam_panel = ttk.LabelFrame(middle, text="USB Webcam Feed", padding=4, width=320)
        self.cam_preview = ttk.Label(self.cam_panel, text="USB cam stopped", anchor="center")
        self.cam_preview.pack(fill="both", expand=True)

        ttk.Label(root, textvariable=self.stats_var, padding=(12, 2)).pack(fill="x")

        wave_box = ttk.Frame(root)
        wave_box.pack(fill="x", padx=10, pady=(0, 6))
        self.waveform = tk.Canvas(
            wave_box, background="#f7fafc", height=180, highlightthickness=1, highlightbackground="#d9e2e8"
        )
        wave_h = ttk.Scrollbar(wave_box, orient="horizontal", command=self.waveform.xview)
        self.waveform.configure(xscrollcommand=wave_h.set)
        self.waveform.grid(row=0, column=0, sticky="ew")
        wave_h.grid(row=1, column=0, sticky="ew")
        wave_box.columnconfigure(0, weight=1)
        self.waveform.bind("<Configure>", self._schedule_redraw)

        # LLM controls are additive: retain the original circuit, lag analysis
        # and waveform UI.
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
        prompt_scroll = ttk.Scrollbar(prompt_row, orient="vertical", command=self.llm_prompt_text.yview)
        self.llm_prompt_text.configure(yscrollcommand=prompt_scroll.set)
        prompt_scroll.pack(side="right", fill="y")
        self.llm_prompt_text.pack(side="left", fill="x", expand=True)
        self.llm_prompt_text.insert("1.0", "Write a continuation based on the context.\n")
        self.llm_prompt_text.bind("<KeyRelease>", self.llm_update_prompt_bits)
        output_row = ttk.Frame(llm_frame)
        output_row.pack(fill="x", pady=(3, 0))
        self.llm_output_text = tk.Text(output_row, height=6, wrap="word", state="disabled")
        output_scroll = ttk.Scrollbar(output_row, orient="vertical", command=self.llm_output_text.yview)
        self.llm_output_text.configure(yscrollcommand=output_scroll.set)
        output_scroll.pack(side="right", fill="y")
        self.llm_output_text.pack(side="left", fill="x", expand=True)
        ttk.Label(llm_frame, textvariable=self.llm_token_status_var).pack(fill="x")
        self.llm_update_prompt_bits()

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
            "retrieval_corpus": "",
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

        self.playback_speed = cfg["speed"]
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
            device=self.device,
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
        return m.queue_preview()

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

        w = max(self.CIRCUIT_MIN_W, c.winfo_width())
        h = max(self.CIRCUIT_MIN_H, c.winfo_height())
        c.scale("all", 0, 0, w / 1160, h / 590)
        c.configure(scrollregion=(0, 0, w, h))

    def _draw_waveforms(self):
        m = self.model
        assert m is not None
        p = self.waveform
        p.delete("all")

        w = max(self.WAVE_MIN_W, p.winfo_width())
        p.configure(scrollregion=(0, 0, w, 180))
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
