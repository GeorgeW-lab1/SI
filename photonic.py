import csv
import math
import queue
import random
import threading
import time
import tkinter as tk

from collections import deque
from tkinter import ttk, filedialog, messagebox

import cv2
import torch

from PIL import Image, ImageTk
from transformers import AutoTokenizer, AutoModelForCausalLM


"""
WEBCAM -> CIRCUIT -> LLM DECODER

Camera:
    Right ROI -> detector A
    Left ROI  -> detector B

Circuit:
    A -> optional randomized driver -> a
    B -> RC, tau 50 ms -> threshold 0.7 V -> b

    truth = a AND b
    delayed = transport_delay(truth, 150 ms)
    Y = a OR b
    D = delayed OR Y

    Q samples D at rising edges of a 10 Hz clock.
    SELECT = Q
    Observe = NOT Q

LLM controls:
    SELECT LOW->HIGH: permit one token
    RC voltage: temperature from 0.2 to 1.4
    delayed AND == 0: greedy decoding
    delayed AND == 1: top-k sampling
    running AND accumulator: generation enable

Accumulator:
    Ignores delay startup.
    Then accumulates delayed AND at clock rising edges.
    Once LOW, it stays LOW until "Re-arm accumulator".

Probe:
    Logs actual AND values, with no artificial noise.

Important:
    This is software logic/RC simulation, not a physical
    transistor-level circuit or an LLM implemented in gates.
"""


def put_latest(q, item):
    """Replace a queued item, keeping only the newest."""
    try:
        q.get_nowait()
    except queue.Empty:
        pass

    try:
        q.put_nowait(item)
    except queue.Full:
        pass


class Camera:
    def __init__(self, index):
        self.index = index
        self.frames = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.error = None

        self.thread = threading.Thread(
            target=self.run,
            daemon=True
        )
        self.thread.start()

    def run(self):
        capture = None

        try:
            capture = cv2.VideoCapture(self.index)

            if not capture.isOpened():
                raise RuntimeError(
                    f"Could not open camera {self.index}."
                )

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while not self.stop_event.is_set():
                ok, frame = capture.read()

                if not ok or frame is None:
                    raise RuntimeError(
                        "Camera stopped returning frames."
                    )

                put_latest(
                    self.frames,
                    (time.perf_counter(), frame)
                )

        except Exception as error:
            self.error = str(error)

        finally:
            if capture is not None:
                capture.release()

    def stop(self):
        self.stop_event.set()


class Decoder:
    """
    One background model worker.

    GUI requests:
        load  -> load model
        start -> tokenize prompt and initialize a session
        token -> perform exactly one decoding step

    At most one model operation is outstanding.
    """

    def __init__(self):
        self.commands = queue.Queue()
        self.events = queue.Queue()

        self.busy = threading.Event()
        self.shutdown = threading.Event()

        self.model = None
        self.tokenizer = None
        self.device = None

        self.active = False
        self.cache = None
        self.next_input = None
        self.generated = []
        self.max_tokens = 128

        self.thread = threading.Thread(
            target=self.run,
            daemon=True
        )
        self.thread.start()

    def submit(self, kind, data=None):
        if self.busy.is_set():
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
                self.events.put(("error", str(error)))

            finally:
                self.busy.clear()

    def load_model(self, name):
        self.active = False

        # Release previous references before loading another model.
        self.cache = None
        self.next_input = None
        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=False
        )

        model = AutoModelForCausalLM.from_pretrained(
            name,
            trust_remote_code=False
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model.to(device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        self.events.put((
            "loaded",
            f"Model loaded on {device}: {name}"
        ))

    def start_session(self, data):
        if self.model is None:
            raise RuntimeError("Load a model first.")

        prompt = data["prompt"]
        self.max_tokens = data["max_tokens"]

        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False
            )
        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            )

        ids = inputs["input_ids"].to(self.device)

        if ids.shape[1] == 0:
            raise RuntimeError("Prompt produced no input tokens.")

        context_limit = getattr(
            self.model.config,
            "max_position_embeddings",
            None
        )

        if isinstance(context_limit, int):
            if ids.shape[1] + self.max_tokens > context_limit:
                raise RuntimeError(
                    "Prompt plus requested output exceeds "
                    f"the model context limit ({context_limit})."
                )

        self.cache = None
        self.next_input = ids
        self.generated = []
        self.active = True

        self.events.put((
            "ready",
            "Prompt ready. Waiting for a permitted SELECT rising edge."
        ))

    def generate_one(self, control):
        if not self.active:
            return

        with torch.inference_mode():
            result = self.model(
                input_ids=self.next_input,
                past_key_values=self.cache,
                use_cache=True
            )

            logits = result.logits[0, -1].float()
            self.cache = result.past_key_values

            temperature = control["temperature"]
            sampling = bool(control["delayed"])

            if sampling:
                k = min(40, logits.shape[-1])

                values, indices = torch.topk(
                    logits / temperature,
                    k
                )

                probabilities = torch.softmax(
                    values,
                    dim=-1
                )

                selected_position = torch.multinomial(
                    probabilities,
                    num_samples=1
                ).item()

                token_id = indices[selected_position].item()
                mode = "sample"
            else:
                token_id = torch.argmax(logits).item()
                mode = "greedy"

            self.generated.append(token_id)

            self.next_input = torch.tensor(
                [[token_id]],
                dtype=torch.long,
                device=self.device
            )

            text = self.tokenizer.decode(
                self.generated,
                skip_special_tokens=True
            )

            eos = self.model.generation_config.eos_token_id

            if eos is None:
                eos = self.tokenizer.eos_token_id

            if isinstance(eos, int):
                eos_ids = {eos}
            elif eos is None:
                eos_ids = set()
            else:
                eos_ids = set(eos)

            finished = (
                token_id in eos_ids
                or len(self.generated) >= self.max_tokens
            )

            if finished:
                self.active = False

            self.events.put((
                "token",
                {
                    "text": text,
                    "token_id": token_id,
                    "count": len(self.generated),
                    "mode": mode,
                    "temperature": temperature,
                    "finished": finished,
                    "control": control
                }
            ))


class CircuitLLM(tk.Tk):
    TAU = 0.050
    DELAY = 0.150
    CLOCK = 0.100

    ROI_A = (0.58, 0.25, 0.92, 0.75)
    ROI_B = (0.08, 0.25, 0.42, 0.75)

    def __init__(self):
        super().__init__()

        self.title("Webcam Circuit-Controlled LLM")
        self.geometry("1250x980")
        self.minsize(1050, 820)

        self.camera = None
        self.decoder = Decoder()

        self.closed = False
        self.after_id = None
        self.photo = None

        self.last_frame_time = None
        self.last_tick = time.perf_counter()

        self.model_loaded = False
        self.session_active = False
        self.pending_start = False

        self.threshold = tk.DoubleVar(value=130)
        self.camera_index = tk.StringVar(value="0")

        self.model_name = tk.StringVar(
            value="HuggingFaceTB/SmolLM2-135M-Instruct"
        )

        self.max_tokens = tk.StringVar(value="128")
        self.randomize_a = tk.BooleanVar(value=False)

        self.camera_status = tk.StringVar(
            value="Camera stopped"
        )
        self.model_status = tk.StringVar(
            value="Load a model to begin."
        )
        self.circuit_status = tk.StringVar()
        self.token_status = tk.StringVar()

        self.detector_a = 0
        self.detector_b = 0
        self.brightness_a = 0.0
        self.brightness_b = 0.0

        self.records = []
        self.token_records = []

        self.reset_circuit_state()
        self.build_ui()

        self.protocol("WM_DELETE_WINDOW", self.close)
        self.loop()

    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------

    def build_ui(self):
        settings = ttk.LabelFrame(
            self,
            text="Model and webcam",
            padding=8
        )
        settings.pack(fill="x", padx=10, pady=8)

        ttk.Label(
            settings,
            text="Model / local directory"
        ).grid(row=0, column=0, padx=4)

        ttk.Entry(
            settings,
            textvariable=self.model_name,
            width=45
        ).grid(row=0, column=1, columnspan=3, padx=4)

        ttk.Button(
            settings,
            text="Load model",
            command=self.load_model
        ).grid(row=0, column=4, padx=4)

        ttk.Label(
            settings,
            text="Max tokens"
        ).grid(row=0, column=5, padx=4)

        ttk.Entry(
            settings,
            textvariable=self.max_tokens,
            width=7
        ).grid(row=0, column=6, padx=4)

        ttk.Label(
            settings,
            text="Camera"
        ).grid(row=1, column=0, padx=4, pady=8)

        ttk.Entry(
            settings,
            textvariable=self.camera_index,
            width=6
        ).grid(row=1, column=1, sticky="w", padx=4)

        ttk.Button(
            settings,
            text="Start camera",
            command=self.start_camera
        ).grid(row=1, column=2, padx=4)

        ttk.Button(
            settings,
            text="Stop camera",
            command=self.stop_camera
        ).grid(row=1, column=3, padx=4)

        ttk.Label(
            settings,
            text="Brightness threshold"
        ).grid(row=1, column=4, padx=4)

        ttk.Scale(
            settings,
            from_=0,
            to=255,
            variable=self.threshold,
            length=180
        ).grid(row=1, column=5, columnspan=2, padx=4)

        ttk.Checkbutton(
            settings,
            text="Randomize Driver A",
            variable=self.randomize_a
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        ttk.Button(
            settings,
            text="Reset circuit",
            command=self.reset_circuit
        ).grid(row=2, column=2, padx=4)

        ttk.Button(
            settings,
            text="Re-arm accumulator",
            command=self.rearm_accumulator
        ).grid(row=2, column=3, padx=4)

        ttk.Button(
            settings,
            text="Export logs",
            command=self.export_logs
        ).grid(row=2, column=4, padx=4)

        center = ttk.Frame(self)
        center.pack(fill="both", expand=True, padx=10)

        camera_frame = ttk.LabelFrame(
            center,
            text="Webcam: B left / A right",
            padding=5
        )
        camera_frame.pack(
            side="left",
            fill="both",
            expand=True
        )

        self.preview = ttk.Label(
            camera_frame,
            text="Start the webcam",
            anchor="center"
        )
        self.preview.pack(fill="both", expand=True)

        circuit_frame = ttk.LabelFrame(
            center,
            text="Live circuit",
            padding=5
        )
        circuit_frame.pack(
            side="right",
            fill="both",
            expand=True,
            padx=(8, 0)
        )

        self.canvas = tk.Canvas(
            circuit_frame,
            width=550,
            height=300,
            background="white",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        ttk.Label(
            self,
            textvariable=self.camera_status,
            padding=(10, 3)
        ).pack(fill="x")

        ttk.Label(
            self,
            textvariable=self.circuit_status,
            padding=(10, 3)
        ).pack(fill="x")

        prompt_frame = ttk.LabelFrame(
            self,
            text="Text prompt",
            padding=5
        )
        prompt_frame.pack(fill="x", padx=10, pady=5)

        self.prompt = tk.Text(
            prompt_frame,
            height=3,
            wrap="word"
        )
        self.prompt.pack(fill="x")

        self.prompt.insert(
            "1.0",
            "Write a short imaginative story about light and shadows."
        )

        controls = ttk.Frame(self, padding=(10, 3))
        controls.pack(fill="x")

        ttk.Button(
            controls,
            text="Arm generation",
            command=self.arm_generation
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            controls,
            text="Stop generation",
            command=self.stop_generation
        ).pack(side="left", padx=6)

        ttk.Label(
            controls,
            textvariable=self.model_status
        ).pack(side="left", padx=10)

        output_frame = ttk.LabelFrame(
            self,
            text="Circuit-controlled generated text",
            padding=5
        )
        output_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        self.output = tk.Text(
            output_frame,
            height=9,
            wrap="word",
            state="disabled"
        )
        self.output.pack(fill="both", expand=True)

        ttk.Label(
            self,
            textvariable=self.token_status,
            padding=(10, 6)
        ).pack(fill="x")

    # -----------------------------------------------------
    # Camera
    # -----------------------------------------------------

    def start_camera(self):
        if self.camera is not None:
            return

        try:
            index = int(self.camera_index.get())
            if index < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Camera",
                "Enter a nonnegative camera index."
            )
            return

        self.last_frame_time = None
        self.camera = Camera(index)
        self.camera_status.set("Opening camera...")

    def stop_camera(self):
        if self.camera is not None:
            self.camera.stop()

        self.last_frame_time = None
        self.camera_status.set("Stopping camera...")

    def camera_fresh(self):
        return (
            self.camera is not None
            and not self.camera.stop_event.is_set()
            and self.last_frame_time is not None
            and time.perf_counter() - self.last_frame_time < 0.75
        )

    @staticmethod
    def bounds(frame, roi):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = roi

        return (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height)
        )

    def process_frame(self, stamp, frame):
        self.last_frame_time = stamp

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        a_bounds = self.bounds(frame, self.ROI_A)
        b_bounds = self.bounds(frame, self.ROI_B)

        ax1, ay1, ax2, ay2 = a_bounds
        bx1, by1, bx2, by2 = b_bounds

        self.brightness_a = float(
            gray[ay1:ay2, ax1:ax2].mean()
        )
        self.brightness_b = float(
            gray[by1:by2, bx1:bx2].mean()
        )

        threshold = self.threshold.get()

        self.detector_a = int(
            self.brightness_a >= threshold
        )
        self.detector_b = int(
            self.brightness_b >= threshold
        )

        preview = frame.copy()

        for label, bounds, state, level in [
            ("A", a_bounds, self.detector_a, self.brightness_a),
            ("B", b_bounds, self.detector_b, self.brightness_b)
        ]:
            x1, y1, x2, y2 = bounds
            color = (40, 210, 40) if state else (50, 70, 230)

            cv2.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                preview,
                f"{label}: {level:.0f} -> {state}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        image = Image.fromarray(
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        )

        image.thumbnail((
            max(320, self.preview.winfo_width()),
            max(240, self.preview.winfo_height())
        ))

        self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo, text="")

        self.camera_status.set(
            f"Live camera | "
            f"A={self.brightness_a:.1f}, "
            f"B={self.brightness_b:.1f}, "
            f"threshold={threshold:.0f}"
        )

    def poll_camera(self):
        if self.camera is None:
            return

        if not self.camera.stop_event.is_set():
            try:
                stamp, frame = self.camera.frames.get_nowait()
                self.process_frame(stamp, frame)
            except queue.Empty:
                pass

        if not self.camera.thread.is_alive():
            error = self.camera.error
            self.camera = None
            self.last_frame_time = None
            self.camera_status.set(error or "Camera stopped")

    # -----------------------------------------------------
    # Circuit
    # -----------------------------------------------------

    def reset_circuit_state(self):
        self.t = 0.0

        self.a = 0
        self.b = 0
        self.v = 0.0

        self.truth = 0
        self.delayed = 0
        self.y = 0
        self.d = 0

        self.q = 0
        self.observe = 1

        self.accumulator = 1
        self.accumulator_samples = 0

        self.next_clock = self.CLOCK
        self.next_random = 0.0
        self.random_bit = 0

        self.delay_queue = deque()
        self.rng = random.Random(7)

        self.select_edges = 0
        self.dispatched = 0
        self.blocked = 0

    def reset_circuit(self):
        self.reset_circuit_state()
        self.records.clear()
        self.last_tick = time.perf_counter()

    def rearm_accumulator(self):
        self.accumulator = 1
        self.accumulator_samples = 0

    def advance_circuit(self, elapsed):
        target = self.t + elapsed

        while self.t < target - 1e-10:
            dt = min(0.005, target - self.t)

            if self.next_clock > self.t + 1e-10:
                dt = min(dt, self.next_clock - self.t)

            self.circuit_step(dt)

    def circuit_step(self, dt):
        new_t = self.t + dt

        if self.t >= self.next_random - 1e-9:
            self.random_bit = self.rng.randrange(2)
            self.next_random += 0.100

        if self.randomize_a.get():
            self.a = self.detector_a & self.random_bit
        else:
            self.a = self.detector_a

        target_voltage = 5.0 * self.detector_b

        self.v = (
            target_voltage
            + (self.v - target_voltage)
            * math.exp(-dt / self.TAU)
        )

        self.b = int(self.v > 0.7)
        self.truth = self.a & self.b

        self.delay_queue.append((
            new_t + self.DELAY,
            self.truth
        ))

        while (
            self.delay_queue
            and self.delay_queue[0][0] <= new_t + 1e-9
        ):
            _, self.delayed = self.delay_queue.popleft()

        self.y = self.a | self.b
        self.d = self.delayed | self.y

        select_rising = False

        if new_t >= self.next_clock - 1e-9:
            old_q = self.q

            self.q = self.d
            self.observe = 1 - self.q
            select_rising = old_q == 0 and self.q == 1

            # Ignore prehistory introduced by the transport delay.
            if new_t >= self.DELAY + self.CLOCK:
                self.accumulator &= self.delayed
                self.accumulator_samples += 1

            self.next_clock += self.CLOCK

        self.t = new_t

        if select_rising:
            self.select_edges += 1
            self.on_select_edge()

        self.records.append({
            "time_s": self.t,
            "detector_A": self.detector_a,
            "detector_B": self.detector_b,
            "a": self.a,
            "rc_voltage_V": self.v,
            "b": self.b,
            "AND": self.truth,
            "delayed_AND": self.delayed,
            "Y": self.y,
            "D": self.d,
            "SELECT": self.q,
            "Observe": self.observe,
            "accumulator": self.accumulator,
            "temperature": self.temperature(),
            "select_rising": int(select_rising)
        })

    def temperature(self):
        return 0.2 + 1.2 * min(1.0, max(0.0, self.v / 5.0))

    def on_select_edge(self):
        if not self.session_active:
            return

        if (
            self.accumulator == 0
            or not self.camera_fresh()
            or self.decoder.busy.is_set()
        ):
            self.blocked += 1
            return

        snapshot = {
            "time_s": self.t,
            "rc_voltage_V": self.v,
            "temperature": self.temperature(),
            "delayed": self.delayed,
            "SELECT": self.q,
            "Observe": self.observe,
            "accumulator": self.accumulator
        }

        if self.decoder.submit("token", snapshot):
            self.dispatched += 1
        else:
            self.blocked += 1

    # -----------------------------------------------------
    # Model / generation
    # -----------------------------------------------------

    def load_model(self):
        if self.decoder.busy.is_set():
            messagebox.showinfo(
                "Busy",
                "Wait for the current model operation."
            )
            return

        name = self.model_name.get().strip()

        if not name:
            return

        self.session_active = False
        self.pending_start = False
        self.model_loaded = False

        if self.decoder.submit("load", name):
            self.model_status.set("Loading model...")

    def arm_generation(self):
        if not self.model_loaded:
            messagebox.showwarning(
                "Model",
                "Load the model first."
            )
            return

        if self.decoder.busy.is_set():
            messagebox.showinfo(
                "Busy",
                "Wait for the current model operation."
            )
            return

        prompt = self.prompt.get("1.0", "end").strip()

        if not prompt:
            messagebox.showwarning(
                "Prompt",
                "Enter a text prompt."
            )
            return

        try:
            limit = int(self.max_tokens.get())
            if not 1 <= limit <= 2048:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Max tokens",
                "Enter an integer between 1 and 2048."
            )
            return

        self.session_active = False
        self.pending_start = True
        self.token_records.clear()

        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")

        if self.decoder.submit(
            "start",
            {"prompt": prompt, "max_tokens": limit}
        ):
            self.model_status.set("Preparing prompt...")
        else:
            self.pending_start = False

    def stop_generation(self):
        self.session_active = False
        self.pending_start = False

        self.model_status.set(
            "Generation stopped. An in-flight token may finish."
        )

    def poll_decoder(self):
        while True:
            try:
                kind, data = self.decoder.events.get_nowait()
            except queue.Empty:
                break

            if kind == "loaded":
                self.model_loaded = True
                self.model_status.set(data)

            elif kind == "ready":
                if self.pending_start:
                    self.session_active = True
                    self.pending_start = False
                    self.model_status.set(data)

            elif kind == "error":
                self.session_active = False
                self.pending_start = False
                self.model_status.set(f"Error: {data}")

            elif kind == "token":
                self.output.configure(state="normal")
                self.output.delete("1.0", "end")
                self.output.insert("1.0", data["text"])
                self.output.see("end")
                self.output.configure(state="disabled")

                record = dict(data["control"])
                record.update({
                    "token_number": data["count"],
                    "token_id": data["token_id"],
                    "mode": data["mode"]
                })
                self.token_records.append(record)

                self.token_status.set(
                    f"Tokens: {data['count']} | "
                    f"Mode: {data['mode']} | "
                    f"Temperature: {data['temperature']:.3f}"
                )

                if data["finished"]:
                    self.session_active = False
                    self.model_status.set(
                        "Generation finished."
                    )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    def box(self, x, y, w, h, text, state=None):
        fill = (
            "#d8f1e2" if state
            else "#eff3f7"
        )

        self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill=fill,
            outline="#a5b5c3"
        )
        self.canvas.create_text(
            x + w / 2,
            y + h / 2,
            text=text,
            font=("Arial", 10),
            justify="center"
        )

    def arrow(self, points):
        self.canvas.create_line(
            *points,
            arrow=tk.LAST,
            fill="#748a9b",
            width=2
        )

    def draw_circuit(self):
        c = self.canvas
        c.delete("all")

        self.box(
            15, 15, 210, 45,
            f"Detector A -> Driver A\n"
            f"{self.detector_a} -> a={self.a}",
            self.a
        )

        self.box(
            300, 15, 230, 60,
            f"Detector B -> RC -> b\n"
            f"{self.detector_b} -> {self.v:.3f} V -> {self.b}\n"
            "tau=50 ms, threshold=0.7 V",
            self.b
        )

        self.box(
            15, 110, 115, 45,
            f"AND(a,b)\n{self.truth}",
            self.truth
        )
        self.box(
            195, 110, 135, 45,
            f"Delay 150 ms\n{self.delayed}",
            self.delayed
        )
        self.box(
            395, 110, 135, 45,
            f"Running AND\n{self.accumulator}",
            self.accumulator
        )

        self.arrow([130, 132, 195, 132])
        self.arrow([330, 132, 395, 132])

        self.box(
            15, 200, 115, 45,
            f"OR(a,b)\nY={self.y}",
            self.y
        )
        self.box(
            195, 200, 135, 45,
            f"OR(delay,Y)\nD={self.d}",
            self.d
        )
        self.box(
            395, 190, 135, 65,
            f"Flip-flop 10 Hz\n"
            f"SELECT={self.q}\n"
            f"Observe={self.observe}",
            self.q
        )

        self.arrow([130, 222, 195, 222])
        self.arrow([262, 155, 262, 200])
        self.arrow([330, 222, 395, 222])

        self.box(
            15, 295, 515, 60,
            "SELECT edge -> one token | accumulator -> enable\n"
            f"RC -> temperature {self.temperature():.3f} | "
            f"delayed AND -> "
            f"{'sampling' if self.delayed else 'greedy'}"
        )

        width = max(1, c.winfo_width())
        height = max(1, c.winfo_height())
        c.scale("all", 0, 0, width / 550, height / 375)

        self.circuit_status.set(
            f"Circuit time: {self.t:.2f} s | "
            f"SELECT edges: {self.select_edges} | "
            f"Token requests: {self.dispatched} | "
            f"Blocked/busy edges: {self.blocked} | "
            f"Accumulator: {self.accumulator}"
        )

    # -----------------------------------------------------
    # Main loop / export / cleanup
    # -----------------------------------------------------

    def loop(self):
        if self.closed:
            return

        self.poll_camera()
        self.poll_decoder()

        now = time.perf_counter()
        elapsed = now - self.last_tick
        self.last_tick = now

        if self.camera_fresh():
            # Avoid fabricating a long history after GUI stalls.
            # Circuit time pauses rather than catching up indefinitely.
            self.advance_circuit(min(elapsed, 0.100))

        self.draw_circuit()

        self.after_id = self.after(20, self.loop)

    def export_logs(self):
        if not self.records:
            messagebox.showinfo(
                "Export",
                "No circuit data yet."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Export circuit log",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )

        if not path:
            return

        try:
            with open(
                path, "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(self.records[0])
                )
                writer.writeheader()
                writer.writerows(self.records)

            if self.token_records:
                token_path = (
                    path[:-4] + "_tokens.csv"
                    if path.lower().endswith(".csv")
                    else path + "_tokens.csv"
                )

                with open(
                    token_path,
                    "w",
                    newline="",
                    encoding="utf-8"
                ) as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=list(self.token_records[0])
                    )
                    writer.writeheader()
                    writer.writerows(self.token_records)

        except OSError as error:
            messagebox.showerror("Export failed", str(error))
            return

        messagebox.showinfo(
            "Export",
            "Circuit log exported. A separate token-control "
            "log was exported if tokens were generated."
        )

    def close(self):
        self.closed = True
        self.session_active = False

        if self.after_id is not None:
            self.after_cancel(self.after_id)

        if self.camera is not None:
            self.camera.stop()

        self.decoder.shutdown.set()
        self.destroy()


if __name__ == "__main__":
    app = CircuitLLM()
    app.mainloop()
