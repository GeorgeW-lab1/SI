
#!/usr/bin/env python3
"""
RUBBER AI - SELF-GENERATING RUBBER CODE
=======================================

Every dataset load generates a new custom .rbc program.

DATASET -> ANALYSIS -> PROGRAM GENERATION
         -> RBC COMPILATION -> EXECUTION
         -> TRIGRAM LEARNING -> PROMPT GENERATION

Features:
    - Automatically generated Rubber Code programs
    - Binary spatial dataset encoding
    - Trigram language learning
    - Two-level sequential arithmetic
    - Probability-based generation
    - Prompt injection
    - Generated program persistence
    - Interactive terminal
    - JSON model saving/loading

Python standard library only.
"""

import os
import re
import json
import math
import shlex
import random
import hashlib
import argparse
import secrets

from collections import Counter, defaultdict


VERSION = "2.0"


# ============================================================
# UTILITIES
# ============================================================

def tokenize(text):

    return re.findall(
        r"[A-Za-z0-9_]+|[^\w\s]",
        text.lower()
    )


def digest(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def hash_value(text):

    return int.from_bytes(
        hashlib.sha256(
            text.encode("utf-8")
        ).digest()[:4],
        "big"
    )


def clamp(value, low, high):

    return max(low, min(high, value))


# ============================================================
# SPATIAL BINARY ENGINE
# ============================================================

class RubberSpace:

    def __init__(self, bits=12):

        self.bits = bits
        self.limit = 1 << bits

        self.positions = []

    def encode(self, token, position):

        h = hash_value(token)

        x = h % self.limit

        y = ((h >> 8) + position) % self.limit

        z = ((h >> 16) ^ position) % self.limit

        return {
            "token": token,
            "position": position,
            "xyz": (x, y, z),
            "binary": (
                format(x, f"0{self.bits}b"),
                format(y, f"0{self.bits}b"),
                format(z, f"0{self.bits}b")
            )
        }

    def encode_dataset(self, tokens):

        self.positions = [
            self.encode(token, i)
            for i, token in enumerate(tokens)
        ]

        return self.positions

    def affinity(self, a, b):

        distance = math.sqrt(
            sum(
                (a[i] - b[i]) ** 2
                for i in range(3)
            )
        )

        maximum = math.sqrt(
            3 * (self.limit - 1) ** 2
        )

        return 1.0 - distance / maximum


# ============================================================
# TWO-LEVEL ARITHMETIC
# ============================================================

class RubberMath:

    def __init__(self, mode="ADD", scale=1.0, weight=0.5):

        self.mode = mode
        self.scale = scale
        self.weight = weight

        self.local_state = 0.0
        self.global_state = 0.0

    def level_one(
        self,
        value,
        position,
        affinity,
        prompt_state
    ):

        result = (
            value * self.scale
            + position * 0.001
            + affinity * 0.5
            + prompt_state * 0.5
        )

        self.local_state = math.tanh(result)

        return self.local_state

    def level_two(self, value, previous=None):

        if previous is None:
            previous = self.global_state

        if self.mode == "ADD":

            result = value + previous * self.weight

        elif self.mode == "MULTIPLY":

            result = value * (
                1.0 + previous * self.weight
            )

        elif self.mode == "AVERAGE":

            result = (value + previous) / 2.0

        elif self.mode == "SUBTRACT":

            result = value - previous * self.weight

        else:

            result = value + previous * self.weight

        self.global_state = math.tanh(result)

        return self.global_state


# ============================================================
# TRIGRAM LANGUAGE ENGINE
# ============================================================

class RubberTrigrams:

    def __init__(self, smoothing=0.1):

        self.trigrams = defaultdict(Counter)
        self.bigrams = defaultdict(Counter)
        self.unigrams = Counter()

        self.vocabulary = set()

        self.smoothing = smoothing

    def learn(self, tokens):

        self.vocabulary.update(tokens)

        self.unigrams.update(tokens)

        for i in range(len(tokens) - 1):

            self.bigrams[
                tokens[i]
            ][tokens[i + 1]] += 1

        for i in range(len(tokens) - 3):

            context = tuple(tokens[i:i + 3])

            target = tokens[i + 3]

            self.trigrams[context][target] += 1

    def predict(
        self,
        context,
        numeric=0.0,
        spatial=0.0,
        prompt_bias=None,
        temperature=0.8
    ):

        if prompt_bias is None:
            prompt_bias = {}

        context = tuple(context[-3:])

        candidates = self.trigrams.get(
            context,
            Counter()
        )

        mode = "TRIGRAM"

        if not candidates:

            candidates = self.bigrams.get(
                context[-1] if context else "",
                Counter()
            )

            mode = "BIGRAM"

        if not candidates:

            candidates = self.unigrams

            mode = "UNIGRAM"

        if not candidates:
            return None

        total = sum(candidates.values())

        vocab = max(
            len(self.vocabulary),
            1
        )

        scores = {}

        for token, count in candidates.items():

            probability = (
                count + self.smoothing
            ) / (
                total + self.smoothing * vocab
            )

            score = math.log(
                max(probability, 1e-12)
            )

            spatial_value = (
                hash_value(token) / 4294967295.0
            ) - 0.5

            score += spatial_value * spatial

            score += numeric * 0.2

            score += prompt_bias.get(token, 0.0)

            scores[token] = score

        temperature = max(0.01, temperature)

        maximum = max(scores.values())

        weights = [
            math.exp(
                (score - maximum) / temperature
            )
            for score in scores.values()
        ]

        selected = random.choices(
            list(scores.keys()),
            weights=weights,
            k=1
        )[0]

        return selected


# ============================================================
# DATASET ANALYSER
# ============================================================

class DatasetAnalyser:

    def analyse(self, tokens):

        counts = Counter(tokens)

        vocabulary = len(counts)

        total = len(tokens)

        frequencies = Counter(
            tuple(tokens[i:i + 3])
            for i in range(max(0, total - 2))
        )

        entropy = 0.0

        if total:

            for count in counts.values():

                p = count / total

                entropy -= p * math.log2(p)

        unique_trigrams = len(frequencies)

        possible_trigrams = max(
            total - 2,
            1
        )

        diversity = (
            unique_trigrams / possible_trigrams
        )

        repetition = (
            1.0 - vocabulary / max(total, 1)
        )

        average_frequency = (
            total / max(vocabulary, 1)
        )

        return {
            "tokens": total,
            "vocabulary": vocabulary,
            "entropy": entropy,
            "trigrams": unique_trigrams,
            "trigram_diversity": diversity,
            "repetition": repetition,
            "average_frequency": average_frequency
        }


# ============================================================
# RUBBER CODE GENERATOR
# ============================================================

class RubberProgramGenerator:

    """
    Generates a new executable RBC program for every
    dataset loading event.

    Program instructions are selected using dataset
    statistics and a fresh random generator.
    """

    def generate(self, analysis, dataset_hash):

        seed = secrets.randbits(64)

        rng = random.Random(seed)

        # Dataset-dependent spatial resolution.
        bits = int(
            clamp(
                8 + analysis["entropy"] * 0.65,
                8,
                20
            )
        )

        # Select arithmetic architecture.
        if analysis["repetition"] > 0.65:

            mode = rng.choice([
                "AVERAGE",
                "ADD"
            ])

        elif analysis["entropy"] > 5:

            mode = rng.choice([
                "ADD",
                "SUBTRACT",
                "AVERAGE"
            ])

        else:

            mode = rng.choice([
                "ADD",
                "MULTIPLY"
            ])

        scale = round(
            rng.uniform(0.75, 1.35),
            4
        )

        weight = round(
            clamp(
                0.25 + analysis["trigram_diversity"] * 0.5,
                0.1,
                0.8
            ),
            4
        )

        temperature = round(
            clamp(
                1.0 - analysis["repetition"] * 0.5,
                0.35,
                1.0
            ),
            4
        )

        spatial_strength = round(
            rng.uniform(0.15, 0.75),
            4
        )

        smoothing = round(
            rng.uniform(0.05, 0.2),
            4
        )

        program_id = secrets.token_hex(6)

        lines = [
            "# GENERATED RUBBER PROGRAM",
            f"# ID {program_id}",
            f"# DATASET {dataset_hash}",
            "",
            "RUBBER 2",
            "",
            f"SPACE BITS {bits}",
            "SPACE DIMENSIONS 3",
            "",
            "TRIGRAM ORDER 3",
            "",
            f"MATH LEVEL1 SCALE {scale}",
            f"MATH LEVEL2 {mode} WEIGHT {weight}",
            "",
            f"PROB SMOOTH {smoothing}",
            f"PROB TEMP {temperature}",
            f"PROB SPATIAL {spatial_strength}",
            "",
            "EXECUTE ENCODE",
            "EXECUTE LEARN",
            "EXECUTE SEQUENCE",
            "EXECUTE READY",
            "",
            "END"
        ]

        return {
            "id": program_id,
            "seed": seed,
            "source": "\n".join(lines),
            "settings": {
                "bits": bits,
                "mode": mode,
                "scale": scale,
                "weight": weight,
                "temperature": temperature,
                "spatial": spatial_strength,
                "smoothing": smoothing
            }
        }


# ============================================================
# RUBBER CODE COMPILER
# ============================================================

class RubberCompiler:

    def compile(self, source):

        instructions = []

        for line_number, line in enumerate(
            source.splitlines(),
            1
        ):

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = shlex.split(line)

            command = parts[0].upper()

            args = parts[1:]

            if command == "RUBBER":
                continue

            elif command == "SPACE":

                instructions.append((
                    "SPACE",
                    args
                ))

            elif command == "TRIGRAM":

                instructions.append((
                    "TRIGRAM",
                    args
                ))

            elif command == "MATH":

                instructions.append((
                    "MATH",
                    args
                ))

            elif command == "PROB":

                instructions.append((
                    "PROB",
                    args
                ))

            elif command == "EXECUTE":

                instructions.append((
                    "EXECUTE",
                    args
                ))

            elif command == "END":

                break

            else:

                raise SyntaxError(
                    f"RBC line {line_number}: "
                    f"Unknown instruction {command}"
                )

        return instructions


# ============================================================
# RUBBER AI ENGINE
# ============================================================

class RubberAI:

    def __init__(self, program_dir="rubber_programs"):

        self.program_dir = program_dir

        os.makedirs(
            program_dir,
            exist_ok=True
        )

        self.generator = RubberProgramGenerator()

        self.compiler = RubberCompiler()

        self.space = RubberSpace()

        self.math = RubberMath()

        self.language = RubberTrigrams()

        self.analyser = DatasetAnalyser()

        self.dataset = []

        self.spatial_dataset = []

        self.analysis = {}

        self.program = None

        self.instructions = []

        self.prompt = ""

        self.prompt_state = 0.0

        self.ready = False

    # --------------------------------------------------------
    # LOAD DATASET AND GENERATE PROGRAM
    # --------------------------------------------------------

    def load_dataset(self, filename):

        print("\n[RUBBER] Loading dataset...")

        with open(
            filename,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as f:

            text = f.read()

        self.dataset = tokenize(text)

        if not self.dataset:

            raise ValueError(
                "Dataset contains no tokens."
            )

        dataset_hash = digest(text)

        # Analyse dataset.
        self.analysis = self.analyser.analyse(
            self.dataset
        )

        print("[RUBBER] Analysing dataset...")

        # Generate a fresh program.
        self.program = self.generator.generate(
            self.analysis,
            dataset_hash
        )

        # Save the generated source.
        program_path = os.path.join(
            self.program_dir,
            f"rubber_{self.program['id']}.rbc"
        )

        with open(
            program_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                self.program["source"]
            )

        print(
            "[RUBBER] Generated:",
            program_path
        )

        # Compile the generated program.
        self.instructions = self.compiler.compile(
            self.program["source"]
        )

        print("[RUBBER] Compiling generated program...")

        # Reset all previous learned state.
        self.space = RubberSpace()

        self.math = RubberMath()

        self.language = RubberTrigrams(
            smoothing=self.program["settings"]["smoothing"]
        )

        self.ready = False

        # Execute generated instructions.
        self.execute_program()

        self.ready = True

        print("[RUBBER] Dataset program ready.")

        return program_path

    # --------------------------------------------------------
    # EXECUTE GENERATED PROGRAM
    # --------------------------------------------------------

    def execute_program(self):

        settings = self.program["settings"]

        for instruction, args in self.instructions:

            if instruction == "SPACE":

                if args[0].upper() == "BITS":

                    bits = int(args[1])

                    self.space = RubberSpace(bits)

            elif instruction == "TRIGRAM":

                if args[0].upper() != "ORDER" or int(args[1]) != 3:

                    raise ValueError(
                        "Rubber requires trigram order 3."
                    )

            elif instruction == "MATH":

                if args[0].upper() == "LEVEL1":

                    settings["scale"] = float(
                        args[2]
                    )

                elif args[0].upper() == "LEVEL2":

                    settings["mode"] = args[1].upper()

                    index = args.index("WEIGHT")

                    settings["weight"] = float(
                        args[index + 1]
                    )

            elif instruction == "PROB":

                key = args[0].upper()

                if key == "TEMP":

                    settings["temperature"] = float(
                        args[1]
                    )

                elif key == "SPATIAL":

                    settings["spatial"] = float(
                        args[1]
                    )

                elif key == "SMOOTH":

                    settings["smoothing"] = float(
                        args[1]
                    )

            elif instruction == "EXECUTE":

                operation = args[0].upper()

                if operation == "ENCODE":

                    self.spatial_dataset = (
                        self.space.encode_dataset(
                            self.dataset
                        )
                    )

                elif operation == "LEARN":

                    self.language.learn(
                        self.dataset
                    )

                elif operation == "SEQUENCE":

                    self.execute_dataset_sequence()

                elif operation == "READY":

                    self.print_program_status()

        self.math = RubberMath(
            mode=settings["mode"],
            scale=settings["scale"],
            weight=settings["weight"]
        )

    # --------------------------------------------------------
    # SEQUENTIAL DATASET PROCESSING
    # --------------------------------------------------------

    def execute_dataset_sequence(self):

        previous = None

        state = 0.0

        for item in self.spatial_dataset:

            if previous is None:

                affinity = 0.5

            else:

                affinity = self.space.affinity(
                    item["xyz"],
                    previous
                )

            value = (
                hash_value(item["token"])
                / 4294967295.0
            )

            local = self.math.level_one(
                value,
                item["position"],
                affinity,
                0.0
            )

            state = self.math.level_two(
                local,
                state
            )

            previous = item["xyz"]

        self.dataset_state = state

    # --------------------------------------------------------
    # PROMPT INPUT
    # --------------------------------------------------------

    def set_prompt(self, prompt):

        self.prompt = prompt

        tokens = tokenize(prompt)

        value = 0.0

        for i, token in enumerate(tokens):

            value += (
                hash_value(token)
                / 4294967295.0
            ) / (i + 1)

        self.prompt_state = math.tanh(value)

        print("[RUBBER] Prompt initialized.")

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    def generate(self, length=100):

        if not self.ready:

            raise RuntimeError(
                "Load a dataset first."
            )

        settings = self.program["settings"]

        prompt_tokens = tokenize(
            self.prompt
        )

        context = prompt_tokens[-3:]

        if len(context) < 3:

            context = (
                self.dataset[:3] + context
            )[-3:]

        state = (
            self.dataset_state + self.prompt_state
        ) / 2.0

        output = []

        previous_position = None

        for step in range(length):

            item = self.spatial_dataset[
                step % len(self.spatial_dataset)
            ]

            if previous_position is None:

                affinity = 0.5

            else:

                affinity = self.space.affinity(
                    item["xyz"],
                    previous_position
                )

            previous_position = item["xyz"]

            value = (
                hash_value(item["token"])
                / 4294967295.0
            )

            local = self.math.level_one(
                value,
                item["position"],
                affinity,
                self.prompt_state
            )

            state = self.math.level_two(
                local,
                state
            )

            token = self.language.predict(
                context,
                numeric=state,
                spatial=settings["spatial"],
                prompt_bias={},
                temperature=settings["temperature"]
            )

            if token is None:
                break

            output.append(token)

            # Shift trigram context.
            context.append(token)

            context = context[-3:]

            # Feed generated token into numerical state.
            generated_value = (
                hash_value(token)
                / 4294967295.0
            )

            local = self.math.level_one(
                generated_value,
                step,
                0.5,
                self.prompt_state
            )

            state = self.math.level_two(
                local,
                state
            )

        result = " ".join(output)

        result = re.sub(
            r"\s+([.,!?;:])",
            r"\1",
            result
        )

        return result

    # --------------------------------------------------------
    # PROGRAM STATUS
    # --------------------------------------------------------

    def print_program_status(self):

        print(
            "[RUBBER] Generated program:",
            self.program["id"]
        )

        print(
            "[RUBBER] Vocabulary:",
            self.analysis["vocabulary"]
        )

        print(
            "[RUBBER] Trigrams:",
            len(self.language.trigrams)
        )

        print(
            "[RUBBER] Spatial bits:",
            self.space.bits
        )

    def status(self):

        return {
            "version": VERSION,
            "ready": self.ready,
            "program_id": (
                self.program["id"]
                if self.program else None
            ),
            "dataset_tokens": len(self.dataset),
            "vocabulary": len(self.language.vocabulary),
            "trigrams": len(self.language.trigrams),
            "dataset_state": getattr(
                self,
                "dataset_state",
                0.0
            ),
            "program_settings": (
                self.program["settings"]
                if self.program else {}
            )
        }

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    def save(self, filename):

        if not self.ready:

            raise RuntimeError(
                "No trained dataset."
            )

        data = {
            "version": VERSION,
            "dataset": self.dataset,
            "program": self.program,
            "settings": self.program["settings"],
            "unigrams": dict(
                self.language.unigrams
            ),
            "bigrams": [
                {
                    "context": context,
                    "counts": dict(counts)
                }
                for context, counts
                in self.language.bigrams.items()
            ],
            "trigrams": [
                {
                    "context": list(context),
                    "counts": dict(counts)
                }
                for context, counts
                in self.language.trigrams.items()
            ]
        }

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False
            )

        print("[RUBBER] Model saved.")

    # --------------------------------------------------------
    # LOAD MODEL WITHOUT REGENERATING PROGRAM
    # --------------------------------------------------------

    def load_model(self, filename):

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        self.dataset = data["dataset"]

        self.program = data["program"]

        self.analysis = self.analyser.analyse(
            self.dataset
        )

        self.instructions = self.compiler.compile(
            self.program["source"]
        )

        self.space = RubberSpace(
            self.program["settings"]["bits"]
        )

        self.spatial_dataset = (
            self.space.encode_dataset(
                self.dataset
            )
        )

        self.language = RubberTrigrams(
            smoothing=self.program["settings"]["smoothing"]
        )

        self.language.unigrams = Counter(
            data["unigrams"]
        )

        self.language.vocabulary = set(
            self.language.unigrams
        )

        for item in data["bigrams"]:

            self.language.bigrams[
                item["context"]
            ] = Counter(item["counts"])

        for item in data["trigrams"]:

            self.language.trigrams[
                tuple(item["context"])
            ] = Counter(item["counts"])

        self.math = RubberMath(
            mode=self.program["settings"]["mode"],
            scale=self.program["settings"]["scale"],
            weight=self.program["settings"]["weight"]
        )

        self.execute_dataset_sequence()

        self.ready = True

        print("[RUBBER] Existing program restored.")


# ============================================================
# INTERACTIVE TERMINAL
# ============================================================

def interactive():

    ai = RubberAI()

    print("""
========================================
             RUBBER AI 2.0
========================================

DATA file.txt       Load dataset and generate new RBC
PROMPT text         Set prompt
GENERATE 100        Generate text
STATUS              Inspect current program
PROGRAM             Display generated source
SAVE model.json     Save model
LOAD model.json     Restore model
EXIT                Quit

A new program is generated ONLY when DATA is executed.
""")

    while True:

        try:

            line = input("\nRUBBER> ").strip()

            if not line:
                continue

            parts = shlex.split(line)

            command = parts[0].upper()

            args = parts[1:]

            if command == "EXIT":

                break

            elif command == "DATA":

                ai.load_dataset(args[0])

            elif command == "PROMPT":

                ai.set_prompt(
                    " ".join(args)
                )

            elif command == "GENERATE":

                length = int(
                    args[0]
                ) if args else 100

                print(
                    "\n" + ai.generate(length) + "\n"
                )

            elif command == "STATUS":

                print(
                    json.dumps(
                        ai.status(),
                        indent=2
                    )
                )

            elif command == "PROGRAM":

                if ai.program:

                    print(
                        ai.program["source"]
                    )

                else:

                    print("No generated program.")

            elif command == "SAVE":

                ai.save(args[0])

            elif command == "LOAD":

                ai.load_model(args[0])

            else:

                print("Unknown command.")

        except Exception as error:

            print("[ERROR]", error)


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Self-generating Rubber Code AI"
    )

    parser.add_argument(
        "--interactive",
        action="store_true"
    )

    parser.add_argument(
        "--dataset"
    )

    parser.add_argument(
        "--prompt",
        default="continue"
    )

    parser.add_argument(
        "--length",
        type=int,
        default=100
    )

    parser.add_argument(
        "--save"
    )

    parser.add_argument(
        "--load-model"
    )

    args = parser.parse_args()

    ai = RubberAI()

    if args.interactive or (
        not args.dataset and not args.load_model
    ):

        interactive()

        return

    if args.load_model:

        ai.load_model(args.load_model)

    else:

        ai.load_dataset(args.dataset)

    ai.set_prompt(args.prompt)

    print(
        "\n" + ai.generate(args.length) + "\n"
    )

    if args.save:

        ai.save(args.save)


if __name__ == "__main__":

    main()
