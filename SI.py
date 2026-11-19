#!/usr/bin/env python3
"""
RUBBER AI - FIXED SKELETON + REINFORCEMENT LEARNING
===================================================

Uses the settings from:
    rubber_programs/rubber_0d25c38b2853.rbc

No generated programs. No compiler. No factory framework.

Pipeline:
    filename -> tokenize -> spatial encode -> trigram learn
             -> sequential state -> RL train -> prompt -> generate
"""

import os
import re
import json
import math
import shlex
import random
import hashlib
import secrets

from collections import Counter, defaultdict


VERSION = "3.1"

# ============================================================
# FIXED PROGRAM SKELETON
# ============================================================

PROGRAM_ID = "0d25c38b2853"

SETTINGS = {
    "space_bits": 14,
    "space_dimensions": 3,
    "trigram_order": 3,

    "math_level1_scale": 1.1182,
    "math_level2_mode": "AVERAGE",
    "math_level2_weight": 0.5806,

    "probability_smoothing": 0.054,
    "probability_temperature": 0.5196,
    "probability_spatial": 0.2461,

    "rl_rate": 0.08,
    "rl_temperature": 0.8,
    "rl_exploration": 0.08,
    "rl_episodes": 40,
    "rl_length": 24
}


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

    def __init__(self, bits=SETTINGS["space_bits"]):
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
            self.encode(token, index)
            for index, token in enumerate(tokens)
        ]

        return self.positions

    def affinity(self, a, b):
        distance = math.sqrt(
            sum(
                (a[index] - b[index]) ** 2
                for index in range(3)
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

    def __init__(self):
        self.mode = SETTINGS["math_level2_mode"]
        self.scale = SETTINGS["math_level1_scale"]
        self.weight = SETTINGS["math_level2_weight"]

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

    def __init__(self):
        self.trigrams = defaultdict(Counter)
        self.bigrams = defaultdict(Counter)
        self.unigrams = Counter()
        self.vocabulary = set()
        self.smoothing = SETTINGS[
            "probability_smoothing"
        ]

    def learn(self, tokens):
        self.vocabulary.update(tokens)
        self.unigrams.update(tokens)

        for index in range(len(tokens) - 1):
            self.bigrams[
                tokens[index]
            ][tokens[index + 1]] += 1

        for index in range(len(tokens) - 3):
            context = tuple(tokens[index:index + 3])
            target = tokens[index + 3]

            self.trigrams[context][target] += 1

    def observed_transitions(self):
        transitions = set()

        for context, counter in self.trigrams.items():
            for target in counter:
                transitions.add(
                    context + (target,)
                )

        return transitions

    def candidates(self, context):
        context = tuple(context[-3:])

        if context in self.trigrams:
            return self.trigrams[context], "TRIGRAM"

        if context and context[-1] in self.bigrams:
            return self.bigrams[context[-1]], "BIGRAM"

        if self.unigrams:
            return self.unigrams, "UNIGRAM"

        return Counter(), "NONE"


# ============================================================
# REINFORCEMENT-LEARNING POLICY
# ============================================================

class RubberRLPolicy:
    """
    REINFORCE-style policy over next-token choices.

    preference[context][token] is updated using:

        preference += learning_rate * (reward - baseline)
    """

    def __init__(self):
        self.preferences = defaultdict(dict)

        self.learning_rate = SETTINGS["rl_rate"]
        self.temperature = SETTINGS["rl_temperature"]
        self.exploration = SETTINGS["rl_exploration"]
        self.l2 = 0.001

        self.episodes = 0
        self.updates = 0
        self.total_reward = 0.0

    def choose(self, context, candidates, rng):
        if not candidates:
            return None

        context = tuple(context[-3:])
        preferences = self.preferences[context]

        for token in candidates:
            preferences.setdefault(token, 0.0)

        if rng.random() < self.exploration:
            return rng.choice(list(candidates))

        maximum = max(
            preferences[token]
            for token in candidates
        )

        tokens = []
        weights = []

        for token in candidates:
            preference = preferences[token]

            weight = math.exp(
                (preference - maximum)
                / self.temperature
            )

            tokens.append(token)
            weights.append(weight)

        return rng.choices(
            tokens,
            weights=weights,
            k=1
        )[0]

    def update(self, context, action, reward, baseline):
        if action is None:
            return

        context = tuple(context[-3:])
        preferences = self.preferences[context]

        preferences.setdefault(action, 0.0)

        advantage = reward - baseline

        preferences[action] += (
            self.learning_rate * advantage
        )

        for token in preferences:
            preferences[token] *= (1.0 - self.l2)

        self.updates += 1
        self.total_reward += reward

    def train_episode(self, episode):
        self.episodes += 1

        rewards = [
            step["reward"]
            for step in episode
        ]

        if not rewards:
            return 0.0

        baseline = sum(rewards) / len(rewards)

        for step in episode:
            self.update(
                step["context"],
                step["action"],
                step["reward"],
                baseline
            )

        return baseline

    def export(self):
        return {
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "exploration": self.exploration,
            "l2": self.l2,
            "episodes": self.episodes,
            "updates": self.updates,
            "total_reward": self.total_reward,
            "preferences": {
                " ".join(context): dict(values)
                for context, values
                in self.preferences.items()
            }
        }

    def import_policy(self, data):
        self.learning_rate = data.get(
            "learning_rate",
            self.learning_rate
        )

        self.temperature = data.get(
            "temperature",
            self.temperature
        )

        self.exploration = data.get(
            "exploration",
            self.exploration
        )

        self.l2 = data.get("l2", self.l2)
        self.episodes = data.get("episodes", 0)
        self.updates = data.get("updates", 0)
        self.total_reward = data.get(
            "total_reward",
            0.0
        )

        for context, values in data.get(
            "preferences",
            {}
        ).items():
            self.preferences[
                tuple(context.split())
            ] = dict(values)


# ============================================================
# RUBBER AI ENGINE
# ============================================================

class RubberAI:

    def __init__(self):
        self.space = RubberSpace()
        self.math = RubberMath()
        self.language = RubberTrigrams()
        self.policy = RubberRLPolicy()

        self.dataset = []
        self.spatial_dataset = []
        self.dataset_hash = ""

        self.dataset_state = 0.0
        self.prompt = ""
        self.prompt_state = 0.0

        self.ready = False

        self.observed_trigrams = set()
        self.observed_bigrams = set()

    # --------------------------------------------------------
    # LOAD FILE
    # --------------------------------------------------------

    def load_dataset(self, filename):
        print(f"\n[RUBBER] Loading: {filename}")

        with open(
            filename,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as file:
            text = file.read()

        self.dataset = tokenize(text)

        if len(self.dataset) < 6:
            raise ValueError(
                "Dataset must contain at least six tokens."
            )

        self.dataset_hash = digest(text)

        print(
            "[RUBBER] Tokens:",
            len(self.dataset)
        )

        print(
            "[RUBBER] Dataset hash:",
            self.dataset_hash[:16]
        )

        self.space = RubberSpace()
        self.math = RubberMath()
        self.language = RubberTrigrams()
        self.policy = RubberRLPolicy()

        self.spatial_dataset = (
            self.space.encode_dataset(
                self.dataset
            )
        )

        self.language.learn(self.dataset)

        self.observed_trigrams = (
            self.language.observed_transitions()
        )

        self.observed_bigrams = set()

        for source, targets in (
            self.language.bigrams.items()
        ):
            for target in targets:
                self.observed_bigrams.add(
                    (source, target)
                )

        self.execute_dataset_sequence()
        self.train_policy()

        self.ready = True

        print("[RUBBER] Dataset ready.")

    # --------------------------------------------------------
    # SEQUENTIAL STATE
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
    # REINFORCEMENT LEARNING
    # --------------------------------------------------------

    def reward_step(
        self,
        context,
        action,
        generated,
        prompt_tokens
    ):
        reward = 0.0

        context = tuple(context[-3:])
        transition = context + (action,)

        if transition in self.observed_trigrams:
            reward += 1.0

        if (
            context[-1],
            action
        ) in self.observed_bigrams:
            reward += 0.35

        if action in self.language.vocabulary:
            reward += 0.10

        if action in prompt_tokens:
            reward += 0.25

        recent = generated[-8:]

        if action in recent:
            reward -= 0.35 * (
                recent.count(action)
                / max(len(recent), 1)
            )

        recent_window = generated[-12:]

        unique_recent = len(set(recent_window))

        reward += 0.10 * (
            unique_recent
            / max(len(recent_window), 1)
        )

        return reward

    def run_generation_episode(
        self,
        seed_tokens,
        length,
        prompt_tokens,
        rng
    ):
        context = list(seed_tokens[-3:])

        if len(context) < 3:
            context = (
                self.dataset[:3] + context
            )[-3:]

        generated = []
        episode = []

        for _ in range(length):
            candidates, mode = (
                self.language.candidates(context)
            )

            if not candidates:
                break

            action = self.policy.choose(
                context,
                list(candidates),
                rng
            )

            if action is None:
                break

            reward = self.reward_step(
                context,
                action,
                generated,
                prompt_tokens
            )

            episode.append(
                {
                    "context": tuple(context[-3:]),
                    "action": action,
                    "reward": reward
                }
            )

            generated.append(action)

            context.append(action)
            context = context[-3:]

        return generated, episode

    def train_policy(
        self,
        episodes=SETTINGS["rl_episodes"],
        length=SETTINGS["rl_length"]
    ):
        if not self.dataset:
            raise RuntimeError(
                "Load a dataset first."
            )

        rng = random.Random(
            secrets.randbits(64)
        )

        print(
            f"\n[RUBBER] RL training: "
            f"{episodes} episodes x {length} tokens"
        )

        total_reward = 0.0

        for _ in range(episodes):
            start = rng.randrange(
                max(1, len(self.dataset) - 3)
            )

            seed_tokens = self.dataset[
                start:start + 3
            ]

            generated, episode = (
                self.run_generation_episode(
                    seed_tokens,
                    length,
                    [],
                    rng
                )
            )

            average_reward = (
                self.policy.train_episode(episode)
            )

            total_reward += average_reward

        print(
            "[RUBBER] RL training complete. "
            f"Average reward: "
            f"{total_reward / max(episodes, 1):.4f}"
        )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    def set_prompt(self, prompt):
        self.prompt = prompt

        tokens = tokenize(prompt)
        value = 0.0

        for index, token in enumerate(tokens):
            value += (
                hash_value(token)
                / 4294967295.0
            ) / (index + 1)

        self.prompt_state = math.tanh(value)

        print("[RUBBER] Prompt initialized.")

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    def generate(
        self,
        length=100,
        learn_online=True
    ):
        if not self.ready:
            raise RuntimeError(
                "Load a dataset first."
            )

        prompt_tokens = tokenize(self.prompt)

        context = prompt_tokens[-3:]

        if len(context) < 3:
            context = (
                self.dataset[:3] + context
            )[-3:]

        state = (
            self.dataset_state + self.prompt_state
        ) / 2.0

        output = []
        episode = []

        previous_position = None

        rng = random.Random(
            secrets.randbits(64)
        )

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

            candidates, mode = (
                self.language.candidates(context)
            )

            if not candidates:
                break

            action = self.policy.choose(
                context,
                list(candidates),
                rng
            )

            if action is None:
                break

            reward = self.reward_step(
                context,
                action,
                output,
                prompt_tokens
            )

            episode.append(
                {
                    "context": tuple(context[-3:]),
                    "action": action,
                    "reward": reward
                }
            )

            output.append(action)

            context.append(action)
            context = context[-3:]

            generated_value = (
                hash_value(action)
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

        if learn_online and episode:
            self.policy.train_episode(episode)

        result = " ".join(output)

        result = re.sub(
            r"\s+([.,!?;:])",
            r"\1",
            result
        )

        return result

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def status(self):
        return {
            "version": VERSION,
            "program_id": PROGRAM_ID,
            "ready": self.ready,
            "dataset_tokens": len(self.dataset),
            "vocabulary": len(
                self.language.vocabulary
            ),
            "trigrams": len(
                self.language.trigrams
            ),
            "dataset_state": self.dataset_state,
            "prompt_state": self.prompt_state,
            "rl_episodes": self.policy.episodes,
            "rl_updates": self.policy.updates,
            "rl_total_reward": (
                self.policy.total_reward
            ),
            "settings": SETTINGS
        }

    # --------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------

    def save(self, filename):
        if not self.ready:
            raise RuntimeError(
                "No trained dataset."
            )

        data = {
            "version": VERSION,
            "program_id": PROGRAM_ID,
            "settings": SETTINGS,
            "dataset": self.dataset,
            "dataset_hash": self.dataset_hash,
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
            ],
            "rl_policy": self.policy.export()
        }

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False
            )

        print("[RUBBER] Model saved.")

    def load_model(self, filename):
        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        self.dataset = data["dataset"]
        self.dataset_hash = data.get(
            "dataset_hash",
            ""
        )

        self.space = RubberSpace(
            SETTINGS["space_bits"]
        )

        self.spatial_dataset = (
            self.space.encode_dataset(
                self.dataset
            )
        )

        self.language = RubberTrigrams()

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

        self.observed_trigrams = (
            self.language.observed_transitions()
        )

        self.observed_bigrams = set()

        for source, targets in (
            self.language.bigrams.items()
        ):
            for target in targets:
                self.observed_bigrams.add(
                    (source, target)
                )

        self.policy = RubberRLPolicy()
        self.policy.import_policy(
            data.get("rl_policy", {})
        )

        self.execute_dataset_sequence()

        self.ready = True

        print(
            "[RUBBER] Model and RL policy restored."
        )


# ============================================================
# TERMINAL UI
# ============================================================

def print_help():
    print("""
========================================
     RUBBER AI 3.1
========================================

DATA filename.txt      Load dataset and train RL
PROMPT text            Set generation prompt
GENERATE 100           Generate 100 tokens
TRAIN 100 24           Run 100 RL episodes of 24 tokens
STATUS                 Show engine and RL status
SETTINGS               Show fixed skeleton settings
SAVE model.json        Save model and RL policy
LOAD model.json        Restore model and RL policy
HELP                   Show commands
EXIT                   Quit
========================================
""")


def interactive():
    ai = RubberAI()

    print_help()

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

            elif command == "HELP":
                print_help()

            elif command == "DATA":
                if not args:
                    print("Usage: DATA filename.txt")
                    continue

                ai.load_dataset(args[0])

            elif command == "PROMPT":
                if not args:
                    print("Usage: PROMPT your prompt text")
                    continue

                ai.set_prompt(" ".join(args))

            elif command == "GENERATE":
                length = (
                    int(args[0]) if args else 100
                )

                print(
                    "\n" + ai.generate(length) + "\n"
                )

            elif command == "TRAIN":
                episodes = (
                    int(args[0]) if args else 100
                )

                length = (
                    int(args[1])
                    if len(args) > 1
                    else SETTINGS["rl_length"]
                )

                ai.train_policy(episodes, length)

            elif command == "STATUS":
                print(
                    json.dumps(
                        ai.status(),
                        indent=2
                    )
                )

            elif command == "SETTINGS":
                print(
                    json.dumps(
                        SETTINGS,
                        indent=2
                    )
                )

            elif command == "SAVE":
                if not args:
                    print("Usage: SAVE model.json")
                    continue

                ai.save(args[0])

            elif command == "LOAD":
                if not args:
                    print("Usage: LOAD model.json")
                    continue

                ai.load_model(args[0])

            else:
                print("Unknown command.")

        except Exception as error:
            print("[ERROR]", error)


if __name__ == "__main__":
    interactive()
