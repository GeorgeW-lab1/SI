import numpy as np
from collections import deque

KB_LIMIT = 99999
PRIMARY_FRAC = 1.75
LL_FLOOR = 1e-2


def V(lam):
    """Element-wise real fifth-root vector operator."""
    lam = np.asarray(lam, dtype=float)
    return np.sign(lam) * np.abs(lam) ** 0.2


# ============================================================
# 1. OPTOELECTRONIC RESERVOIR
# ============================================================

class WordOptoelectronicLLM:
    def __init__(self, corpus, extra_corpora=None, dt=0.001, seed=42):
        self.rng = np.random.default_rng(seed)
        self.dt = float(dt)

        self.raw = [corpus[:KB_LIMIT]]
        self.raw += [x[:KB_LIMIT] for x in (extra_corpora or [])]

        self.primary = self.tokenize(self.raw[0])
        self.extras = [self.tokenize(x) for x in self.raw[1:]]
        self.extras = [tokens for tokens in self.extras if tokens]

        datasets = [self.primary] + self.extras
        self.vocab = sorted(set(token for data in datasets for token in data))
        if not self.vocab:
            raise ValueError(
                "The corpus must contain at least three whitespace-separated words."
            )

        self.vocab_size = len(self.vocab)
        self.tok_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.id_to_tok = dict(enumerate(self.vocab))

        self.roi = self.rng.uniform(0.2, 4.8, (self.vocab_size, 2))

        self.vcc = 5.0
        self.high = 3.0
        self.low = 1.5
        self.tau = 0.0047
        self.vth = 0.7
        self.alpha = 0.25

        self.state_dim = 8
        self.split = 6
        self.W_main = np.zeros((6, self.vocab_size))
        self.W_aux = np.zeros((2, self.vocab_size))
        self.reset()

    @staticmethod
    def tokenize(text):
        words = text.split()
        return [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]

    def reset(self):
        self.driver = 0.0
        self.vrc = 0.0
        self.top = 0.0
        self.bot = 0.0
        self.probe = 0.0

        delay_length = max(1, int(0.010 / self.dt))
        self.delay = deque([0] * delay_length, maxlen=delay_length)

        self.dff = 0
        self.clock = 0
        self.steps = 0

    def step(self, a, b, ext=0):
        self.steps += 1

        # Hysteretic driver.
        if a >= self.high:
            self.driver = 5.0
        elif a <= self.low:
            self.driver = 0.0

        # RC reservoir response.
        self.vrc += (self.driver - self.vrc) * self.dt / self.tau

        if self.vrc > self.vth:
            self.vrc -= 0.65 * (self.vrc - self.vth)
            node = self.vth + (self.vrc - self.vth) * 0.2
        else:
            node = self.vrc

        self.top = 0.25 * node + 0.75 * self.top
        self.bot = 0.25 * b + 0.75 * self.bot

        ct = int(self.top > 0.5)
        cb = int(self.bot > 0.5)
        gate = ct & cb

        delayed = self.delay[0]
        self.delay.append(gate)

        clk = cb | int(bool(ext))
        if clk and not self.clock:
            self.dff = delayed
        self.clock = clk

        # Keep bitwise logic integer-only; convert to float afterward.
        raw_top_logic = int(a > 2.5)
        sec_and = raw_top_logic & int(delayed)
        probability = np.clip(
            sec_and * 0.8 + self.rng.normal(0.0, 0.15), 0.0, 1.0
        )
        self.probe = 0.7 * self.probe + 0.3 * probability

        state = np.array([
            self.driver / 5.0,
            self.vrc / 0.7,
            self.top / 5.0,
            self.bot / 5.0,
            gate,
            delayed,
            self.dff,
            self.probe * (1.0 if self.steps % 2 == 0 else 0.5),
        ], dtype=float)

        # Apply the fifth-root vector operator to every reservoir state.
        return V(state)

    def encode(self, tokens, ticks=20):
        output = []

        for token in tokens:
            token_id = self.tok_to_id.get(token, 0)
            a, b = self.roi[token_id]

            state = np.zeros(self.state_dim)
            for tick in range(ticks):
                state = self.step(
                    a + self.rng.normal(0.0, 0.05),
                    b + self.rng.normal(0.0, 0.05),
                    int(tick == ticks // 2),
                )
            output.append(state)

        return np.asarray(output, dtype=float).reshape(-1, self.state_dim)

    def ridge(self, datasets, state_slice, alpha):
        dimension = state_slice.stop - state_slice.start
        gram = np.zeros((dimension, dimension))
        cross = np.zeros((dimension, self.vocab_size))

        for tokens in datasets:
            if len(tokens) < 2:
                continue

            self.reset()
            X = self.encode(tokens[:-1])[:, state_slice]

            Y = np.zeros((len(tokens) - 1, self.vocab_size))
            for i, token in enumerate(tokens[1:]):
                Y[i, self.tok_to_id[token]] = 1.0

            gram += X.T @ X
            cross += X.T @ Y

        self.reset()
        return np.linalg.solve(
            gram + alpha * np.eye(dimension),
            cross
        )

    def train(self, alpha=0.01):
        self.W_main = self.ridge(
            [self.primary], slice(0, 6), alpha
        )
        self.W_aux = self.ridge(
            self.extras or [self.primary], slice(6, 8), alpha
        )
        return self

    def probs(self, state, temp=0.4):
        state = np.atleast_2d(state)

        logits = (
            state[:, :6] @ self.W_main
            + state[:, 6:] @ self.W_aux
        )

        scale = np.maximum(
            0.1,
            temp * (0.8 + 0.4 * state[:, 7, None])
        )
        logits = (logits - logits.max(axis=1, keepdims=True)) / scale

        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def generate(self, prompt, num_words=250, temperature=0.8):
        tokens = self.tokenize(prompt)
        if not tokens:
            return prompt

        self.reset()
        state = self.encode(tokens)[-1]

        for _ in range(num_words):
            probabilities = self.probs(state, temperature)[0]
            next_token = self.id_to_tok[
                self.rng.choice(self.vocab_size, p=probabilities)
            ]
            tokens.append(next_token)
            state = self.encode([next_token])[-1]

        return " ".join(tokens)


# ============================================================
# 2. NATURALNESS ASSESSMENT AGAINST ORIGINAL DATA
# ============================================================

class NaturalnessAssessor:
    weights = (0.65, 0.25, 0.20, 0.20)

    def __init__(self, text):
        words = text[:KB_LIMIT].split()
        if len(words) < 6:
            raise ValueError(
                "Naturalness assessment needs at least six words of source text."
            )

        cut = max(3, int(len(words) * 0.9))
        cut = min(cut, len(words) - 3)

        self.train = " ".join(words[:cut])
        self.hold = WordOptoelectronicLLM.tokenize(" ".join(words[cut:]))

        all_tokens = WordOptoelectronicLLM.tokenize(" ".join(words))
        self.seen = set(zip(all_tokens[:-1], all_tokens[1:]))

        training_tokens = WordOptoelectronicLLM.tokenize(self.train)
        self.start = training_tokens[-1] if training_tokens else None

    def assess(self, model, n=80, temp=0.4, seed=1234):
        if not model.vocab:
            return {
                "likelihood": 0.0,
                "seen": 0.0,
                "coherence": 0.0,
                "diversity": 0.0,
                "composite": 0.0,
            }

        model.rng = np.random.default_rng(seed)
        model.reset()

        X = (
            model.encode(self.hold[:-1])
            if len(self.hold) > 1
            else np.empty((0, 8))
        )

        log_likelihoods = []
        for i, state in enumerate(X):
            probabilities = model.probs(state, temp)[0]
            target = model.tok_to_id.get(self.hold[i + 1])
            target_probability = (
                probabilities[target] if target is not None else 0.0
            )
            log_likelihoods.append(
                np.log(max(target_probability, LL_FLOOR))
            )

        likelihood = (
            np.clip(
                1.0 + np.mean(log_likelihoods) / np.log(1.0 / LL_FLOOR),
                0.0,
                1.0,
            )
            if log_likelihoods else 0.0
        )

        model.rng = np.random.default_rng(seed + 1)
        model.reset()

        previous = (
            self.start
            if self.start in model.tok_to_id
            else model.vocab[0]
        )
        state = model.encode([previous])[-1]

        generated = []
        seen_count = 0
        coherence_count = 0

        for _ in range(n):
            probabilities = model.probs(state, temp)[0]
            next_token = model.id_to_tok[
                model.rng.choice(model.vocab_size, p=probabilities)
            ]

            seen_count += (previous, next_token) in self.seen
            coherence_count += (
                next_token.split()[:2] == previous.split()[1:]
            )

            generated.append(next_token)
            previous = next_token
            state = model.encode([next_token])[-1]

        model.reset()

        diversity = len(set(generated)) / max(1, n)
        seen_score = seen_count / max(1, n)
        coherence = coherence_count / max(1, n)

        a, b, c, d = self.weights
        composite = (
            a * likelihood
            + b * seen_score
            + c * coherence
            + d * diversity
        )

        return {
            "likelihood": float(likelihood),
            "seen": float(seen_score),
            "coherence": float(coherence),
            "diversity": float(diversity),
            "composite": float(composite),
        }


# ============================================================
# 3. EXECUTION
# ============================================================

def main():
    filename = input("Primary filename: ").strip()

    try:
        with open(filename, "r", encoding="utf-8") as file:
            primary = file.read()[:KB_LIMIT]
    except OSError as error:
        print(f"Could not open file: {error}")
        return

    if len(primary.split()) < 6:
        print("Please provide a source file containing at least six words.")
        return

    print("\\n--- TRAINING ---")
    model = WordOptoelectronicLLM(primary, [primary]).train()

    print("\\n--- NATURALNESS ASSESSMENT ---")
    assessor = NaturalnessAssessor(primary)
    scores = assessor.assess(model)
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")

    print("\\n--- MODEL READY ---")
    print("Vector operator: V(lambda) = sign(lambda) * abs(lambda)**(1/5)")
    print("Type /quit to stop.")

    while True:
        try:
            prompt = input("\\nUSER: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\\nExiting.")
            break

        if prompt.lower() in {"/quit", "quit", "exit"}:
            break
        if not prompt:
            continue

        print("MODEL:", model.generate(prompt, 250, 10.0))


if __name__ == "__main__":
    main()
