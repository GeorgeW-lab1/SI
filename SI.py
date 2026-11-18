import numpy as np
import re
from collections import Counter, deque

KB_LIMIT = 99999
PRIMARY_FRAC = 0.75
LL_FLOOR = 1e-6


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
# 2. H PROPOSITIONS: SYNTACTIC TRANSFORMATION
# ============================================================

def h1(text):
    return text.lower()


def h2(text):
    return re.sub(r"[^\w\s']", " ", text)


def h3(text):
    words = text.split()
    return " ".join(
        word for i, word in enumerate(words)
        if i == 0 or word != words[i - 1]
    )


def h4(text):
    words = text.split()
    output = []
    i = 0

    while i < len(words):
        if (
            i + 3 < len(words)
            and words[i:i + 2] == words[i + 2:i + 4]
        ):
            output += words[i:i + 2]
            i += 4
        else:
            output.append(words[i])
            i += 1

    return " ".join(output)


def h5(text):
    words = text.split()
    counts = Counter(words)
    repeated = [word for word in words if counts[word] > 1]
    return " ".join(repeated) if repeated else text


def h6(text):
    words = text.split()
    counts = Counter(words)
    number_to_remove = max(1, int(len(counts) * 0.02))
    common = {word for word, _ in counts.most_common(number_to_remove)}
    remaining = [word for word in words if word not in common]
    return " ".join(remaining) if remaining else text


def h7(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    unique = dict.fromkeys(
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    )
    return " ".join(unique)


H = {
    "H1": h1,
    "H2": h2,
    "H3": h3,
    "H4": h4,
    "H5": h5,
    "H6": h6,
    "H7": h7,
}


def normal(text):
    return " ".join(text.split())


def simmer(fn, text, max_iter=100):
    """Iterate a transformation until stable or resolve a detected cycle."""
    current = normal(text)
    seen = {current: 0}
    history = [current]

    for iteration in range(max_iter):
        next_text = normal(fn(current))

        if next_text == current:
            return current, iteration + 1, "SETTLED"

        if next_text in seen:
            cycle = history[seen[next_text]:]
            settled = min(cycle, key=lambda item: (len(item), item))
            return settled, iteration + 1, "CYCLE_RESOLVED"

        seen[next_text] = len(history)
        history.append(next_text)
        current = next_text

    return current, max_iter, "ITERATION_LIMIT"


def build_h():
    transforms = dict(H)

    for name_a, fn_a in H.items():
        for name_b, fn_b in H.items():
            if name_a != name_b:
                transforms[f"{name_a}>{name_b}"] = (
                    lambda text, fa=fn_a, fb=fn_b: fb(fa(text))
                )

    return transforms


def syntactic_analysis(text):
    results = {}

    for name, transform in build_h().items():
        settled, iterations, status = simmer(transform, text)
        results[name] = {
            "text": settled,
            "iterations": iterations,
            "status": status,
        }

    return results


# ============================================================
# 3. NATURALNESS ASSESSMENT AGAINST ORIGINAL DATA
# ============================================================

class NaturalnessAssessor:
    weights = (0.35, 0.25, 0.20, 0.20)

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
# 4. CONTINUOUS SYNTACTIC SIMMERING AND LEARNING
# ============================================================

def continuous_train(
    original,
    extra_texts=None,
    max_rounds=100,
    explore=0.5,
    tau=0.3,
    ema=0.3,
    min_keep=0.5,
    seed=0,
):
    assessor = NaturalnessAssessor(original)
    rng = np.random.default_rng(seed)

    primary = normal(assessor.train)
    extras = [normal(text) for text in (extra_texts or [])]

    names = list(H)
    reward = {name: 0.0 for name in names}
    tries = {name: 0 for name in names}

    rejected = set()
    history = []

    def build(primary_text, extra_text):
        model = WordOptoelectronicLLM(
            primary_text, extra_text
        ).train()
        return model, assessor.assess(model)

    best_model, score = build(primary, extras)
    base_length = max(1, len(primary.split()))

    for round_index in range(max_rounds):
        logits = np.array([
            reward[name] + explore / np.sqrt(1 + tries[name])
            for name in names
        ])

        probabilities = np.exp(
            (logits - logits.max()) / max(tau, 1e-6)
        )
        probabilities /= probabilities.sum()

        chosen_h = names[rng.choice(len(names), p=probabilities)]
        tries[chosen_h] += 1

        new_primary = simmer(H[chosen_h], primary)[0]
        new_extras = [
            simmer(H[chosen_h], text)[0]
            for text in extras
        ]

        if new_primary == primary and new_extras == extras:
            continue

        key = (chosen_h, new_primary, tuple(new_extras))
        if key in rejected:
            continue

        retained = len(new_primary.split()) / base_length
        if retained < min_keep:
            reward[chosen_h] = (
                (1.0 - ema) * reward[chosen_h] - ema
            )
            rejected.add(key)
            continue

        model, new_score = build(new_primary, new_extras)
        delta = new_score["composite"] - score["composite"]
        accepted = delta > 1e-4

        reward[chosen_h] = (
            (1.0 - ema) * reward[chosen_h]
            + ema * np.tanh(20.0 * delta)
        )

        history.append({
            "round": round_index + 1,
            "H": chosen_h,
            "accepted": accepted,
            "delta": float(delta),
            "score": new_score,
        })

        if accepted:
            primary = new_primary
            extras = new_extras
            best_model = model
            score = new_score
            base_length = max(1, len(primary.split()))
        else:
            rejected.add(key)

        print(
            f"{round_index + 1}: {chosen_h} | "
            f"{delta:+.4f} | "
            f"{'ACCEPT' if accepted else 'REJECT'}"
        )

    return best_model, history, reward


# ============================================================
# 5. EXECUTION
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

    print("\n--- SYNTACTIC SIMMERING ---")
    for name, result in syntactic_analysis(primary).items():
        print(
            f"{name:8s} {result['status']:16s} "
            f"{result['iterations']} iterations"
        )

    print("\n--- CONTINUOUS TRAINING ---")
    model, history, rewards = continuous_train(
        primary,
        [primary],
        max_rounds=4,
    )

    print("\n--- MODEL READY ---")
    print("Vector operator: V(lambda) = sign(lambda) * abs(lambda)**(1/5)")
    print("Enter a prompt using words found in the training corpus.")
    print("Type /quit to stop.")

    while True:
        try:
            prompt = input("\nUSER: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if prompt.lower() in {"/quit", "quit", "exit"}:
            break

        if not prompt:
            continue

        print("MODEL:", model.generate(prompt, 250, 0.8))


if __name__ == "__main__":
    main()
