#!/usr/bin/env python3
"""Minimal corpus generator with an iterative text deconstructor."""


import random
import re
from collections import Counter, defaultdict



TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Generator:
    def __init__(self):
        self.words = []
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)

    def load_corpus(self, path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        self.words = TOKEN_RE.findall(text)
        self.bigram.clear()
        self.trigram.clear()
        for i in range(len(self.words) - 1):
            self.bigram[self.words[i]][self.words[i + 1]] += 1
        for i in range(len(self.words) - 2):
            self.trigram[tuple(self.words[i:i + 2])][self.words[i + 2]] += 1
        print(f"Loaded {len(self.words)} tokens.")


    @staticmethod
    def detokenize(tokens):
        text = " ".join(tokens)
        return re.sub(r"\s+([.,!?;:])", r"\1", text)


    def deconstruct(self, text):
        """Collapse immediate token repeats and repeated adjacent phrase loops."""
        tokens = TOKEN_RE.findall(text)
        if not tokens:
            return ""

        # Collapse runs of identical tokens (e.g. "word word word").
        collapsed = []
        for token in tokens:
            if not collapsed or token.lower() != collapsed[-1].lower():
                collapsed.append(token)

        # Collapse repeated adjacent phrases, from longest to shortest.
        changed = True
        while changed:
            changed = False
            for size in range(min(8, len(collapsed) // 2), 1, -1):
                i = 0
                while i + 2 * size <= len(collapsed):
                    left = collapsed[i:i + size]
                    right = collapsed[i + size:i + 2 * size]
                    if [x.lower() for x in left] == [x.lower() for x in right]:
                        del collapsed[i + size:i + 2 * size]
                        changed = True
                    else:
                        i += 1
                if changed:
                    break
        return self.detokenize(collapsed)


    def _sample(self, context):
        options = self.trigram.get(tuple(context[-2:]))
        if not options:
            options = self.bigram.get(context[-1], Counter()) if context else Counter()
        if not options:
            options = Counter(self.words)
        return random.choices(list(options.keys()), weights=list(options.values()), k=1)[0]


    def generate(self, prompt, length=100, deconstruction_rounds=1):
        """Generate, deconstruct (replacing the text), then regenerate from its reduced context."""
        if not self.words:
            raise RuntimeError("Load a corpus first.")
        if length < 0:
            raise ValueError("Token length must be non-negative.")
        if deconstruction_rounds < 0:
            raise ValueError("Deconstruction rounds must be non-negative.")

        current = prompt.strip()
        context = TOKEN_RE.findall(current)[-2:]
        if not context:
            context = [random.choice(self.words)]

        # Initial generation.
        generated = []
        for _ in range(length):
            token = self._sample(context)
            generated.append(token)
            context = (context + [token])[-2:]
        current = self.detokenize(TOKEN_RE.findall(current) + generated)

        # Each cycle replaces current text with its deconstructed form,
        # then generates a fresh continuation from the reduced context.
        for _ in range(deconstruction_rounds):
            current = self.deconstruct(current)  # replacement, not a side analysis
            context = TOKEN_RE.findall(current)[-2:]
            if not context:
                context = [random.choice(self.words)]
            continuation = []
            for _ in range(length):
                token = self._sample(context)
                continuation.append(token)
                context = (context + [token])[-2:]
            current = self.detokenize(TOKEN_RE.findall(current) + continuation)

        return current



def main():
    model = Generator()
    path = input("Corpus file (.txt): ").strip().strip('"')
    model.load_corpus(path)


    while True:
        prompt = input("\nPrompt (or EXIT): ").strip()
        if prompt.upper() == "EXIT":
            break
        if not prompt:
            continue
        try:
            length = int(input("Tokens per pass: "))
            rounds = int(input("Deconstruction/regeneration rounds (0 = off): "))
            print("\n" + model.generate(prompt, length, rounds))
        except ValueError as e:
            print("Invalid input:", e)
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
