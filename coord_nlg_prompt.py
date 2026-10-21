"""Character-coordinate text generator with interactive prompting.

Each character gets a coordinate (position in the character set, frequency rank),
both learned from the data. Generation uses an order-N character Markov model;
when a context was never seen, it backs off to shorter contexts and finally to
the coordinate rule: empirical frequency weighted by distance from the last char.
"""
import math
import random
from collections import Counter, defaultdict


def load_text():
    with open(input("Filename: ").strip(), encoding="utf-8") as f:
        return f.read()


class CoordModel:
    def __init__(self, text, order=4, spread=15.0):
        self.order, self.spread = order, spread
        self.freq = Counter(text)
        self.chars = sorted(self.freq)                                 # position axis
        ranked = [c for c, _ in self.freq.most_common()]               # rank axis
        self.coords = {c: (self.chars.index(c) + 1, ranked.index(c) + 1)
                       for c in self.chars}
        # n-gram tables for every context length 1..order
        self.tables = [defaultdict(Counter) for _ in range(order + 1)]
        for i in range(len(text) - 1):
            for n in range(1, order + 1):
                if i - n + 1 >= 0:
                    self.tables[n][text[i - n + 1:i + 1]][text[i + 1]] += 1

    def distance(self, a, b):
        (x1, y1), (x2, y2) = self.coords[a], self.coords[b]
        return math.hypot(x1 - x2, y1 - y2)

    def coord_fallback(self, prev):
        if prev in self.coords:
            w = [self.freq[c] * math.exp(-self.distance(prev, c) / self.spread)
                 for c in self.chars]
        else:  # prompt contained a character the data never saw
            w = [self.freq[c] for c in self.chars]
        return random.choices(self.chars, weights=w, k=1)[0]

    def next_char(self, history, temperature=0.8):
        for n in range(min(self.order, len(history)), 0, -1):      # back off
            counts = self.tables[n].get(history[-n:])
            if counts:
                chars = list(counts)
                w = [counts[c] ** (1 / temperature) for c in chars]
                return random.choices(chars, weights=w, k=1)[0]
        return self.coord_fallback(history[-1]) if history else random.choices(
            self.chars, weights=[self.freq[c] for c in self.chars], k=1)[0]

    def generate(self, prompt="", length=500, temperature=0.8):
        """Continue `prompt` with `length` new characters."""
        out = prompt
        for _ in range(length):
            out += self.next_char(out, temperature)
        return out


HELP = """Type a prompt and press Enter to generate a continuation.
  \\n in a prompt becomes a newline (e.g.  USER:\\n)
  :len N     characters to generate     (current: {length})
  :temp T    temperature, 0.1-2.0       (current: {temp})
  :order N   retrain with context size N (current: {order})
  :help      show this message
  :quit      exit"""


def main():
    text = load_text()
    model = CoordModel(text, order=6)
    length, temp = 400, 0.8
    print(f"{len(text):,} chars, {len(model.chars)} unique")
    for ch in "mM e\n":
        if ch in model.coords:
            print(repr(ch), model.coords[ch])
    print("\n" + HELP.format(length=length, temp=temp, order=model.order))

    while True:
        try:
            line = input("\nPrompt> ")
        except (EOFError, KeyboardInterrupt):
            break
        if line.startswith(":"):
            cmd, _, arg = line[1:].partition(" ")
            try:
                if cmd in ("quit", "q", "exit"):
                    break
                elif cmd == "len":
                    length = max(1, int(arg))
                elif cmd == "temp":
                    temp = min(max(float(arg), 0.05), 5.0)
                elif cmd == "order":
                    model = CoordModel(text, order=max(1, int(arg)))
                elif cmd == "help":
                    print(HELP.format(length=length, temp=temp, order=model.order))
                else:
                    print("Unknown command. Try :help")
                    continue
                print(f"ok (len={length}, temp={temp}, order={model.order})")
            except ValueError:
                print("Bad value for that command.")
            continue
        prompt = line.replace("\\n", "\n")
        print("\n" + model.generate(prompt, length, temp))


if __name__ == "__main__":
    main()
