#!/usr/bin/env python3
""" AI: standard-library corpus solver, answer pipeline and feedback RL.

Run: python _ai.py
This is a retrieval/rule-based assistant, not a general-purpose LLM.
"""
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict

SETTINGS = dict(space_bits=14, math_level1_scale=1.1182,
                math_level2_weight=0.5806, probability_smoothing=0.054,
                probability_temperature=0.5196, probability_spatial=0.2461,
                rl_rate=0.08, rl_temperature=0.8, rl_exploration=0.08,
                rl_episodes=40, rl_length=24)

STOP = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'being', 'but', 'by',
    'did', 'do', 'does', 'doing', 'done', 'for', 'from', 'had', 'has', 'have',
    'having', 'he', 'her', 'hers', 'him', 'his', 'how', 'i', 'if', 'in', 'into',
    'is', 'it', 'its', 'me', 'my', 'of', 'on', 'or', 'our', 'ours', 'she', 'so',
    'than', 'that', 'the', 'their', 'theirs', 'them', 'then', 'there', 'these',
    'they', 'this', 'those', 'to', 'too', 'up', 'very', 'was', 'we', 'were',
    'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'you',
    'your', 'yours',
}


def tokens(text):
    return re.findall(r"[^\W_]+(?:['\u2019][^\W_]+)*|[^\w\s]", text.lower(), re.UNICODE)


def words(text):
    return [t for t in tokens(text) if any(c.isalnum() for c in t)]


def untokenize(ts):
    s = ' '.join(ts)
    return re.sub(r'\s+([.,!?;:])', r'\1', s)


def hashed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big')


def probabilities(logits):
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    total = sum(weights)
    return [w / total for w in weights]


class AI:
    def __init__(self):
        self.rng = random.Random()
        self.text = ''
        self.dataset = []
        self.sentences = []
        self.index = defaultdict(set)
        self.counts = []
        self.unigrams = Counter()
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)
        self.preferences = defaultdict(dict)
        self.baseline = 0.0
        self.episodes = 0
        self.prompt = ''
        self.translations = {}
        self.default_chain = ['UNDERSTAND', 'RETRIEVE', 'ANSWER', 'GENERATE']
        self.chain_values = {}
        self.evidence_values = {}
        self.last_feedback = None
        self.last_trace = []
        self.dataset_state = 0.0

    def position(self, token, i):
        h = hashed(token)
        limit = 1 << SETTINGS['space_bits']
        return (h % limit, ((h >> 8) + i) % limit, ((h >> 16) ^ i) % limit)

    def arithmetic(self, token, i, state):
        x, y, z = self.position(token, i)
        limit = (1 << SETTINGS['space_bits']) - 1
        affinity = 1 - (x + y + z) / (3 * limit)
        local = math.tanh(hashed(token) / 4294967295 * SETTINGS['math_level1_scale']
                          + i / (i + 1000) + affinity * .5)
        return math.tanh((local + state) / 2)

    def fit(self, text):
        ts = tokens(text)
        if len(ts) < 6:
            raise ValueError('Provide at least six tokens of source text.')
        self.text, self.dataset = text, ts
        self.unigrams = Counter(ts)
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)
        self.preferences = defaultdict(dict)
        self.chain_values, self.evidence_values = {}, {}
        self.last_feedback = None
        self.last_trace = []
        self.baseline, self.episodes = 0.0, 0
        for i in range(len(ts) - 1):
            self.bigrams[ts[i]][ts[i + 1]] += 1
        for i in range(len(ts) - 3):
            self.trigrams[tuple(ts[i:i + 3])][ts[i + 3]] += 1
        self.sentences = []
        for s in re.split(r'(?<=[.!?])\s+|\n+', text):
            s = s.strip()
            if not words(s):
                continue
            if len(words(s)) > 150:
                w = s.split()
                self.sentences.extend(' '.join(w[i:i + 100]) for i in range(0, len(w), 100))
            else:
                self.sentences.append(s)
        self.index = defaultdict(set)
        self.counts = []
        for i, s in enumerate(self.sentences):
            c = Counter(words(s))
            self.counts.append(c)
            for w in c:
                self.index[w].add(i)
        state = 0.0
        for i, t in enumerate(ts):
            state = self.arithmetic(t, i, state)
        self.dataset_state = state

    def require_data(self):
        if not self.dataset:
            raise ValueError('Load source text with DATA first.')

    def retrieve(self, query, limit=5):
        terms = set(words(query)) - STOP
        if not terms:
            return []
        scores = defaultdict(float)
        n = len(self.sentences)
        for w in terms:
            ids = self.index.get(w, ())
            idf = math.log(1 + n / (1 + len(ids)))
            for i in ids:
                c = self.counts[i]
                scores[i] += idf * c[w] / (1 + 0.02 * sum(c.values()))
        for i in scores:
            scores[i] *= math.exp(.2 * max(-3, min(3, self.evidence_values.get(self.sentences[i], 0))))
        return sorted(scores, key=scores.get, reverse=True)[:limit]

    def candidates(self, context):
        return (self.trigrams.get(tuple(context[-3:])) or
                self.bigrams.get(context[-1] if context else '') or self.unigrams)

    def sample(self, context, history, bias, numeric, training):
        counts = self.candidates(context)
        items = counts.most_common(256)
        actions = [t for t, _ in items]
        key = tuple(context[-3:])
        pref = self.preferences[key]
        temp = SETTINGS['rl_temperature'] if training else SETTINGS['probability_temperature']
        logits = []
        for t, c in items:
            spatial = (hashed(t) / 4294967295 - .5) * numeric
            score = (math.log(c + SETTINGS['probability_smoothing']) + pref.get(t, 0)
                     + .5 * math.log1p(bias[t]) - .3 * history[-12:].count(t)
                     + SETTINGS['probability_spatial'] * spatial)
            logits.append(score / temp)
        soft = probabilities(logits)
        eps = SETTINGS['rl_exploration'] if training else 0.0
        mix = [(1 - eps) * p + eps / len(actions) for p in soft]
        j = self.rng.choices(range(len(actions)), weights=mix, k=1)[0]
        factor = (1 - eps) * soft[j] / mix[j] / temp
        return actions[j], (key, actions, soft, j, factor)

    def update_episode(self, trace):
        if not trace:
            return
        total = 0.0
        for record, reward in trace:
            key, actions, soft, j, factor = record
            advantage = reward - self.baseline
            prefs = self.preferences[key]
            for k, t in enumerate(actions):
                value = prefs.get(t, 0) + SETTINGS['rl_rate'] * advantage * factor * ((k == j) - soft[k])
                if abs(value) > 1e-12:
                    prefs[t] = max(-10, min(10, value))
            total += reward
        mean = total / len(trace)
        self.baseline = .9 * self.baseline + .1 * mean
        self.episodes += 1

    def generate(self, prompt, length=80, training=False, learn=True, seed=None):
        self.require_data()
        if not 1 <= length <= 2000:
            raise ValueError('Length must be between 1 and 2000.')
        context = list(seed or tokens(prompt)[-3:] or self.dataset[:3])
        bias = Counter(words(prompt))
        out, trace = [], []
        state = self.dataset_state
        for i in range(length):
            state = self.arithmetic(context[-1], i, state)
            t, record = self.sample(context, out, bias, state, training)
            reward = (.8 if t in self.trigrams.get(tuple(context[-3:]), {}) else 0)
            reward += .2 * min(1, bias[t]) - .25 * out[-8:].count(t)
            trace.append((record, reward))
            out.append(t)
            context = (context + [t])[-3:]
        if learn:
            self.update_episode(trace)
        return untokenize(out)

    def train(self, episodes=40, length=24):
        self.require_data()
        if not 1 <= episodes <= 10000 or not 1 <= length <= 2000:
            raise ValueError('Episodes: 1..10000; length: 1..2000.')
        for _ in range(episodes):
            i = self.rng.randrange(max(1, len(self.dataset) - 2))
            self.generate('', length, True, True, self.dataset[i:i + 3])
        return f'Trained {episodes} additional episodes; baseline reward {self.baseline:.3f}.'

    def save(self, path):
        self.require_data()
        data = dict(version='4.0', text=self.text, settings=SETTINGS, prompt=self.prompt,
                    translations=self.translations, chain_values=self.chain_values,
                    evidence_values=self.evidence_values, default_chain=self.default_chain,
                    baseline=self.baseline, episodes=self.episodes,
                    policy=[dict(context=list(k), values=v) for k, v in self.preferences.items()])
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        if data.get('version') != '4.0':
            raise ValueError('Use a version 4.0 save, or reload the original text with DATA.')
        self.fit(data['text'])
        self.prompt = data.get('prompt', '')
        self.translations = data.get('translations', {})
        self.chain_values = data.get('chain_values', {})
        self.evidence_values = data.get('evidence_values', {})
        self.default_chain = data.get('default_chain', self.default_chain)
        self.baseline = data.get('baseline', 0.0)
        self.episodes = data.get('episodes', 0)
        for row in data.get('policy', []):
            self.preferences[tuple(row['context'])] = row['values']


HELP = '''
SI 4.0
DATA filename.txt        Load plain text and run 40 RL episodes
LOAD model.json          Restore a version 4.0 save
SAVE model.json          Save source, policy, translations and feedback
PROMPT text              Set free-generation prompt
GENERATE 80              Sample n-gram text (not a factual answer)
TRAIN 40 24              Additional RL episodes and tokens per episode
STATUS / SETTINGS / HELP / EXIT

Paths may contain spaces or Windows backslashes; outer quotes are optional.
Ordinary input without a command is treated as SOLVE.
'''


def filename(s):
    return s.strip().strip('"').strip("'")


def interactive():
    ai = AI()
    print(HELP)
    print('Limitations: retrieval and rules, not general language understanding.')
    try:
        initial = input('Dataset filename (Enter to skip): ').strip()
        if initial:
            with open(filename(initial), encoding='utf-8', errors='replace') as f:
                ai.fit(f.read())
            print(ai.train())
    except (OSError, ValueError) as e:
        print('[ERROR]', e)
    except (EOFError, KeyboardInterrupt):
        return
    while True:
        try:
            line = input('\nSI> ').strip()
            if not line:
                continue
            first, _, arg = line.partition(' ')
            command = first.upper()
            arg = arg.strip()
            if command in ('EXIT', 'QUIT', '/QUIT'):
                break
            elif command == 'HELP':
                print(HELP)
            elif command == 'DATA':
                with open(filename(arg), encoding='utf-8', errors='replace') as f:
                    ai.fit(f.read())
                print(ai.train())
            elif command == 'SAVE':
                ai.save(filename(arg))
                print('Saved.')
            elif command == 'LOAD':
                ai.load(filename(arg))
                print('Restored.')
            elif command == 'PROMPT':
                ai.prompt = arg
                print('Prompt set.')
            elif command == 'GENERATE':
                print(ai.generate(ai.prompt, int(arg or 80)))
            elif command == 'TRAIN':
                values = arg.split()
                print(ai.train(int(values[0]) if values else 40, int(values[1]) if len(values) > 1 else 24))
            elif command == 'STATUS':
                print(json.dumps(dict(tokens=len(ai.dataset), vocabulary=len(ai.unigrams),
                      sentences=len(ai.sentences), contexts=len(ai.trigrams), episodes=ai.episodes,
                      baseline=ai.baseline, default_chain=ai.default_chain,
                      rated_chains=ai.chain_values), indent=2))
            elif command == 'SETTINGS':
                print(json.dumps(SETTINGS, indent=2))
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye.')
            break
        except (OSError, ValueError, KeyError, TypeError) as e:
            print('[ERROR]', e)


if __name__ == '__main__':
    interactive()
