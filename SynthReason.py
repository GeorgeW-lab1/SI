#!/usr/bin/env python3
# Copyright. George Frederick Wagenknecht - 2015/2016/2017
"""
SynthReason - trigram edition, open-ended (unlimited) DES, inference on load
============================================================================
trigrams.txt is the only thing that is learned. Everything else is COMPUTED
from it every time the trigrams are loaded (and after every new input):

 1. RELATIONS   Chains of trigrams are walked:  a  r1 r2 ..  b
                    cats live in houses  ->  (cats, "live in", houses)
                    children love summer ->  (children, "love", summer)
                a and b are entities, the words between them are the
                relation.  The LAST relation word is its DES ("head"):
                "in", "love", "because", "plants" ... any word can become
                a DES, so there is no fixed list - the set of DES is whatever
                the trigrams contain.
 2. INFERENCE   Relations with the same transitive DES are composed until
                nothing new appears (no hop limit):
                    cats live in houses + houses are in towns -> cats in towns
 3. ALIASES     A DES nobody has classified is matched to known ones by the
                entities it connects (distribution), so a new word such as
                "inside" can inherit "where" from "in".
                Named questions (who/what/where/when/why/how/describe/feel/
                count) are just classes of DES; cross-typing (why<->how,
                what<->describe ...) lets one DES answer another.
 4. FEEDBACK    Every inferred relation is turned back into trigrams
                (kept apart from learned ones so inference never feeds on
                itself), and topics are recomputed from learned + inferred.

Ask with a class ("where are cats?") or with ANY DES word ("love summer?",
"because soil?").  knowledgebase.txt is regenerated as a readable view:

    [@in,where|what:topic>word:topic>word:]    @ learned   ~ inferred

Usage
    python synthreason_open_des.py                 # menu
    python synthreason_open_des.py learn
    python synthreason_open_des.py teach [file]    # default test.txt
    python synthreason_open_des.py run
    python synthreason_open_des.py facts | des | topics | status

Files: trigrams.txt (learned, persisted), inferred_trigrams.txt and
knowledgebase.txt (regenerated each time).
"""

import argparse
import os
import random
import re
import sys
from collections import Counter, defaultdict

KNOWLEDGEBASE = "knowledgebase.txt"
TRIGRAM_FILE = "trigrams.txt"
INFERRED_FILE = "inferred_trigrams.txt"
LOADFILE_PATH = "test.txt"

BOS, EOS = "<s>", "</s>"
PAD = {BOS, EOS}
FUNC = "_"

STOP = set("""a an the is are was were be been being am of in on at to for by with
and or but nor not no it its this that these those i you he she we they me my
your his her our their as from so if then than do does did have has had will
would can could should may might must shall into over under about there here
what which who whom whose when where why how also very just""".split())
DETS = set("a an the its their his her our my your this that these those".split())

NUMWORDS = set("one two three four five six seven eight nine ten many few several "
               "all each every number count".split())
COPULA = {"is", "are", "was", "were", "am"}

# Seed knowledge that maps the NAMED questions onto DES words. Everything
# else (any other DES) is discovered from the trigrams.
CUES = {
    "why":   set("because since therefore hence thus".split()) | {"so"},
    "how":   set("by using through via".split()),
    "where": set("in at near inside under beside above behind between outside "
                 "around on".split()),
    "when":  set("when before after during while until".split()),
    "who":   set("he she they people man woman person everyone someone child "
                 "children friend mother father teacher".split()),
    "feel":  set("feel feels love loves hate hates happy sad angry afraid fear "
                 "joy like likes enjoy".split()),
}
QWORDS = {"who", "what", "where", "when", "why", "how", "describe", "feel", "count"}
QNOISE = QWORDS | {"many", "much", "number", "which", "tell", "me", "about"}

# a relation of class x can also answer these other classes
CROSS_TAG = {
    "why": ["how"], "how": ["why"], "what": ["describe"],
    "describe": ["what", "feel"], "who": ["what"], "where": ["what"],
    "when": ["what"], "feel": ["describe"], "count": ["describe"],
}

# DES heads whose relations compose:  a h b  +  b h c  =>  a h c
TRANSITIVE = (COPULA | CUES["where"] | {"before", "after"} |
              {"because", "since", "therefore", "hence", "thus"})

# words that glue relations together (never entities)
FUNCW = (STOP | CUES["why"] | CUES["how"] | CUES["where"] |
         {"before", "after", "during", "while", "until"})

MAXREL = 4                # longest relation walk (words between entities + 1)
MAX_PATHS = 5000          # safety bound per start word
MAX_DERIVED = 200000      # safety bound for the inference fixpoint
ALIAS_MIN = 0.25          # entity overlap needed to alias one DES to another

NOT_ADV = set("family only early daily likely lonely friendly belly holy italy "
              "apply reply supply ugly silly lovely lively".split())


class BackToMenu(Exception):
    pass


# ----------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------
def tokenize(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def sentences(text):
    text = re.sub(r"[,:<>\[\]]", "", text)
    out = []
    for s in re.split(r"[.!?]+", text):
        toks = tokenize(s)
        if toks:
            out.append(toks)
    return out


def read_text(path):
    with open(path, "r", errors="ignore") as f:
        return " ".join(f.read().replace("\r", "\n").split("\n"))


def padded_trigrams(toks):
    p = [BOS, BOS] + list(toks) + [EOS]
    return [tuple(p[i:i + 3]) for i in range(len(p) - 2)]


def is_entity(w):
    return w not in PAD and w not in FUNCW


# ----------------------------------------------------------------------------
# Relations (computed from trigrams)
# ----------------------------------------------------------------------------
class Rel:
    def __init__(self, a, phrase, b, n, kind):
        self.a, self.phrase, self.b = a, tuple(phrase), b
        self.n = n
        self.kind = kind                    # '@' from trigrams, '~' inferred
        self.direct = set()                 # classes it belongs to
        self.inferred = set()               # classes it can ALSO answer

    @property
    def head(self):
        return self.phrase[-1]

    @property
    def words(self):
        return [self.a] + list(self.phrase) + [self.b]

    def text(self):
        return " ".join(self.words)

    def key(self):
        return (self.a, self.phrase, self.b)


def extract(tri):
    """Walk trigram chains  a r1 r2 .. b  and return the relations found."""
    succ = defaultdict(Counter)
    first = defaultdict(list)
    for t, n in tri.items():
        succ[(t[0], t[1])][t[2]] += n
        if is_entity(t[0]):
            first[t[0]].append((t, n))

    cands = {}
    for a, lst in first.items():
        if a in NUMWORDS:                       # modifier, not a subject
            continue
        stack = [(list(t), n) for t, n in lst]
        budget = MAX_PATHS
        while stack and budget > 0:
            seq, cnt = stack.pop()
            budget -= 1
            tail = seq[1:]
            if any(w in PAD for w in tail):
                continue
            last = tail[-1]
            if is_entity(last):
                relw = tail[:-1]
                rel = tuple(w for w in relw if w not in DETS)
                if not rel or a == last:
                    continue
                if any(w not in FUNCW for w in relw[1:]):
                    continue                    # only the first word may be a verb
                key = (a, rel, last)
                cands[key] = max(cands.get(key, 0), cnt)
            elif len(tail) < MAXREL:
                for z, n in succ.get((seq[-2], seq[-1]), {}).items():
                    stack.append((seq + [z], min(cnt, n)))

    # "cats live in houses": drop the shadow (live, in, houses) of (cats, live in, houses)
    tails = {(rel[0], rel[1:], b) for (a, rel, b) in cands if len(rel) > 1}
    return [Rel(a, rel, b, n, "@") for (a, rel, b), n in cands.items()
            if (a, rel, b) not in tails]


def closure(base):
    """Compose transitive relations until nothing new appears."""
    known = {(r.a, r.head, r.b) for r in base}
    allr = list(base)
    derived = []
    changed = True
    while changed and len(derived) < MAX_DERIVED:
        changed = False
        by_a = defaultdict(list)
        for r in allr:
            by_a[r.a].append(r)
        for r in list(allr):
            if r.head not in TRANSITIVE:
                continue
            for s in by_a.get(r.b, []):
                if s.head != r.head or s.b == r.a:
                    continue
                key = (r.a, r.head, s.b)
                if key in known:
                    continue
                known.add(key)
                nr = Rel(r.a, (r.head,), s.b, min(r.n, s.n), "~")
                derived.append(nr)
                allr.append(nr)
                changed = True
    return derived


def classes_of(r):
    s = set()
    h = r.head
    if h in COPULA:
        s |= {"what", "describe"}
    for d, cues in CUES.items():
        if d != "who" and h in cues:
            s.add(d)
    if r.a in CUES["who"]:
        s.add("who")
    if r.b.endswith("ly") and len(r.b) > 4 and r.b not in NOT_ADV:
        s.add("how")
    if r.a in NUMWORDS or r.b in NUMWORDS or r.a.isdigit() or r.b.isdigit():
        s.add("count")
    return s


def annotate(rels):
    """Assign classes, then alias unclassified DES to classified ones by the
    entities they connect, then add cross-typed classes."""
    ents = defaultdict(set)
    hclass = defaultdict(set)
    for r in rels:
        r.direct = classes_of(r)
        ents[r.head] |= {r.a, r.b}
        hclass[r.head] |= r.direct
    alias = defaultdict(set)
    for h2, e2 in ents.items():
        if hclass[h2]:
            continue
        for h1, e1 in ents.items():
            if h1 == h2 or not hclass[h1]:
                continue
            inter = e1 & e2
            if inter and len(inter) / len(e1 | e2) >= ALIAS_MIN:
                alias[h2] |= hclass[h1]
    for r in rels:
        extra = set(alias.get(r.head, ()))
        for d in r.direct | extra:
            extra |= set(CROSS_TAG.get(d, ()))
        r.inferred = extra - r.direct


# ----------------------------------------------------------------------------
# Trigram / topic model
# ----------------------------------------------------------------------------
class Model:
    def __init__(self):
        self.tri = Counter()        # learned (persisted)
        self.itri = Counter()       # inferred (recomputed each rebuild)
        self.rels = []
        self.heads = set()
        self._wt = self._succ = self._freq = None

    # ---- persistence / loading
    def load(self, path=TRIGRAM_FILE):
        self.tri.clear()
        if os.path.isfile(path):
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    try:
                        trip, n = line.rstrip("\n").split("\t")
                        t = tuple(trip.split(" "))
                        if len(t) == 3:
                            self.tri[t] += int(n)
                    except ValueError:
                        continue
        self.rebuild()                       # <- inference happens here, on load

    def save_all(self):
        with open(TRIGRAM_FILE, "w") as f:
            for t, n in self.tri.items():
                f.write("%s\t%d\n" % (" ".join(t), n))
        with open(INFERRED_FILE, "w") as f:
            for t, n in self.itri.items():
                f.write("%s\t%d\n" % (" ".join(t), n))
        export_kb(self.rels)

    def add(self, toks):
        for t in padded_trigrams(toks):
            self.tri[t] += 1

    # ---- computation
    def rebuild(self):
        base = extract(self.tri)
        derived = closure(base)
        self.rels = base + derived
        annotate(self.rels)
        self.itri = Counter()
        for r in derived:                    # inferences become trigrams too
            for t in padded_trigrams(r.words):
                self.itri[t] += 1
        self.heads = {r.head for r in self.rels}
        self._wt = self._succ = self._freq = None

    def combined(self):
        c = Counter(self.tri)
        c.update(self.itri)
        return c

    # ---- topics (from learned + inferred trigrams)
    def _index(self):
        tri = self.combined()
        freq = Counter()
        for (_, w, _), n in tri.items():
            if w not in PAD:
                freq[w] += n
        self._freq = freq
        wt = defaultdict(Counter)
        succ = defaultdict(Counter)
        for t, n in tri.items():
            succ[(t[0], t[1])][t[2]] += n
            words = [w for w in t if w not in PAD and w not in STOP]
            if not words:
                continue
            top = max(words, key=lambda w: (freq[w], w))
            for w in words:
                wt[w][top] += n
        self._wt, self._succ = wt, succ

    def topic(self, word):
        if word in STOP:
            return FUNC
        if self._wt is None:
            self._index()
        c = self._wt.get(word)
        if not c:
            return None
        return max(c.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def topic_counts(self):
        if self._wt is None:
            self._index()
        c = Counter()
        for w in self._wt:
            c[self.topic(w)] += 1
        return c

    def generate(self, seed, maxlen=14):
        if self._succ is None:
            self._index()
        cands = [(t, n) for t, n in self.combined().items()
                 if seed in t and t[0] != EOS and t[1] != EOS]
        if not cands:
            return ""
        t = random.choices([c[0] for c in cands], [c[1] for c in cands])[0]
        out = [w for w in t if w not in PAD]
        state = (t[1], t[2])
        while len(out) < maxlen and state[1] != EOS:
            nxt = self._succ.get(state)
            if not nxt:
                break
            w = random.choices(list(nxt), list(nxt.values()))[0]
            if w == EOS:
                break
            out.append(w)
            state = (state[1], w)
        return " ".join(out)


MODEL = Model()


def export_kb(rels, kb=KNOWLEDGEBASE):
    tmp = kb + ".tmp"
    with open(tmp, "w") as f:
        for r in rels:
            head = ",".join([r.head] + sorted(r.direct))
            if r.inferred:
                head += "|" + ",".join(sorted(r.inferred))
            body = "".join("%s>%s:" % (MODEL.topic(w) or FUNC, w) for w in r.words)
            f.write("[%s%s:%s]" % (r.kind, head, body))
    os.replace(tmp, kb)


# ----------------------------------------------------------------------------
# Questions / answering (any DES)
# ----------------------------------------------------------------------------
def parse_question(text):
    """-> (is_question, des or None, content tokens).
    des is a named class (who/where/...) or ANY known DES word."""
    toks = tokenize(text)
    des, used = None, None
    for i, w in enumerate(toks):
        if w in QWORDS:
            des, used = w, i
            if w == "how" and i + 1 < len(toks) and toks[i + 1] in ("many", "much"):
                des = "count"
            break
    if des is None:
        heads = [i for i, w in enumerate(toks) if w in MODEL.heads]
        if heads:
            pick = [i for i in heads if toks[i] not in STOP] or heads
            used = pick[0]
            des = toks[used]
    is_q = "?" in text or (bool(toks) and toks[0] in QWORDS)
    content = [w for i, w in enumerate(toks)
               if w not in QNOISE and not (des not in QWORDS and i == used)]
    return is_q, des, content


def answer(text):
    """Best relation for the question -> (text, route note) or (None, None)."""
    _, des, content = parse_question(text)
    topics = Counter(t for t in (MODEL.topic(w) for w in content) if t and t != FUNC)
    words = set(w for w in content if w not in STOP)
    if not topics and not words:
        return None, None
    best, bestkey = None, None
    for r in MODEL.rels:
        rw = set(r.words)
        rt = {MODEL.topic(w) for w in r.words} - {None, FUNC}
        overlap = sum(topics[t] for t in rt if t in topics) + 2 * len(words & rw)
        if overlap == 0:
            continue
        if des is None:
            rank = 2
        elif des == r.head or des in r.direct:
            rank = 3 if r.kind == "@" else 2
        elif des in r.inferred:
            rank = 1
        else:
            continue
        key = (rank, overlap, r.n, -len(r.words))
        if bestkey is None or key > bestkey:
            best, bestkey = r, key
    if best:
        if des is None or des == best.head or des in best.direct:
            note = ("direct via '%s'" % best.head if best.kind == "@"
                    else "chained inference via '%s'" % best.head)
        else:
            note = "inferred: '%s' answered from a '%s' relation" % (des, best.head)
        return best.text(), note
    if topics:
        text = MODEL.generate(topics.most_common(1)[0][0])
        if text:
            return text, "trigram walk"
    return None, None


# ----------------------------------------------------------------------------
# Learning
# ----------------------------------------------------------------------------
def learn_sentences(sents, verbose=True):
    before = {(r.kind, r.key()) for r in MODEL.rels}
    tri0, inf0 = len(MODEL.tri), len(MODEL.itri)
    for s in sents:
        MODEL.add(s)
    MODEL.rebuild()                      # trigrams -> relations -> inference
    MODEL.save_all()
    new = [r for r in MODEL.rels if (r.kind, r.key()) not in before]
    if verbose:
        print("  trigrams %d -> %d, inferred trigrams %d -> %d"
              % (tri0, len(MODEL.tri), inf0, len(MODEL.itri)))
        for r in [x for x in new if x.kind == "@"][:10]:
            print("  relation [%s] %s" % (r.head, r.text()))
        der = [x for x in new if x.kind == "~"]
        for r in der[:10]:
            print("  inferred [%s] %s" % (r.head, r.text()))
        if len(der) > 10:
            print("  ... and %d more inferred" % (len(der) - 10))
    return new


def teach(path=None):
    path = path or LOADFILE_PATH
    if not os.path.isfile(path):
        print("Cannot find '%s'. Put some plain text in it and try again." % path)
        return
    sents = sentences(read_text(path))
    if not sents:
        print("Nothing to learn in '%s'." % path)
        return
    new = learn_sentences(sents, verbose=False)
    nl = sum(r.kind == "@" for r in MODEL.rels)
    nd = sum(r.kind == "~" for r in MODEL.rels)
    print("Learned %d sentence(s). Trigrams: %d learned + %d inferred."
          % (len(sents), len(MODEL.tri), len(MODEL.itri)))
    print("Relations: %d from trigrams, %d inferred, %d distinct DES (%d new this time)."
          % (nl, nd, len(MODEL.heads), len(new)))


# ----------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------
def status():
    nl = sum(r.kind == "@" for r in MODEL.rels)
    nd = sum(r.kind == "~" for r in MODEL.rels)
    print("\nTrigrams: %d learned, %d inferred   topics: %d"
          % (len(MODEL.tri), len(MODEL.itri), len(MODEL.topic_counts())))
    print("Relations: %d from trigrams, %d inferred   distinct DES: %d"
          % (nl, nd, len(MODEL.heads)))
    for f in (LOADFILE_PATH, TRIGRAM_FILE, INFERRED_FILE, KNOWLEDGEBASE):
        print("  %-22s %s" % (f, "%d bytes" % os.path.getsize(f)
                              if os.path.isfile(f) else "(missing)"))
    print()


def show_facts(n=40):
    if not MODEL.rels:
        print("No relations yet - use Teach first.")
        return
    for r in MODEL.rels[-n:]:
        extra = ",".join(sorted(r.direct))
        alt = ",".join(sorted(r.inferred))
        print("  %s [%s%s%s] %s" % (r.kind, r.head, (" =" + extra) if extra else "",
                                    (" | " + alt) if alt else "", r.text()))


def show_des():
    if not MODEL.rels:
        print("No DES yet - use Teach first.")
        return
    by = defaultdict(list)
    for r in MODEL.rels:
        by[r.head].append(r)
    print("\nDES found in the trigrams (%d):" % len(by))
    for h, rs in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cls = set().union(*(r.direct for r in rs))
        alt = set().union(*(r.inferred for r in rs))
        print("  %-12s %3d  class: %-16s also answers: %s"
              % (h, len(rs), ",".join(sorted(cls)) or "-", ",".join(sorted(alt)) or "-"))
    print()


def show_topics(n=20):
    c = MODEL.topic_counts()
    if not c:
        print("No topics yet - use Teach first.")
        return
    print("\nTop topics (topic: number of words under it)")
    for t, k in c.most_common(n):
        if t == FUNC:
            continue
        members = [w for w in MODEL._wt if MODEL.topic(w) == t][:8]
        print("  %-14s %3d   %s" % (t, k, ", ".join(sorted(members))))
    print()


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def learn_mode():
    print("SynthReason (open DES) - type 'menu' to go back")
    print("State something to teach it. Ask with who/what/where/when/why/how/describe/")
    print("feel/count - or with ANY DES word, e.g. 'love summer?'.")
    print("Commands: load file, show facts, show des, show topics, show answerstring")
    try:
        while True:
            line = input("USER: ").strip()
            low = line.lower()
            if low in ("quit", "exit", "menu"):
                raise BackToMenu()
            if not line:
                continue
            if low == "load file":
                teach(LOADFILE_PATH)
            elif low == "show facts":
                show_facts()
            elif low == "show des":
                show_des()
            elif low == "show topics":
                show_topics()
            elif low == "show answerstring":
                print(open(KNOWLEDGEBASE, errors="ignore").read()
                      if os.path.isfile(KNOWLEDGEBASE) else "")
            elif low.startswith("topic "):
                w = low.split(None, 1)[1]
                print("topic of '%s': %s" % (w, MODEL.topic(w)))
            else:
                is_q, _, _ = parse_question(line)
                if is_q:
                    text, note = answer(line)
                    print("[%s]  (%s)" % (text, note) if text else "[no answer yet]")
                else:
                    sents = sentences(line)
                    if sents:
                        learn_sentences(sents)
    except (BackToMenu, EOFError, KeyboardInterrupt):
        print()


def des_options(key):
    """Every DES (class or open-ended head) that touches the key word."""
    topic = MODEL.topic(key)
    out = []
    for r in MODEL.rels:
        rt = {MODEL.topic(w) for w in r.words}
        if key in r.words or (topic and topic in rt):
            for d in [r.head] + sorted(r.direct) + sorted(r.inferred):
                if d not in out:
                    out.append(d)
    return out


def run_mode():
    """0.9N: walks from a context word, asking it every DES it has - named
    classes and open-ended ones - so one DES's answer seeds the next."""
    print("SynthReason 0.9N - open DES walk")
    try:
        if not MODEL.rels:
            print("Nothing learned yet - use 'Teach' first.")
            return
        while True:
            key = input("\nContext (blank or 'menu' = back):").strip().lower()
            if key in ("", "menu", "quit", "exit"):
                return
            spoken, tried = 0, set()
            for _ in range(30):
                if spoken >= 7:
                    break
                opts = [d for d in des_options(key) if (key, d) not in tried]
                if not opts:
                    break
                des = random.choice(opts)
                tried.add((key, des))
                q = ("how many %s?" % key) if des == "count" else "%s %s?" % (des, key)
                text, note = answer(q)
                if not text:
                    continue
                spoken += 1
                print("%-10s %s  (%s)" % (des + ":", text, note))
                nxt = [w for w in text.split() if w not in STOP and w != key]
                if nxt:
                    key = random.choice(nxt)
            if spoken == 0:
                print("(nothing learned about '%s')" % key)
    except (EOFError, KeyboardInterrupt):
        print()


def menu():
    while True:
        print("""
SynthReason - open DES, inference on load
  1  Learn / chat
  2  Teach from a text file
  3  Run SynthReason 0.9N
  4  Show relations
  5  Show DES
  6  Show topics
  7  Status
  0  Quit""")
        try:
            c = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if c == "1":
            learn_mode()
        elif c == "2":
            try:
                p = input("Text file [%s]:" % LOADFILE_PATH).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            teach(p or None)
        elif c == "3":
            run_mode()
        elif c == "4":
            show_facts()
        elif c == "5":
            show_des()
        elif c == "6":
            show_topics()
        elif c == "7":
            status()
        elif c in ("0", "q", "quit", "exit"):
            return


def main(argv=None):
    ap = argparse.ArgumentParser(description="SynthReason open-DES trigram inference")
    ap.add_argument("command", nargs="?",
                    choices=["learn", "teach", "run", "facts", "des", "topics", "status"])
    ap.add_argument("file", nargs="?", help="text file (teach)")
    args = ap.parse_args(argv)

    MODEL.load()                         # loads trigrams AND infers from them
    nd = sum(r.kind == "~" for r in MODEL.rels)
    if MODEL.tri:
        print("Loaded %d trigrams; inferred %d relation(s) across %d DES."
              % (len(MODEL.tri), nd, len(MODEL.heads)))

    if args.command == "learn":
        learn_mode()
    elif args.command == "teach":
        teach(args.file)
    elif args.command == "run":
        run_mode()
    elif args.command == "facts":
        show_facts()
    elif args.command == "des":
        show_des()
    elif args.command == "topics":
        show_topics()
    elif args.command == "status":
        status()
    else:
        menu()
    return 0


if __name__ == "__main__":
    sys.exit(main())
