
import numpy as np, re
from collections import Counter, deque

KB_LIMIT, PRIMARY_FRAC = 99999, 0.75
LL_FLOOR = 1e-6

# ============================================================
# 1. OPTOELECTRONIC RESERVOIR
# ============================================================
class WordOptoelectronicLLM:
    def __init__(self, corpus, extra_corpora=None, dt=.001, seed=42):
        self.rng, self.dt = np.random.default_rng(seed), dt
        self.raw = [corpus[:KB_LIMIT]] + [(x[:KB_LIMIT]) for x in (extra_corpora or [])]
        self.primary = self.tokenize(self.raw[0])
        self.extras = [self.tokenize(x) for x in self.raw[1:]]
        self.extras = [x for x in self.extras if x]
        self.vocab = sorted(set(t for ds in [self.primary]+self.extras for t in ds))
        self.vocab_size = len(self.vocab)
        self.tok_to_id = {w:i for i,w in enumerate(self.vocab)}
        self.id_to_tok = dict(enumerate(self.vocab))
        self.roi = self.rng.uniform(.2,4.8,(self.vocab_size,2))
        self.vcc, self.high, self.low = 5.,3.,1.5
        self.tau, self.vth, self.alpha = .0047,.7,.25
        self.state_dim, self.split = 8,6
        self.W_main = np.zeros((6,self.vocab_size))
        self.W_aux = np.zeros((2,self.vocab_size))
        self.reset()

    @staticmethod
    def tokenize(t):
        w=t.split()
        return [" ".join(w[i:i+3]) for i in range(len(w)-2)]

    def reset(self):
        self.driver=self.vrc=self.top=self.bot=self.probe=0.
        self.delay=deque([0]*max(1,int(.010/self.dt)),maxlen=max(1,int(.010/self.dt)))
        self.dff=self.clock=self.steps=0

    def step(self,a,b,ext=0):
        self.steps+=1
        if a>=self.high:self.driver=5.
        elif a<=self.low:self.driver=0.
        self.vrc+=(self.driver-self.vrc)*self.dt/self.tau
        if self.vrc>self.vth:
            self.vrc-=.65*(self.vrc-self.vth)
            node=self.vth+(self.vrc-self.vth)*.2
        else:node=self.vrc
        self.top=.25*node+.75*self.top
        self.bot=.25*b+.75*self.bot
        ct,cb=int(self.top>.5),int(self.bot>.5)
        gate=ct&cb
        delayed=self.delay[0]; self.delay.append(gate)
        clk=cb|ext
        if clk and not self.clock:self.dff=delayed
        self.clock=clk
        raw_top_logic = int(a > 2.5)
        sec_and = raw_top_logic & int(delayed)
        p = np.clip(sec_and * 0.8 + self.rng.normal(0, 0.15), 0.0, 1.0)
        self.probe=.7*self.probe+.3*p
        return np.array([self.driver/5,self.vrc/.7,self.top/5,self.bot/5,
                         gate,delayed,self.dff,self.probe*(1 if self.steps%2==0 else .5)])

    def encode(self,tokens,ticks=20):
        out=[]
        for word in tokens:
            a,b=self.roi[self.tok_to_id.get(word,0)]
            for i in range(ticks):
                s=self.step(a+self.rng.normal(0,.05),b+self.rng.normal(0,.05),
                            int(i==ticks//2))
            out.append(s)
        return np.array(out).reshape(-1,8)

    def ridge(self,datasets,sl,alpha):
        d=sl.stop-sl.start; A=np.zeros((d,d)); B=np.zeros((d,self.vocab_size))
        for toks in datasets:
            if len(toks)<2:continue
            self.reset(); X=self.encode(toks[:-1])[:,sl]
            Y=np.zeros((len(toks)-1,self.vocab_size))
            for i,t in enumerate(toks[1:]):Y[i,self.tok_to_id[t]]=1
            A+=X.T@X; B+=X.T@Y
        self.reset()
        return np.linalg.solve(A+alpha*np.eye(d),B)

    def train(self,alpha=.01):
        self.W_main=self.ridge([self.primary],slice(0,6),alpha)
        self.W_aux=self.ridge(self.extras or [self.primary],slice(6,8),alpha)
        return self

    def probs(self,s,temp=.4):
        s=np.atleast_2d(s)
        z=s[:,:6]@self.W_main+s[:,6:]@self.W_aux
        z=(z-z.max(axis=1,keepdims=True))/np.maximum(.1,temp*(.8+.4*s[:,7,None]))
        e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

    def generate(self,prompt,num_words=250,temperature=.8):
        toks=self.tokenize(prompt)
        if not toks:return ""
        self.reset(); state=self.encode(toks)[-1]
        for _ in range(num_words):
            p=self.probs(state,temperature)[0]
            word=self.id_to_tok[self.rng.choice(self.vocab_size,p=p)]
            toks.append(word); state=self.encode([word])[-1]
        return " ".join(toks)

# ============================================================
# 2. H PROPOSITIONS: SYNTACTIC TRANSFORMATION
# ============================================================
def h1(t): return t.lower()
def h2(t): return re.sub(r"[^\w\s']"," ",t)
def h3(t):
    w=t.split(); return " ".join(x for i,x in enumerate(w) if not i or x!=w[i-1])
def h4(t):
    w=t.split(); o=[]; i=0
    while i<len(w):
        if i+3<len(w) and w[i:i+2]==w[i+2:i+4]:o+=w[i:i+2]; i+=4
        else:o.append(w[i]); i+=1
    return " ".join(o)
def h5(t):
    w=t.split(); c=Counter(w); k=[x for x in w if c[x]>1]
    return " ".join(k) if k else t
def h6(t):
    w=t.split(); c=Counter(w); n=max(1,int(len(c)*.02))
    k=[x for x in w if x not in {a for a,_ in c.most_common(n)}]
    return " ".join(k) if k else t
def h7(t):
    s=re.split(r"(?<=[.!?])\s+",t)
    return " ".join(dict.fromkeys(x.strip() for x in s if x.strip()))

H={"H1":h1,"H2":h2,"H3":h3,"H4":h4,"H5":h5,"H6":h6,"H7":h7}

def normal(t): return " ".join(t.split())

def simmer(fn,text,max_iter=100):
    """Iterate until syntactic stability or resolve a transformation cycle."""
    current=normal(text); seen={current:0}; history=[current]
    for i in range(max_iter):
        nxt=normal(fn(current))
        if nxt==current:return current,i+1,"SETTLED"
        if nxt in seen:
            cycle=history[seen[nxt]:]
            return min(cycle,key=lambda x:(len(x),x)),i+1,"CYCLE_RESOLVED"
        seen[nxt]=len(history); history.append(nxt); current=nxt
    return current,max_iter,"ITERATION_LIMIT"

def build_h():
    transforms=dict(H)
    for a,fa in H.items():
        for b,fb in H.items():
            if a!=b:transforms[f"{a}>{b}"]=lambda t,fa=fa,fb=fb:fb(fa(t))
    return transforms

def syntactic_analysis(text):
    results={}
    for name,fn in build_h().items():
        settled,it,status=simmer(fn,text)
        results[name]={"text":settled,"iterations":it,"status":status}
    return results

# ============================================================
# 3. NATURALNESS ASSESSMENT AGAINST ORIGINAL DATA
# ============================================================
class NaturalnessAssessor:
    weights=(.35,.25,.20,.20)
    def __init__(self,text):
        w=text[:KB_LIMIT].split(); cut=max(3,int(len(w)*.9))
        self.train=" ".join(w[:cut]); self.hold=WordOptoelectronicLLM.tokenize(" ".join(w[cut:]))
        all_t=WordOptoelectronicLLM.tokenize(" ".join(w))
        self.seen=set(zip(all_t[:-1],all_t[1:]))
        tr=WordOptoelectronicLLM.tokenize(self.train)
        self.start=tr[-1] if tr else None

    def assess(self,m,n=80,temp=.4,seed=1234):
        m.rng=np.random.default_rng(seed); m.reset()
        X=m.encode(self.hold[:-1]) if len(self.hold)>1 else np.empty((0,8))
        ll=[]
        for i,s in enumerate(X):
            p=m.probs(s,temp)[0]
            target=m.tok_to_id.get(self.hold[i+1])
            ll.append(np.log(max(p[target] if target is not None else 0,LL_FLOOR)))
        likelihood=np.clip(1+np.mean(ll)/np.log(1/LL_FLOOR),0,1) if ll else 0.
        m.rng=np.random.default_rng(seed+1); m.reset()
        prev=self.start if self.start in m.tok_to_id else m.vocab[0]
        state=m.encode([prev])[-1]; gen=[]; seen=coh=0
        for _ in range(n):
            p=m.probs(state,temp)[0]; nxt=m.id_to_tok[m.rng.choice(m.vocab_size,p=p)]
            seen+=(prev,nxt) in self.seen
            coh+=nxt.split()[:2]==prev.split()[1:]
            gen.append(nxt); prev=nxt; state=m.encode([nxt])[-1]
        m.reset(); div=len(set(gen))/max(1,n)
        seen/=max(1,n); coh/=max(1,n)
        a,b,c,d=self.weights
        return dict(likelihood=likelihood,seen=seen,coherence=coh,
                    diversity=div,composite=a*likelihood+b*seen+c*coh+d*div)

# ============================================================
# 4. CONTINUOUS SYNTACTIC SIMMERING AND LEARNING
# ============================================================
def continuous_train(original,extra_texts=None,max_rounds=100,
                     explore=.5,tau=.3,ema=.3,min_keep=.5,seed=0):
    assessor=NaturalnessAssessor(original); rng=np.random.default_rng(seed)
    primary=normal(assessor.train); extras=[normal(x) for x in (extra_texts or [])]
    names=list(H); reward={h:0. for h in names}; tries={h:0 for h in names}
    rejected=set(); history=[]
    def build(p,e):
        m=WordOptoelectronicLLM(p,e).train()
        return m,assessor.assess(m)
    best,score=build(primary,extras); base=max(1,len(primary.split()))
    for r in range(max_rounds):
        logits=np.array([reward[h]+explore/np.sqrt(1+tries[h]) for h in names])
        p=np.exp((logits-logits.max())/tau); p/=p.sum()
        h=names[rng.choice(len(names),p=p)]; tries[h]+=1
        np_,ne=simmer(H[h],primary)[0],[simmer(H[h],x)[0] for x in extras]
        if np_==primary and ne==extras:continue
        key=(h,hash(primary))
        if key in rejected:continue
        kept=len(np_.split())/base
        if kept<min_keep:
            reward[h]=(1-ema)*reward[h]-ema; rejected.add(key); continue
        model,new=build(np_,ne); delta=new["composite"]-score["composite"]
        accepted=delta>1e-4
        reward[h]=(1-ema)*reward[h]+ema*np.tanh(20*delta)
        history.append(dict(round=r+1,H=h,accepted=accepted,delta=delta,score=new))
        if accepted:primary,extras,best,score=np_,ne,model,new
        else:rejected.add(key)
        print(f"{r+1}: {h} | {delta:+.4f} | {'ACCEPT' if accepted else 'REJECT'}")
    return best,history,reward

# ============================================================
# 5. EXECUTION
# ============================================================
if __name__=="__main__":
    with open(input("Primary filename: "),"r",encoding="utf-8") as f:
        primary=f.read()[:19999]
    print("\n--- SYNTACTIC SIMMERING ---")
    for name,result in syntactic_analysis(primary).items():
        print(name,result["status"],result["iterations"])
    llm,history,weights=continuous_train(primary,[primary],max_rounds=4)
    while True:
        prompt=input("\nUSER: ")
        print("MODEL:",llm.generate(prompt,250,.8))
