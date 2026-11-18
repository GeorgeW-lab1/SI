import numpy as np
import re
from collections import deque
KB_LIMIT = -1
class WordOptoelectronicLLM:
    def __init__(self, corpus, dt=0.001, seed=42):
        """
        Word-level Optoelectronic Reservoir & Stochastic Probe LLM.
        dt: Simulation timestep in seconds (1ms default)
        """
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        
        # Tokenize corpus into words and punctuation tokens
        self.tokens = self.tokenize(corpus)
        self.vocab = list(set(self.tokens))
        self.vocab_size = len(self.vocab)
        self.tok_to_id = {w: i for i, w in enumerate(self.vocab)}
        self.id_to_tok = {i: w for i, w in enumerate(self.vocab)}
        
        # Map each unique word to a distinct dual-ROI optical intensity pair (0V to 5V)
        self.roi_embeddings = self.rng.uniform(0.2, 4.8, size=(self.vocab_size, 2))
        
        # --- CIRCUIT PARAMETERS FROM SCHEMATIC ---
        self.vcc = 5.0
        
        # Driver B: Schmitt Trigger Hysteresis Thresholds
        self.schmitt_high = 3.0
        self.schmitt_low = 1.5
        self.driver_b_state = 0.0
        
        # RC Integrator Node (R = 1k Ohm, C = 4.7 uF -> tau = 4.7 ms)
        self.R = 1000.0
        self.C = 4.7e-6
        self.tau_rc = self.R * self.C  # 0.0047 s
        self.v_rc = 0.0
        
        # NPN Transistor Clamping Threshold
        self.v_th_npn = 0.7
        
        # Driver A: EMA Filter & 0.5V Comparators
        self.ema_alpha = 0.25
        self.ema_top = 0.0
        self.ema_bot = 0.0
        self.v_ref_comp = 0.5
        
        # Post-AND Delay Line (T_dly = 10 ms)
        self.t_dly_steps = max(1, int(0.010 / self.dt))
        self.delay_line = deque([0] * self.t_dly_steps, maxlen=self.t_dly_steps)
        
        # D Flip-Flop (DFF) & Clock State
        self.dff_q = 0
        self.prev_clk = 0
        
        # Secondary Probe Branch (T_probe = 2 * T_main)
        self.step_count = 0
        self.probe_accumulator = 0.0
        
        # Trainable Readout Matrix (maps multi-tap circuit state -> vocab logits)
        self.state_dim = 8
        self.W_out = np.zeros((self.state_dim, self.vocab_size))

    @staticmethod
    def tokenize(text):
        """Generates trigrams from the tokenized text."""
        tokens = text.split()
        # Create tuples of 3 consecutive tokens
        arr = []
        for i in range(len(tokens) - 2):
            arr.append(tokens[i] + " " + tokens[i+1] + " " + tokens[i+2])
        return arr

    def step_circuit(self, roi_top_v, roi_bot_v, ext_y=0, ext_n=0):
        """Advances the analog/digital schematic by one timestep (dt)."""
        self.step_count += 1
        
        # 1. DRIVER B (Schmitt Trigger on Top ROI)
        if roi_top_v >= self.schmitt_high:
            self.driver_b_state = self.vcc
        elif roi_top_v <= self.schmitt_low:
            self.driver_b_state = 0.0
            
        # 2. RC INTEGRATOR NODE (R=1k, C=4.7uF)
        dv_rc = ((self.driver_b_state - self.v_rc) / self.tau_rc) * self.dt
        self.v_rc += dv_rc
        
        # 3. NPN TRANSISTOR THRESHOLD (V_th = 0.7V soft-clamping)
        if self.v_rc > self.v_th_npn:
            bleed = (self.v_rc - self.v_th_npn) * 0.65
            self.v_rc -= bleed
            v_node_a = self.v_th_npn + (self.v_rc - self.v_th_npn) * 0.2
        else:
            v_node_a = self.v_rc

        # 4. DRIVER A (Top & Bottom EMA Filters + 0.5V Comparators)
        self.ema_top = self.ema_alpha * v_node_a + (1 - self.ema_alpha) * self.ema_top
        self.ema_bot = self.ema_alpha * roi_bot_v + (1 - self.ema_alpha) * self.ema_bot
        
        comp_top = 1 if self.ema_top > self.v_ref_comp else 0
        comp_bot = 1 if self.ema_bot > self.v_ref_comp else 0
        
        # 5. MAIN OPTOELECTRONIC AND GATE
        main_and = comp_top & comp_bot
        
        # 6. DELAY LINE (T_dly = 10ms)
        and_dly = self.delay_line[0]
        self.delay_line.append(main_and)
        
        # 7. OR LOGIC GATES & D FLIP-FLOP (DFF)
        or_1 = comp_bot | ext_y
        clk_lw = or_1 | ext_n
        
        if clk_lw == 1 and self.prev_clk == 0:
            self.dff_q = and_dly
        self.prev_clk = clk_lw
        
        # 8. INDEPENDENT SECONDARY PROBE BRANCH
        raw_top_logic = 1 if roi_top_v > 2.5 else 0
        sec_and = raw_top_logic & and_dly
        
        epsilon = self.rng.normal(0.0, 0.15)
        probe_prob = np.clip(sec_and * 0.8 + epsilon, 0.0, 1.0)
        
        self.probe_accumulator = 0.7 * self.probe_accumulator + 0.3 * probe_prob
        probe_out = self.probe_accumulator if (self.step_count % 2 == 0) else self.probe_accumulator * 0.5
            
        return np.array([
            self.driver_b_state / self.vcc,
            self.v_rc / self.v_th_npn,
            self.ema_top / self.vcc,
            self.ema_bot / self.vcc,
            float(main_and),
            float(and_dly),
            float(self.dff_q),
            probe_out
        ])

    def encode_sequence(self, token_list, ticks_per_word=20):
        """Feeds a sequence of word tokens through the optical ROIs."""
        states = []
        for word in token_list:
            idx = self.tok_to_id.get(word, 0) # Fallback to 0 if OOV
            v_top, v_bot = self.roi_embeddings[idx]
            for t in range(ticks_per_word):
                noise = self.rng.normal(0, 0.05, size=2)
                y_pulse = 1 if (t == ticks_per_word // 2) else 0
                s = self.step_circuit(v_top + noise[0], v_bot + noise[1], ext_y=y_pulse)
            states.append(s)
        return np.array(states)

    def train(self, ridge_alpha=1e-2):
        """Trains readout weights using word-level transition history."""
        seq_tokens = self.tokens
        X = self.encode_sequence(seq_tokens[:-1])
        Y = np.zeros((len(seq_tokens) - 1, self.vocab_size))
        
        for i, target_word in enumerate(seq_tokens[1:]):
            Y[i, self.tok_to_id[target_word]] = 1.0
            
        I = np.eye(self.state_dim)
        self.W_out = np.linalg.solve(X.T @ X + ridge_alpha * I, X.T @ Y)

    def generate(self, prompt_text, num_words=30, temperature=0.5):
        """Autoregressively generates subsequent words."""
        prompt_tokens = self.tokenize(prompt_text)
        states = self.encode_sequence(prompt_tokens)
        current_state = states[-1]
        
        generated_tokens = list(prompt_tokens)
        
        for _ in range(num_words):
            logits = current_state @ self.W_out
            hw_temp = max(0.1, temperature * (0.8 + 0.4 * current_state[7]))
            exp_logits = np.exp((logits - np.max(logits)) / hw_temp)
            probs = exp_logits / np.sum(exp_logits)
            
            next_id = self.rng.choice(self.vocab_size, p=probs)
            next_word = self.id_to_tok[next_id]
            generated_tokens.append(next_word)
            
            current_state = self.encode_sequence([next_word])[-1]
            
        # Format output neatly with spacing around punctuation
        output_str = ""
        for w in generated_tokens:
            if re.match(r'[.,;?!:]', w):
                output_str += w
            else:
                output_str += " " + w
        return output_str.strip()

# --- EXAMPLE EXECUTION ---
if __name__ == "__main__":
    with open("xaa", "r", encoding="utf-8") as file:
        training_corpus = file.read()[:KB_LIMIT]

    llm = WordOptoelectronicLLM(training_corpus)
    llm.train()
    while True:
        output = llm.generate(prompt_text=input("USER: "), num_words=250, temperature=0.4)
        print("Generated Word Trigram Output:\n", output)
