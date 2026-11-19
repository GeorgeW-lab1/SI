from collections import deque
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class V(np.ndarray):
    """Lightweight ndarray subclass for V(...)."""
    def __new__(cls, input_array):
        return np.asarray(input_array, dtype=float).view(cls)


class ReservoirCell:
    def __init__(self, high=3.5, low=1.5, dt=0.1, tau=1.0, vth=0.7, delay_len=4, seed=42):
        self.high = high
        self.low = low
        self.dt = dt
        self.tau = tau
        self.vth = vth
        self.rng = np.random.default_rng(seed)
        self.delay_len = delay_len
        self.reset()

    def reset(self):
        self.steps = 0
        self.driver = 0.0
        self.vrc = 0.0
        self.top = 0.0
        self.bot = 0.0
        self.delay = deque([0] * self.delay_len, maxlen=self.delay_len)
        self.clock = 0
        self.dff = 0
        self.probe = 0.0

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
        return V(sorted(np.log(np.exp(state))))


class SyntacticGeometryLayer(nn.Module):
    """
    Applies a syntactic geometry layer over an n-gram window (batch, seq_len, n, dim):
      1. Encodes syntactic role/offset phase across window positions (relative to head w_t).
      2. Computes geometric transition features (stepwise deltas, head-relative deltas,
         and cosine similarities) capturing syntactic relations in embedding space.
      3. Replicates/expands geometric structure across k syntactic channels to control
         the current layer's representation via gated modulation.
    """
    def __init__(self, n: int, hidden_dim: int, num_geom_replicas: int = 4):
        super().__init__()
        self.n = n
        self.hidden_dim = hidden_dim
        self.num_geom_replicas = num_geom_replicas

        # Syntactic offset embeddings for positions 0..n-1 (index n-1 is current head word t)
        self.syntax_pos_emb = nn.Parameter(torch.randn(1, 1, n, hidden_dim) * 0.02)

        # Geometric feature dim per token in window:
        #   base (hidden_dim) + stepwise_delta (hidden_dim) + head_delta (hidden_dim) + cos_to_head (1)
        geom_in_dim = hidden_dim * 3 + 1

        # Replicate geometric manifold across K syntactic channels
        self.replicate_proj = nn.Linear(n * geom_in_dim, hidden_dim * num_geom_replicas)
        self.norm = nn.LayerNorm(hidden_dim)

        # Syntactic controller: produces a gate & shift to steer the current layer
        self.syntax_controller = nn.Sequential(
            nn.Linear(hidden_dim * (num_geom_replicas + 1), hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  # [gate_logits, delta_shift]
        )

    def _apply_rotary_syntax_phase(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies deterministic syntactic phase rotation along token offsets relative to head (n-1).
        x: (batch, seq_len, n, dim)
        """
        _, _, n, dim = x.shape
        half_dim = dim // 2
        if half_dim == 0:
            return x

        # Relative syntactic distance from head word: -(n-1), ..., -1, 0
        offsets = torch.arange(n, device=x.device, dtype=x.dtype) - float(n - 1)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=x.device, dtype=x.dtype) / max(half_dim, 1)
        )
        angles = offsets[:, None] * freqs[None, :]  # (n, half_dim)
        cos_a = torch.cos(angles)[None, None, :, :]
        sin_a = torch.sin(angles)[None, None, :, :]

        x_even = x[..., : 2 * half_dim : 2]
        x_odd = x[..., 1 : 2 * half_dim : 2]
        rot_even = x_even * cos_a - x_odd * sin_a
        rot_odd = x_even * sin_a + x_odd * cos_a

        rotated = x.clone()
        rotated[..., : 2 * half_dim : 2] = rot_even
        rotated[..., 1 : 2 * half_dim : 2] = rot_odd
        return rotated

    def forward(self, win_emb: torch.Tensor, current_layer_repr: torch.Tensor):
        """
        win_emb: (batch, seq_len, n, hidden_dim)
        current_layer_repr: (batch, seq_len, hidden_dim)
        Returns:
          controlled_repr: (batch, seq_len, hidden_dim)
          geom_summary: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, n, dim = win_emb.shape

        # 1. Inject syntactic position & rotary phase geometry
        geom_tokens = self._apply_rotary_syntax_phase(win_emb + self.syntax_pos_emb)

        # Head word (current token t) is at index n - 1
        head_tok = geom_tokens[:, :, -1:, :].expand(-1, -1, n, -1)

        # Stepwise syntactic transition: v_k - v_{k-1} (with zero delta at k=0)
        prev_tokens = torch.cat(
            [geom_tokens[:, :, :1, :], geom_tokens[:, :, :-1, :]], dim=2
        )
        step_delta = geom_tokens - prev_tokens

        # Geometric offset to syntactic head: v_k - v_head
        head_delta = geom_tokens - head_tok

        # Cosine similarity with head word as scalar geometric alignment
        cos_sim = F.cosine_similarity(geom_tokens, head_tok, dim=-1).unsqueeze(-1)

        geom_features = torch.cat([geom_tokens, step_delta, head_delta, cos_sim], dim=-1)
        flat_geom = geom_features.reshape(batch_size, seq_len, -1)

        # 2. Replicate across K syntactic geometric channels
        replicated = self.replicate_proj(flat_geom)  # (batch, seq_len, K * dim)
        geom_summary = self.norm(
            replicated.view(batch_size, seq_len, self.num_geom_replicas, dim).mean(dim=2)
        )

        # 3. Syntactically control the current layer
        ctrl_in = torch.cat([replicated, current_layer_repr], dim=-1)
        gate_logits, shift = self.syntax_controller(ctrl_in).chunk(2, dim=-1)
        syntax_gate = torch.sigmoid(gate_logits)

        controlled_repr = self.norm(
            current_layer_repr * (0.5 + syntax_gate) + shift * syntax_gate
        )
        return controlled_repr, geom_summary


class NGramReservoirLLM(nn.Module):
    """
    Word n-gram reservoir language model with a SyntacticGeometryLayer that replicates
    geometric syntactic relations to control the current layer before driving the reservoir.
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        n: int = 3,
        num_cells: int = 32,
        hidden_dim: int = 128,
        num_geom_replicas: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        if n < 1:
            raise ValueError("n must be >= 1")
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.n = n
        self.num_cells = num_cells
        self.hidden_dim = hidden_dim
        self.cells = [ReservoirCell(seed=seed + i) for i in range(num_cells)]

        # Embedding for each of the n words in the n-gram: (t-n+1, ..., t)
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.ngram_proj = nn.Sequential(
            nn.Linear(hidden_dim * n, hidden_dim),
            nn.GELU(),
        )

        # Syntactic geometry layer controlling the current layer
        self.syntax_geom = SyntacticGeometryLayer(
            n=n,
            hidden_dim=hidden_dim,
            num_geom_replicas=num_geom_replicas,
        )

        # Syntactically controlled n-gram context -> controls (a, b, ext) per cell
        self.to_controls = nn.Linear(hidden_dim, num_cells * 3)

        # Reservoir state + controlled n-gram + syntactic geometry summary -> next-word logits
        self.readout = nn.Sequential(
            nn.Linear(num_cells * 8 + hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def build_ngram_windows(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)
        returns windows: (batch, seq_len, n) where index -1 is current word t
        and index 0 is t - n + 1.
        """
        batch_size, seq_len = input_ids.shape
        if self.n > 1:
            pad = torch.full(
                (batch_size, self.n - 1),
                self.pad_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            padded = torch.cat([pad, input_ids], dim=1)
        else:
            padded = input_ids

        slices = [padded[:, k : k + seq_len] for k in range(self.n)]
        return torch.stack(slices, dim=-1)

    def encode_ngrams(self, input_ids: torch.Tensor):
        """
        returns:
          controlled_repr: (batch, seq_len, hidden_dim)
          geom_summary: (batch, seq_len, hidden_dim)
        """
        windows = self.build_ngram_windows(input_ids)  # (batch, seq_len, n)
        win_emb = self.token_emb(windows)  # (batch, seq_len, n, hidden_dim)
        flat_ngram = win_emb.flatten(start_dim=2)  # (batch, seq_len, n * hidden_dim)
        base_repr = self.ngram_proj(flat_ngram)  # (batch, seq_len, hidden_dim)

        controlled_repr, geom_summary = self.syntax_geom(win_emb, base_repr)
        return controlled_repr, geom_summary

    def rollout_reservoir(self, ngram_repr: torch.Tensor) -> torch.Tensor:
        """
        ngram_repr: (batch, seq_len, hidden_dim)
        returns reservoir_feats: (batch, seq_len, num_cells * 8)
        """
        batch_size, seq_len, _ = ngram_repr.shape
        with torch.no_grad():
            raw_ctrl = self.to_controls(ngram_repr).view(
                batch_size, seq_len, self.num_cells, 3
            )
            a_ctrl = torch.sigmoid(raw_ctrl[..., 0]) * 5.0
            b_ctrl = torch.sigmoid(raw_ctrl[..., 1]) * 5.0
            ext_ctrl = (raw_ctrl[..., 2] > 0.0).to(torch.int64)

            out = np.zeros((batch_size, seq_len, self.num_cells, 8), dtype=np.float32)

            for b_idx in range(batch_size):
                for cell in self.cells:
                    cell.reset()
                for t in range(seq_len):
                    for c_idx, cell in enumerate(self.cells):
                        st = cell.step(
                            float(a_ctrl[b_idx, t, c_idx].item()),
                            float(b_ctrl[b_idx, t, c_idx].item()),
                            int(ext_ctrl[b_idx, t, c_idx].item()),
                        )
                        out[b_idx, t, c_idx] = np.asarray(st, dtype=np.float32)

        return torch.from_numpy(out).to(ngram_repr.device).view(batch_size, seq_len, -1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        controlled_repr, geom_summary = self.encode_ngrams(input_ids)
        res_states = self.rollout_reservoir(controlled_repr)
        combined = torch.cat([res_states, controlled_repr, geom_summary], dim=-1)
        return self.readout(combined)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
        temperature: float = 0.8,
    ) -> list[int]:
        self.eval()
        tokens = list(prompt_ids)
        device = next(self.parameters()).device
        for _ in range(max_new_tokens):
            x = torch.tensor([tokens[-64:]], dtype=torch.long, device=device)
            logits = self(x)[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            tokens.append(next_id)
        return tokens


def tokenize_words(text: str) -> list[str]:
    """Split text into word tokens and punctuation tokens (lowercased)."""
    return text.lower().split()


def tokenize_trigrams(text: str) -> list[tuple[str, str, str]]:
    """Return word trigrams from lowercased whitespace-split tokens."""
    tokens = tokenize_words(text)
    return list(zip(tokens, tokens[1:], tokens[2:]))


def detokenize_words(tokens: list[str]) -> str:
    """Join word and punctuation tokens with clean spacing."""
    out = []
    for tok in tokens:
        if tok in {"<pad>", "<unk>"}:
            continue
        if out and re.match(r"^[^\w\s]+$", tok):
            out[-1] = out[-1] + tok
        elif out:
            out.append(" " + tok)
        else:
            out.append(tok)
    return "".join(out)


# --- Quick smoke test on sample text ---
if __name__ == "__main__":
    with open(input("Filename: "), "r", encoding="utf-8") as file:
        corpus = file.read()

    words = tokenize_words(corpus)
    special_tokens = ["<pad>", "<unk>"]
    vocab = special_tokens + words
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}
    pad_id = stoi["<pad>"]
    unk_id = stoi["<unk>"]

    data = torch.tensor([stoi.get(w, unk_id) for w in words], dtype=torch.long)
    ngram_size = 5
    seq_len = min(24, max(1, len(data) - 1))
    batch_size = 8

    model = NGramReservoirLLM(
        vocab_size=len(vocab),
        pad_id=pad_id,
        n=ngram_size,
        num_cells=16,
        hidden_dim=64,
        num_geom_replicas=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    for step_idx in range(150):
        ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[i : i + seq_len] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in ix])

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, len(vocab)), y.view(-1))
        loss.backward()
        optimizer.step()

        if (step_idx + 1) % 30 == 0:
            print(f"Step {step_idx + 1:3d} | Loss: {loss.item():.4f}")

    prompt_words = tokenize_words("what are the truths about God")
    prompt_ids = [stoi.get(w, unk_id) for w in prompt_words]
    generated_ids = model.generate(prompt_ids, max_new_tokens=80)
    print("Generated:", detokenize_words([itos[i] for i in list(reversed(generated_ids))]))
