"""PediGuide model: decoder-only Transformer (GPT-style), written from scratch.

Input: token IDs from our BPE tokenizer (NOT characters, NOT text).
Output: next-token logits for every position.

Config lives in one block below. Everything else derives from it.
Usage smoke test:  python model/model.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------- config (Role 1 signs off on these) ----------------
batch_size = 32       # sequences per training batch (Role 7 uses this)
block_size = 256      # max context length the model can look back
n_embd = 256          # width of every vector inside the model
n_head = 4            # attention heads per block (head size = n_embd // n_head)
n_layer = 4           # stacked transformer blocks
dropout = 0.2          # regularization: randomly zeroes activations while training
# (raised 0.1 -> 0.2: our corpus is small, so we regularize harder to slow memorization)
vocab_size = 8192     # MUST match tokenizer.json (overwritten by train.py at runtime)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# --------------------------------------------------------------------

torch.manual_seed(1337)


class Head(nn.Module):
    """One attention head: every position gathers a weighted mix of past positions."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular mask: row t may only look at columns <= t (no peeking ahead).
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)                       # (B,T,hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5      # (B,T,T) raw affinities
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)                            # (B,T,T) weights sum to 1
        wei = self.dropout(wei)
        v = self.value(x)                                       # (B,T,hs)
        return wei @ v                                          # (B,T,hs) weighted mix


class MultiHeadAttention(nn.Module):
    """n_head heads in parallel, concatenated back to n_embd."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-position memory: expands 4x, non-linearity, projects back."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: communicate (attention) then compute (feed-forward).
    Pre-norm + residual: each sub-layer refines x instead of replacing it."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size=vocab_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f'sequence {T} exceeds block_size {self.block_size}'
        x = self.token_embedding_table(idx) + \
            self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_ids=None):
        """Greedy-ish sampling loop. stop_ids (list of token IDs, e.g. encoding of
        '### Instruction:') halts generation the moment the tail matches it,
        so one answer can't roll into the next question. Role 9 extends this."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
            if stop_ids is not None and idx.shape[1] >= len(stop_ids):
                if idx[0, -len(stop_ids):].tolist() == list(stop_ids):
                    idx = idx[:, :-len(stop_ids)]  # strip the stop marker itself
                    break
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    # Smoke test: random IDs in, shapes + generation out. No data needed.
    model = GPTLanguageModel().to(device)
    print(f'{model.num_params() / 1e6:.2f}M parameters on {device}')
    xb = torch.randint(0, vocab_size, (2, 16)).to(device)
    yb = torch.randint(0, vocab_size, (2, 16)).to(device)
    logits, loss = model(xb, yb)
    assert logits.shape == (2, 16, vocab_size), logits.shape
    assert loss.item() > 0 and loss.item() < 30, loss.item()
    gen = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 20)
    assert gen.shape == (1, 21) and gen.max() < vocab_size
    print(f'smoke OK: logits {tuple(logits.shape)}, loss {loss.item():.2f}, generated {gen.shape[1]} tokens')
