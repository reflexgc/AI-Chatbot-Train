"""Model acceptance tests — the five levels (see plan: Phase B).
Self-running:  python tests/test_model.py   (no pytest needed)

Level 1: shapes      — does it run?
Level 2: causality   — does it cheat? (changing future tokens must not
                       move past predictions)
Level 3: gradients   — can it learn at all? (no NaN, params move)
Level 4: overfit     — can it memorize one batch? (loss must collapse)
Level 5: budget+smoke — param count in budget, generation produces valid IDs
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from model.model import GPTLanguageModel, n_embd, n_head, n_layer, block_size, vocab_size

torch.manual_seed(0)
DEVICE = 'cpu'  # unit tests always run on CPU, everywhere


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        raise AssertionError(name)
    return True


def fresh_model():
    torch.manual_seed(0)
    return GPTLanguageModel().to(DEVICE)


def test_level1_shapes():
    m = fresh_model().eval()
    xb = torch.randint(0, vocab_size, (2, 16))
    logits, loss = m(xb, xb)
    check('L1 logits shape', tuple(logits.shape) == (2, 16, vocab_size), str(tuple(logits.shape)))
    emb = m.token_embedding_table(xb)
    check('L1 embedding shape', tuple(emb.shape) == (2, 16, n_embd), str(tuple(emb.shape)))
    h = m.blocks[0].sa(m.blocks[0].ln1(emb))
    check('L1 attention preserves shape', tuple(h.shape) == (2, 16, n_embd), str(tuple(h.shape)))


def test_level2_causality():
    m = fresh_model().eval()  # eval kills dropout: outputs must be deterministic
    base = torch.randint(0, vocab_size, (1, 16))
    with torch.no_grad():
        ref, _ = m(base)
    altered = base.clone()
    altered[0, -1] = (altered[0, -1] + 1) % vocab_size  # change ONLY the last token
    with torch.no_grad():
        new, _ = m(altered)
    same = torch.equal(ref[:, :-1, :], new[:, :-1, :])
    check('L2 past predictions unchanged by future edit', same,
          'first 15 positions bit-identical' if same else 'LEAK: past moved')
    moved = not torch.equal(ref[:, -1, :], new[:, -1, :])
    check('L2 last position reacts to its own input', moved, 'sanity: model is not frozen')


def test_level3_gradients():
    m = fresh_model().train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    xb = torch.randint(0, vocab_size, (2, 16))
    _, loss = m(xb, xb)
    opt.zero_grad()
    loss.backward()
    grads_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    check('L3 no NaN/inf in gradients', grads_ok)
    before = [p.detach().clone() for p in m.parameters()]
    opt.step()
    moved = any(not torch.equal(b, p.detach()) for b, p in zip(before, m.parameters()))
    check('L3 optimizer step moves parameters', moved)


def test_level4_overfit(steps=200):
    m = fresh_model().train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    xb = torch.randint(0, vocab_size, (4, 32))  # ONE batch, memorized by repetition
    _, l0 = m(xb, xb)
    for _ in range(steps):
        _, loss = m(xb, xb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _, l1 = m(xb, xb)
    check('L4 loss collapses on one batch', l1.item() < 0.5,
          f'{l0.item():.2f} -> {l1.item():.4f} (must end < 0.50)')


def test_level5_budget_and_generate():
    m = fresh_model()
    n = m.num_params()
    check('L5 param budget 5M-20M', 5e6 < n < 20e6, f'{n / 1e6:.2f}M '
          f'(L{n_layer} H{n_head} D{n_embd} V{vocab_size})')
    m.eval()
    with torch.no_grad():
        gen = m.generate(torch.zeros((1, 1), dtype=torch.long), 20)
    check('L5 generation shape/range', tuple(gen.shape) == (1, 21) and gen.max() < vocab_size,
          f'shape {tuple(gen.shape)}, max id {int(gen.max())}')


if __name__ == '__main__':
    test_level1_shapes()
    test_level2_causality()
    test_level3_gradients()
    test_level4_overfit()
    test_level5_budget_and_generate()
    print('ALL MODEL TESTS GREEN')
