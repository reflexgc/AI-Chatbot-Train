"""PediGuide training loop (Role 7).
Next-token prediction with AdamW, cosine schedule + warmup, grad clipping,
periodic val eval, checkpoint save/resume. Works on CPU, T1000, or Colab T4.

Usage:
  python train/train.py --steps 50000 --batch 16 --out model/ckpt.pt
  python train/train.py --resume model/ckpt.pt --steps 100000
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from model.model import GPTLanguageModel, block_size, n_embd, n_head, n_layer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(data, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def evaluate(model, data, batch_size, iters=50):
    model.eval()
    losses = []
    for _ in range(iters):
        xb, yb = get_batch(data, batch_size)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def lr_schedule(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=50000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--eval-interval', type=int, default=500)
    ap.add_argument('--ckpt-interval', type=int, default=2000)
    ap.add_argument('--out', default='model/ckpt.pt')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--max-norm', type=float, default=1.0)
    args = ap.parse_args()

    tok_meta = json.load(open('tokenizer/tokenizer.json', encoding='utf-8'))
    vocab_size = len(tok_meta['vocab'])
    train_data = torch.load('data/formatted/train.bin', weights_only=True)
    val_data = torch.load('data/formatted/val.bin', weights_only=True)
    print(f'device={DEVICE} vocab={vocab_size} '
          f'train_tokens={len(train_data)} val_tokens={len(val_data)}', flush=True)

    model = GPTLanguageModel(vocab_size=vocab_size).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']
        print(f"resumed from {args.resume} at step {start_step}, "
              f"loss {ckpt['loss']:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        for pg in opt.param_groups:
            pg['lr'] = lr_schedule(step, args.warmup, args.steps, args.lr)
        xb, yb = get_batch(train_data, args.batch)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        opt.step()

        if step % args.eval_interval == 0 or step == args.steps:
            vl = evaluate(model, val_data, args.batch)
            print(f'step {step}: train {loss.item():.4f} val {vl:.4f} '
                  f'lr {opt.param_groups[0]["lr"]:.2e}', flush=True)
        if step % args.ckpt_interval == 0 or step == args.steps:
            torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                        'step': step, 'loss': loss.item(),
                        'config': {'n_embd': n_embd, 'n_head': n_head,
                                   'n_layer': n_layer, 'block_size': block_size,
                                   'vocab_size': vocab_size}}, args.out)

    print('training done ->', args.out, flush=True)


if __name__ == '__main__':
    main()
