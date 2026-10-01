"""Standardized sample generator: same prompts, every run, comparable outputs.
Ends vibe-based judging: each checkpoint answers a fixed prompt set that is
saved to a text file for scoring.

Usage (Colab or local):
  python eval/generate_samples.py --ckpt <path> --out eval/samples_run1.txt [--temp 0.3]
  Override a lying/missing config label explicitly:
  python eval/generate_samples.py --ckpt X --embd 128 --heads 4 --layers 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.model import GPTLanguageModel
from tokenizer.tokenizer import BPETokenizer

PROMPTS = [
    'My 3-year-old has a runny nose and a mild cough. What could it be?',
    'My 2-month-old has a fever of 38.5C. What should I do?',
    'My child has vomiting and watery diarrhea. What helps?',
    'What dose of amoxicillin should my 4-year-old take?',
    'My 8-month-old wheezes with a cold. Is it bronchiolitis?',
    'My toddler has a barky cough at midnight. What is it?',
    'Honey-colored crusts around my child nose. What is it?',
    'My baby has a cold and is not feeding well. What now?',
    'Can children take ibuprofen for fever?',
    'My 6-year-old has sudden sore throat and fever without cough. Strep?',
    'What are the signs of dehydration in a child with diarrhea?',
    'My child seized during a fever. What do I do?',
    'Sandpaper rash with sore throat in my 5-year-old. Scarlet fever?',
    'My 4-year-old has itchy blisters all over. Chickenpox?',
    'How much ORS does my child need?',
    'Can my child fly with an ear infection?',
    'My 2-year-old pulls at her ear and has a fever. What could it be?',
    'Does diarrhea in children need antibiotics?',
    'My child coughs every night but seems fine by day. Is that asthma?',
    'What is ORS and when do I use it?',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--temp', type=float, default=0.3)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--max-tokens', type=int, default=100)
    ap.add_argument('--embd', type=int, default=None)
    ap.add_argument('--heads', type=int, default=None)
    ap.add_argument('--layers', type=int, default=None)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    # Explicit flags beat the label (labels lied once already).
    kw = dict(vocab_size=cfg.get('vocab_size', 4096),
              n_embd=args.embd or cfg.get('n_embd', 256),
              n_head=args.heads or cfg.get('n_head', 4),
              n_layer=args.layers or cfg.get('n_layer', 4),
              block_size=cfg.get('block_size', 256))
    print('building model:', kw, flush=True)
    tok = BPETokenizer.load('tokenizer/tokenizer.json')
    model = GPTLanguageModel(**kw)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    import inspect as _inspect
    _supports_stop = 'stop_ids' in _inspect.signature(model.generate).parameters
    stop = tok.encode('\n### Instruction:') if _supports_stop else None

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(f"checkpoint: {args.ckpt} step={ckpt.get('step')} "
                f"val={ckpt.get('val_loss', 'n/a')}\n{'=' * 60}\n")
        for p in PROMPTS:
            prompt = f'### Instruction: {p}\n### Response:'
            ids = torch.tensor([tok.encode(prompt)]).to(device)
            with torch.no_grad():
                gen_kw = dict(temperature=args.temp, top_k=args.top_k)
                if stop is not None:
                    gen_kw['stop_ids'] = stop
                out = model.generate(ids, args.max_tokens, **gen_kw)
            body = tok.decode(out[0].tolist())[len(prompt):].split('### Instruction:')[0]
            f.write(f'\nQ: {p}\nA:{body.strip()}\n{"-" * 60}\n')
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
