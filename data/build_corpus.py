"""Corpus builder (Role 4's job, automated).
Merges qa_part*.jsonl -> strips display boilerplate -> 90/10 split ->
data.txt / train.txt / val.txt + stats.

The disclaimer sentence is STRIPPED from training targets on purpose:
it appears in ~97% of answers and the model learns it as content
(frequency mush). The display layer appends it deterministically
instead — guaranteed coverage, zero wasted learning.
Emergency directives ("seek emergency care") are KEPT: only the
boilerplate prefix/sentence is removed.
Usage:  python data/build_corpus.py
"""
import json
import glob
import random
import re
import os

HERE = os.path.dirname(os.path.abspath(__file__))

BOILERPLATE = [
    'This is educational information only, not medical advice.',
    'This is educational information only.',
]


def strip_boilerplate(text):
    out = text
    for b in BOILERPLATE:
        out = out.replace(' ' + b, '').replace(b, '')
    # Keep the urgent tail: "This is educational information only; seek X."
    # -> "Seek X."
    out = re.sub(r'This is educational information only;\s*', '', out)
    out = re.sub(r'\s{2,}', ' ', out).strip()
    return out


def block(r):
    return ('### Instruction: ' + r['instruction'].strip() + '\n'
            + '### Response: ' + strip_boilerplate(r['output']).strip())


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, 'formatted', 'qa_part*.jsonl'))):
        for line in open(f, encoding='utf-8'):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert rows, 'no Q&A parts found'
    assert all(set(r) == {'instruction', 'output'} and r['instruction'].strip()
               and r['output'].strip() for r in rows), 'malformed row detected'

    random.seed(42)
    random.shuffle(rows)


    def is_refusal(r):
        ins = r['instruction'].lower()
        return ins.startswith('what dose') or ins.startswith('deworming medicine dose')

    def is_emergency(r):
        out = r['output'].lower()
        return ('emergency care immediately' in out or 'call emergency now' in out
                or 'go now' in out or 'seek emergency care' in out)

    # Stratified split: each behavior group splits 90/10 independently,
    # so val always tests refusals, emergencies, and routine answers.
    tr, va = [], []
    groups = {'refusal': [], 'emergency': [], 'routine': []}
    for r in rows:
        if is_refusal(r):
            groups['refusal'].append(r)
        elif is_emergency(r):
            groups['emergency'].append(r)
        else:
            groups['routine'].append(r)
    for name, g in groups.items():
        random.shuffle(g)
        cut = max(1, int(len(g) * 0.9)) if len(g) > 1 else 0
        tr.extend(g[:cut])
        va.extend(g[cut:])
        print(f'group {name}: train={len(g[:cut])} val={len(g[cut:])}')
    random.shuffle(tr)
    random.shuffle(va)
    assert not (set(r['instruction'] for r in tr) & set(r['instruction'] for r in va)), \
        'train/val overlap!'

    for name, subset in [('data', rows), ('train', tr), ('val', va)]:
        path = os.path.join(HERE, 'formatted', name + '.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(block(r) for r in subset) + '\n')

    n_tok_words = sum(len(block(r).split()) for r in rows)
    print(f'pairs={len(rows)} train={len(tr)} val={len(va)} overlap=0')
    print(f'approx words={n_tok_words} '
          f'data.txt bytes={os.path.getsize(os.path.join(HERE, "formatted", "data.txt"))}')
    # Safety-shape audit: every split must contain all four behaviors.
    for name, subset in [('train', tr), ('val', va)]:
        outs = ' '.join(r['output'] for r in subset)
        print(f'{name}: refusals={outs.count("do not provide drug doses")} '
              f'emergency={outs.count("immediate medical care") + outs.count("emergency care")}')


if __name__ == '__main__':
    main()
