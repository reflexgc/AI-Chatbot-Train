"""Byte-level BPE tokenizer — trains on our corpus, encodes text to IDs.

API (see tokenizer/EXPECTED.md):
    train(text, vocab_size) | encode(text) | decode(ids) | save/load(path)

Usage:
    python tokenizer/tokenizer.py --train data/formatted/data_sample.txt --vocab 500 --out tokenizer_trial.json
"""
import argparse
import json
import re
import collections


class BPETokenizer:
    def __init__(self):
        # IDs 0-255 are raw bytes: nothing is ever "unknown".
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}  # (bytes, bytes) -> new id
        self._tok2id = {v: k for k, v in self.vocab.items()}

    @staticmethod
    def _words(text):
        # Words keep their trailing space: "fever " is one unit.
        return [m.group(0).encode('utf-8')
                for m in re.finditer(r'\S+\s*', text)]

    @staticmethod
    def _merge_word(parts, pair, merged):
        out, i = [], 0
        while i < len(parts):
            if i < len(parts) - 1 and parts[i] == pair[0] and parts[i + 1] == pair[1]:
                out.append(merged)
                i += 2
            else:
                out.append(parts[i])
                i += 1
        return out

    def train(self, text, vocab_size=8192):
        counts = collections.Counter(self._words(text))
        splits = {w: [bytes([b]) for b in w] for w in counts}
        while len(self.vocab) < vocab_size:
            pair_counts = collections.Counter()
            for w, c in counts.items():
                s = splits[w]
                for a, b in zip(s, s[1:]):
                    pair_counts[(a, b)] += c
            if not pair_counts:
                break
            (a, b), _ = pair_counts.most_common(1)[0]
            new_id = len(self.vocab)
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = a + b
            for w in counts:
                splits[w] = self._merge_word(splits[w], (a, b), a + b)
        self._tok2id = {v: k for k, v in self.vocab.items()}
        return self

    def encode(self, text):
        ids = []
        for w in self._words(text):
            # Start from single bytes, apply lowest-ranked merge first,
            # repeat until no learned pair remains. Lowest rank = learned
            # earliest = most frequent, so frequent pieces fuse first.
            parts = [bytes([b]) for b in w]
            while len(parts) >= 2:
                best, best_rank = None, None
                for a, b in zip(parts, parts[1:]):
                    r = self.merges.get((a, b))
                    if r is not None and (best_rank is None or r < best_rank):
                        best, best_rank = (a, b), r
                if best is None:
                    break
                parts = self._merge_word(parts, best, best[0] + best[1])
            ids.extend(self._tok2id[p] for p in parts)
        return ids

    def decode(self, ids):
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')

    def save(self, path):
        # latin-1 maps bytes 0-255 one-to-one, so vocab survives JSON.
        data = {'vocab': {str(k): v.decode('latin-1') for k, v in self.vocab.items()},
                'merges': [[a.decode('latin-1'), b.decode('latin-1'), i]
                           for (a, b), i in self.merges.items()]}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = {int(k): v.encode('latin-1') for k, v in data['vocab'].items()}
        tok.merges = {(a.encode('latin-1'), b.encode('latin-1')): i
                      for a, b, i in data['merges']}
        tok._tok2id = {v: k for k, v in tok.vocab.items()}
        return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, help='input .txt corpus')
    ap.add_argument('--vocab', type=int, default=8192)
    ap.add_argument('--out', required=True, help='output .json vocab file')
    args = ap.parse_args()
    text = open(args.train, encoding='utf-8').read()
    tok = BPETokenizer().train(text, vocab_size=args.vocab)
    tok.save(args.out)
    print(f'trained vocab={len(tok.vocab)} merges={len(tok.merges)} -> {args.out}')
    # Self-check on the training text itself:
    ids = tok.encode(text)
    assert tok.decode(ids) == text, 'round-trip failed'
    print(f'self-check OK: {len(text)} chars -> {len(ids)} tokens')


if __name__ == '__main__':
    main()
