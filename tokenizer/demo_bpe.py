"""Demo byte-level BPE tokenizer (reference only — Role 5 writes the real one).
Trains on data/formatted/data_sample.txt and prints expected results.
Usage: python tokenizer/demo_bpe.py
"""
import re
import collections


class DemoBPE:
    def __init__(self):
        # First 256 entries are raw bytes -> nothing can ever be "unknown".
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}  # (bytes, bytes) -> new token id

    @staticmethod
    def _words(text):
        # Split into words, keeping each word's trailing space with it.
        return [m.group(0).encode('utf-8')
                for m in re.finditer(r'\S+\s*', text)]

    def train(self, text, vocab_size=300):
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
                s, out, i = splits[w], [], 0
                while i < len(s):
                    if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                        out.append(a + b)
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                splits[w] = out
        self._tok2id = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        ids = []
        for w in self._words(text):
            parts = [bytes([b]) for b in w]
            while len(parts) >= 2:
                best, best_rank = None, None
                for a, b in zip(parts, parts[1:]):
                    r = self.merges.get((a, b))
                    if r is not None and (best_rank is None or r < best_rank):
                        best, best_rank = (a, b), r
                if best is None:
                    break
                out, i = [], 0
                while i < len(parts):
                    if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best:
                        out.append(parts[i] + parts[i + 1])
                        i += 2
                    else:
                        out.append(parts[i])
                        i += 1
                parts = out
            ids.extend(self._tok2id[p] for p in parts)
        return ids

    def decode(self, ids):
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')


def main():
    text = open('data/formatted/data_sample.txt', encoding='utf-8').read()
    tok = DemoBPE()
    tok.train(text, vocab_size=300)
    print('vocab size:', len(tok.vocab), '| merges learned:', len(tok.merges))

    print('\n-- first 10 learned merges --')
    for (a, b), i in list(tok.merges.items())[:10]:
        print(f'{i}: {a!r} + {b!r} -> {(a + b)!r}')

    print('\n-- how domain words split --')
    for w in ['bronchiolitis', 'paracetamol', 'dehydration', 'fever']:
        ids = tok.encode(w)
        print(f'{w}: {len(ids)} tokens {ids}')

    print('\n-- round-trip: every block --')
    blocks = text.strip().split('\n\n')
    bad = 0
    for n, b in enumerate(blocks):
        if tok.decode(tok.encode(b)) != b:
            bad += 1
            print('FAILED block', n)
    print(f'{len(blocks) - bad}/{len(blocks)} blocks round-tripped')

    print('\n-- hostile inputs --')
    for s in ['fever 38.5\u00b0C', 'amoxicillin-clavulanate 500mg',
              'MY BABY WONT STOP CRYING', 'diarhea for 3 days',
              '\u0645\u0631\u062d\u0628\u0627 \U0001f389']:
        ok = tok.decode(tok.encode(s)) == s
        shown = s.encode('unicode_escape').decode('ascii')
        print(f'{"OK " if ok else "FAIL"} {shown} -> {len(tok.encode(s))} tokens')

    print('\n-- determinism --')
    a, b = tok.encode(blocks[0]), tok.encode(blocks[0])
    print('same input, same ids:', a == b)


if __name__ == '__main__':
    main()
