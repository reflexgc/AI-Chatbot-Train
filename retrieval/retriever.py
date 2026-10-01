"""Hybrid v1 retrieval layer: find the answer, don't predict it.

Pipeline per user question:
  1. Retriever scores the question against every training question (TF-IDF +
     cosine, hand-rolled, no dependencies).
  2. Score >= threshold -> answer = matched row's output, lightly wrapped
     with a source header. Facts always come from the corpus, never invented.
  3. Score < threshold  -> exact fallback sentence + the question is appended
     to data/unknowns.log (the authoring queue for later).

Usage:
  python retrieval/answer.py "My baby has a cold and is not feeding well."
"""
import argparse
import collections
import math
import os
import re
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FALLBACK = "I can't find an answer for your request."
UNKNOWN_LOG = os.path.join(ROOT, 'data', 'unknowns.log')


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


STOPWORDS = frozenset('''a an the and or is are was were be been have has had do does
    did can could should would will shall may might must what which who whom how when
    where why my your his her its our their mine yours hers ours i you he she we they
    them him her us me this that these those in on at to for of with by from as it s t
    not no yes if so than too very just then there here it\'s dont doesn\'t isn\'t'''.split())


class Retriever:
    """TF-IDF cosine retriever over (instruction -> output) pairs."""

    def __init__(self, docs):
        # docs: list of (instruction, output)
        self.docs = docs
        self.vocab = {}
        df = collections.Counter()
        self.tf = []
        for instr, _ in docs:
            toks = tokenize(instr)
            counts = collections.Counter(toks)
            self.tf.append(counts)
            for t in counts:
                df[t] += 1
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
        n = len(docs)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        # Informative = appears in fewer than half the docs. (Median-IDF was
        # tried first: with Zipfian vocab the median sits among rare terms,
        # so even 'cold' failed to qualify. Document frequency is robust.)
        self.df = dict(df)
        self._n_docs = n
        self.norms = []
        for counts in self.tf:
            s = sum(((1.0 + math.log(c)) * self.idf[t]) ** 2 for t, c in counts.items())
            self.norms.append(math.sqrt(s) or 1.0)

    def _vector(self, text):
        counts = collections.Counter(tokenize(text))
        vec, norm = {}, 0.0
        for t, c in counts.items():
            if t in self.idf:
                w = (1.0 + math.log(c)) * self.idf[t]
                vec[t] = w
                norm += w * w
        return vec, math.sqrt(norm) or 1.0

    def content_overlap(self, text, doc_index):
        """Non-stopword query terms present in the matched doc. Explicit
        stopword list (not statistics): deterministic and explainable —
        function words can never vote, in any corpus."""
        return [t for t in set(tokenize(text))
                if t not in STOPWORDS and t in self.tf[doc_index]]

    def query(self, text, top_k=1):
        qvec, qnorm = self._vector(text)
        scored = []
        for i, (instr, out) in enumerate(self.docs):
            counts = self.tf[i]
            dot = 0.0
            for t, w in qvec.items():
                if t in counts:
                    dot += w * (1.0 + math.log(counts[t])) * self.idf[t]
            scored.append((dot / (qnorm * self.norms[i]), instr, out, i))
        scored.sort(key=lambda s: -s[0])
        return scored[:top_k]

    def __len__(self):
        return len(self.docs)


def load_pairs(path):
    pairs, cur = [], None
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('### Instruction: '):
                cur = [line[len('### Instruction: '):], '']
            elif line.startswith('### Response: ') and cur is not None:
                cur[1] = line[len('### Response: '):]
                pairs.append(tuple(cur))
                cur = None
            elif cur is not None and line.strip():
                # Continuation lines belong to the current response.
                cur[1] += ' ' + line.strip()
    return pairs


def log_unknown(question):
    os.makedirs(os.path.dirname(UNKNOWN_LOG), exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(UNKNOWN_LOG, 'a', encoding='utf-8') as f:
        f.write(f'{ts}\t{question.strip()}\n')


def answer(retriever, question, threshold=0.25, min_terms=2):
    scored = retriever.query(question, top_k=1)[0]
    score, _, output, idx = scored
    # Double gate: raw score AND shared informative terms. Short
    # out-of-domain queries ("capital of France?") can clear a pure
    # score bar on function words alone; they carry zero content terms.
    overlap = retriever.content_overlap(question, idx)
    if score < threshold or len(overlap) < min_terms:
        log_unknown(question)
        return FALLBACK, score
    return 'Based on our pediatric guide:\n' + output, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('question')
    ap.add_argument('--threshold', type=float, default=0.25)
    ap.add_argument('--min-terms', type=int, default=2)
    ap.add_argument('--corpus', default=os.path.join(ROOT, 'data', 'formatted', 'data.txt'))
    args = ap.parse_args()
    r = Retriever(load_pairs(args.corpus))
    text, score = answer(r, args.question, args.threshold, args.min_terms)
    print(f'[score {score:.3f}]')
    print(text)


if __name__ == '__main__':
    main()
