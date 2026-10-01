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


# American/British spelling variants share one token. Without this,
# "diarrhea" (user) never meets "diarrhoea" (WHO rows) — the single
# most damaging silent mismatch in a medical corpus.
SPELLING = {
    'diarrhea': 'diarrhea', 'diarrhoea': 'diarrhea',
    'behavior': 'behavior', 'behaviour': 'behavior',
    'color': 'color', 'colour': 'color',
    'center': 'center', 'centre': 'center',
    'anemia': 'anemia', 'anaemia': 'anemia',
    'edema': 'edema', 'oedema': 'edema',
    'hemoglobin': 'hemoglobin', 'haemoglobin': 'hemoglobin',
    'estrogen': 'estrogen', 'oestrogen': 'estrogen',
    'esophagus': 'esophagus', 'oesophagus': 'esophagus',
    'pediatric': 'pediatric', 'paediatric': 'pediatric',
    'immunize': 'immunize', 'immunise': 'immunize',
    'normalize': 'normalize', 'normalise': 'normalize',
    'odor': 'odor', 'odour': 'odor',
}


def stem(word):
    """Tiny rule stemmer: coughing->cough, fevers->fever, babies->baby,
    headaches->headach, doses->dos. Singular/plural reunite via ordered
    stripping (plural -s first, then silent -e), so both forms land together.
    Spelling variants canonicalize first (diarrhoea->diarrhea).
    Crude by design, but applied IDENTICALLY to docs and queries, so
    over-stemming only ever merges — never splits — matches."""
    word = SPELLING.get(word, word)
    if word in STOPWORDS:
        return word  # stopwords stay recognizable for filtering
    if len(word) > 5 and word.endswith('ies'):
        return word[:-3] + 'y'
    if len(word) > 5 and word.endswith('ing'):
        return word[:-3]
    if len(word) > 4 and word.endswith('ed'):
        return word[:-2]
    if len(word) > 3 and word.endswith('s'):
        word = word[:-1]
    if len(word) > 7 and word.endswith('ment'):
        word = word[:-4]
    if len(word) > 4 and word.endswith('e'):
        word = word[:-1]
    return word


def tokenize(text):
    return [stem(w) for w in re.findall(r'[a-z0-9]+', text.lower())]


def tokenize_with_originals(text):
    """(original, stemmed) pairs — lets filters see pre-stemming words."""
    words = re.findall(r'[a-z0-9]+', text.lower())
    return [(w, stem(w)) for w in words]


STOPWORDS = frozenset('''a an the and or is are was were be been have has had do does
    did can could should would will shall may might must what which who whom how when
    where why my your his her its our their mine yours hers ours i you he she we they
    them him her us me this that these those in on at to for of with by from as it s t
    not no yes if so than too very just then there here it\'s dont doesn\'t isn\'t'''.split())

# Social intents bypass retrieval entirely: they share vocabulary with real
# rows ("tell me about yourself" ~ "telling people...") yet need no knowledge.
# Keep responses canned, honest, and clearly non-medical.
SOCIAL = [
    (re.compile(r'\b(hello|hi|hey|salam|good morning|good evening)\b'), 
     "Hello! I'm PediGuide. Ask me a children's health question and I'll look it up in our pediatric guide."),
    (re.compile(r'who are you|your name|about yourself|what are you'),
     "I'm PediGuide, a pediatric learning assistant. I answer children's health "
     'questions from our guide — or tell you plainly when I can\'t find an answer.'),
    (re.compile(r'\b(thanks|thank you|shukran)\b'),
     "You're welcome! Anything else about your child's health?"),
    (re.compile(r'\b(bye|goodbye|good night)\b'),
     'Bye — take care!'),
    (re.compile(r'what can you do|help me|how do you work'),
     'Ask any children\'s health question: I search our pediatric guide and answer '
     'from it, or say so honestly when the answer isn\'t there.'),
]


class Retriever:
    """TF-IDF cosine retriever over (instruction -> output) pairs."""

    def __init__(self, docs):
        # docs: list of (instruction, output, source)
        self.docs = docs
        self.vocab = {}
        df = collections.Counter()
        self.tf = []
        for instr, _, _ in docs:
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
            s = sum(((1.0 + math.log(c)) * self.idf[t]) ** 2
                    for t, c in counts.items() if t not in STOPWORDS)
            self.norms.append(math.sqrt(s) or 1.0)

    def _vector(self, text):
        counts = collections.Counter(tokenize(text))
        vec, norm = {}, 0.0
        for t, c in counts.items():
            if t in self.idf and t not in STOPWORDS:
                w = (1.0 + math.log(c)) * self.idf[t]
                vec[t] = w
                norm += w * w
        return vec, math.sqrt(norm) or 1.0

    def content_overlap(self, text, doc_index):
        """Stemmed non-stopword query terms present in the matched doc.
        Guards on the ORIGINAL word too, so mangled stopwords ('these'->'thes')
        can never sneak back in as content."""
        return [s for w, s in tokenize_with_originals(text)
                if w not in STOPWORDS and s not in STOPWORDS
                and s in self.tf[doc_index]]

    def query(self, text, top_k=1):
        qvec, qnorm = self._vector(text)
        scored = []
        for i, (instr, out, _src) in enumerate(self.docs):
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

    def flush():
        if cur is not None and cur[0].strip() and cur[1].strip():
            pairs.append((cur[0], cur[1], cur[2]))

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('### Instruction: '):
                flush()  # also closes old-format blocks without Source lines
                cur = [line[len('### Instruction: '):], '', 'PediGuide pediatric guide']
            elif line.startswith('### Response: ') and cur is not None:
                cur[1] = line[len('### Response: '):]
            elif line.startswith('### Source: ') and cur is not None:
                cur[2] = line[len('### Source: '):].strip() or cur[2]
            elif cur is not None and line.strip():
                # Continuation lines belong to the current response.
                cur[1] += ' ' + line.strip()
    flush()
    return pairs


def log_unknown(question):
    os.makedirs(os.path.dirname(UNKNOWN_LOG), exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(UNKNOWN_LOG, 'a', encoding='utf-8') as f:
        f.write(f'{ts}\t{question.strip()}\n')


def answer(retriever, question, threshold=0.25, min_terms=2):
    for pattern, canned in SOCIAL:
        if pattern.search(question.lower()):
            return canned, 1.0
    scored = retriever.query(question, top_k=1)[0]
    score, _, output, idx = scored
    # Double gate: raw score AND shared informative terms. Short
    # out-of-domain queries ("capital of France?") can clear a pure
    # score bar on function words alone; they carry zero content terms.
    overlap = retriever.content_overlap(question, idx)
    if score < threshold or len(overlap) < min_terms:
        log_unknown(question)
        return FALLBACK, score
    source = retriever.docs[idx][2]
    return f'Based on {source}:\n' + output, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('question')
    ap.add_argument('--threshold', type=float, default=0.2)
    ap.add_argument('--min-terms', type=int, default=2)
    ap.add_argument('--corpus', default=os.path.join(ROOT, 'data', 'formatted', 'data.txt'))
    args = ap.parse_args()
    r = Retriever(load_pairs(args.corpus))
    text, score = answer(r, args.question, args.threshold, args.min_terms)
    print(f'[score {score:.3f}]')
    print(text)


if __name__ == '__main__':
    main()
