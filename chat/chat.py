"""PediGuide terminal chat (hybrid v1): retrieval-first answering.

  python chat/chat.py [--threshold 0.25] [--min-terms 2]

Commands inside the chat:  /score (toggle score display)  /quit
Covered questions -> source-headed answer. Anything else -> the exact
fallback sentence + logged to data/unknowns.log for later authoring.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.retriever import Retriever, load_pairs, answer, FALLBACK

BANNER = (
    'PediGuide (educational tool only - not medical advice).\n'
    'Ask a pediatric health question. Type /quit to exit.'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.25)
    ap.add_argument('--min-terms', type=int, default=2)
    ap.add_argument('--corpus', default=os.path.join('data', 'formatted', 'data.txt'))
    args = ap.parse_args()

    print('loading knowledge base...', flush=True)
    retriever = Retriever(load_pairs(args.corpus))
    print(f'ready: {len(retriever)} indexed answers, '
          f'threshold={args.threshold}, min_terms={args.min_terms}')
    print(BANNER)
    show_score = True
    while True:
        try:
            q = input('\nYou: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye.')
            break
        if not q:
            continue
        if q.lower() in ('/quit', '/exit', 'exit', 'quit'):
            print('Bye.')
            break
        if q.lower() == '/score':
            show_score = not show_score
            print(f'(score display {"on" if show_score else "off"})')
            continue
        text, score = answer(retriever, q, args.threshold, args.min_terms)
        if show_score:
            print(f'[score {score:.3f}'
                  + (' - below threshold, logged for authoring' if text == FALLBACK else '')
                  + ']')
        print('PediGuide:', text)


if __name__ == '__main__':
    main()
