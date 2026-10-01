# Tokenizer — Definition of Done (Role 5)

The teammate's delivery passes iff everything below holds.
Auditor runs the commands; no judgment calls, only pass/fail.

## 1. Required files

- `tokenizer/tokenizer.py` — the implementation, written by hand
- `tests/test_tokenizer.py` — the verification script (runs green)
- `tokenizer/tokenizer.json` — real vocabulary, trained on the final
  corpus ONLY (a trial file on the sample does not count)

## 2. Required API (`tokenizer.py` must expose exactly this)

- `train(text: str, vocab_size: int)` — learn merges from text
- `encode(text: str) -> list[int]` — text to token IDs
- `decode(ids: list[int]) -> str` — token IDs back to text
- `save(path)` / `load(path)` — vocabulary persists and reloads

## 3. Acceptance tests (run on `data/formatted/data_sample.txt`)

All must pass. Reference behavior: `tokenizer/demo_bpe.py`.

1. **Round-trip: 8/8 blocks.** Every `Instruction/Response` block must
   satisfy `decode(encode(block)) == block`, byte-identical.
2. **Hostile inputs, zero crashes:**
   - `fever 38.5°C`
   - `amoxicillin-clavulanate 500mg`
   - `MY BABY WONT STOP CRYING`
   - `diarhea for 3 days` (typo must survive, not be corrected)
   - Arabic + emoji (must round-trip via byte fallback)
3. **Determinism:** `encode(x) == encode(x)` on repeated calls.
4. **No unknown-token crashes, ever.** Byte-level fallback mandatory:
   the first 256 vocab entries are raw bytes 0-255.
5. **Vocab size is a parameter**, demonstrated at two values
   (e.g., 500 on the sample; 8192 and 16384 on the real corpus),
   with tokens-per-block reported for each.

## 4. Real-corpus deliverable (after `data.txt` exists)

- Retrain on the full corpus at the chosen vocab size (8k or 16k).
- Report: total tokens of the encoded corpus, plus how
  `bronchiolitis`, `paracetamol`, `dehydration` split
  (expect 2-4 tokens each — the compression demo for the Doctor).
- Save as `tokenizer/tokenizer.json`. This file is what training reads.

## 5. Ownership interview (spoken, before merge)

1. "Run `decode(encode(...))` live and narrate each step."
2. "Why does a bigger vocab mean fewer tokens per sentence but a
   bigger model?" (Expected: embedding table = vocab x dim.)
3. "What happens on a character the tokenizer never saw?"
   (Expected: degrades to byte tokens, never crashes.)

Fail any section = not merged. Rejected delivery returns with the
failing test number, never a vague "fix it".
