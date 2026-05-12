<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Tokenization</div>
# Byte Pair Encoding

## 1. The BPE Algorithm
Iteratively merge the most frequent pair of adjacent symbols:

```python
vocab = list(all_characters)
while len(vocab) < target_size:
    pair = most_frequent_adjacent_pair(training_corpus)
    vocab.append(pair[0] + pair[1])
    replace_all(training_corpus, pair, merged_token)
```

## 2. Example
Starting with characters: "l", "o", "w", "e", "r", " "
- Step 1: Merge "lo" (most frequent pair) -> "lo"
- Step 2: Merge "ow" -> "low"
- Step 3: Merge "er" -> "er"
- Step 4: Merge "low " -> "low_" (adding space)

## 3. Properties
- Deterministic: same corpus + same merges = same tokens
- Reversible: tokens can be concatenated to recover original text
- Subword coverage: any word = sequence of known subwords
- Frequency-based: more frequent patterns get merged first

## 4. BPE Variants
| Variant | Used By | Difference |
|---|---|---|
| BPE | GPT, RoBERTa | Byte-level |
| WordPiece | BERT | Likelihood-based merging |
| SentencePiece | T5, Llama | Language-agnostic, treats spaces as characters |
| Unigram | XLNet | Probabilistic subword segmentation |
## 📝 Self-Assessment
<details class="mcq"><summary>BPE algorithm?</summary><ol type="A"><li>Random merging</li><li>Iteratively merge most frequent adjacent symbol pairs up to target vocabulary size</li><li>Split words</li><li>Character-level only</li></ol><div class="answer">✅ Answer: B -- BPE starts with characters, repeatedly merges the most common adjacent pair. Controlled by target vocab size.</div></details>
<details class="mcq"><summary>Starting point of BPE?</summary><ol type="A"><li>Words</li><li>Individual characters (or bytes). Merges build up subwords from the bottom</li><li>Sentences</li><li>Phonemes</li></ol><div class="answer">✅ Answer: B -- BPE begins with the character set. Through iterative merging, common subwords like "ing" and "tion" emerge.</div></details>
<details class="mcq"><summary>BPE reversibility?</summary><ol type="A"><li>Cannot reverse</li><li>Fully reversible -- concatenating all BPE tokens reproduces original text exactly</li><li>Lossy</li><li>Partial</li></ol><div class="answer">✅ Answer: B -- Unlike some tokenizers, BPE tokens just need to be concatenated (with space handling) to recover the original string.</div></details>
<details class="mcq"><summary>BERT tokenizer type?</summary><ol type="A"><li>BPE</li><li>WordPiece -- similar to BPE but uses likelihood-based merging instead of pure frequency</li><li>SentencePiece</li><li>Unigram</li></ol><div class="answer">✅ Answer: B -- WordPiece merges pairs that maximize training data likelihood. Functionally similar to BPE with slightly different criteria.</div></details>
<details class="mcq"><summary>SentencePiece advantage?</summary><ol type="A"><li>Faster</li><li>Language-agnostic -- treats input as raw byte stream, no language-specific preprocessing needed</li><li>Smaller vocab</li><li>Better accuracy</li></ol><div class="answer">✅ Answer: B -- SentencePiece works directly on raw text/bytes. No need for language-specific tokenization rules or pre-tokenization.</div></details>
