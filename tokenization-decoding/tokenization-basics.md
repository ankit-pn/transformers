<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Tokenization</div>
# Tokenization Basics

## 1. Text to Numbers
Tokenization converts raw text into a sequence of integer token IDs:

$$\text{"Hello world!"} \rightarrow [15496, 995, 0]$$

## 2. Token Types
| Level | Example | Pros | Cons |
|---|---|---|---|
| Character | H-e-l-l-o | Tiny vocab, no OOV | Long sequences |
| Word | Hello | Semantic | Large vocab, OOV |
| Subword | Hel-lo | Balanced | More complex |
| Byte | 0x48-0x65-... | Universal | Very long |

## 3. Special Tokens
| Token | Purpose |
|---|---|
| [CLS] / <s> | Sequence start |
| [SEP] / </s> | Sequence end / separator |
| [PAD] | Padding to uniform length |
| [UNK] | Unknown tokens |
| [MASK] | Masked language modeling |

## 4. Vocabulary Size Tradeoff
Larger vocab = shorter sequences (each subword carries more meaning) but more parameters in embedding. Smaller vocab = longer sequences but fewer params. Sweet spot: 30K-50K tokens.
## 📝 Self-Assessment
<details class="mcq"><summary>Tokenization purpose?</summary><ol type="A"><li>Text compression</li><li>Convert raw text to integer sequences that the model can process</li><li>Translation</li><li>Summarization</li></ol><div class="answer">✅ Answer: B -- Language models operate on numbers, not strings. Tokenization bridges human text and model input space.</div></details>
<details class="mcq"><summary>Subword tokenization advantage?</summary><ol type="A"><li>Simpler</li><li>Balanced -- handles rare words via decomposition (un-believ-able), no unknown tokens, moderate vocab size</li><li>Larger vocab</li><li>Longer sequences</li></ol><div class="answer">✅ Answer: B -- Subwords decompose unknown/rare words into known pieces. Open vocabulary: any word can be represented.</div></details>
<details class="mcq"><summary>[UNK] token purpose?</summary><ol type="A"><li>Start of sequence</li><li>Unknown token -- returned when a token is not in the vocabulary. Minimized by subword tokenization</li><li>End of sequence</li><li>Padding</li></ol><div class="answer">✅ Answer: B -- When a character sequence has no vocabulary match, [UNK] is produced. Subword tokenization largely eliminates [UNK].</div></details>
<details class="mcq"><summary>Vocabulary size sweet spot?</summary><ol type="A"><li>10K</li><li>30K-50K -- balances sequence length (larger = shorter sequences) against embedding parameters</li><li>100K+</li><li>1K</li></ol><div class="answer">✅ Answer: B -- 30-50K subwords cover most languages well. Too small = long sequences; too large = many embedding parameters.</div></details>
<details class="mcq"><summary>[PAD] token usage?</summary><ol type="A"><li>Start token</li><li>Padding -- fills shorter sequences to uniform batch length. Attention mask prevents attending to [PAD]</li><li>End token</li><li>Classification</li></ol><div class="answer">✅ Answer: B -- Batching requires equal-length tensors. [PAD] fills short sequences; attention mask ensures [PAD] is ignored.</div></details>
