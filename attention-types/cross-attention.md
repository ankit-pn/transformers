<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Types</div>
# Cross-Attention

## 1. Attending Across Sequences
Cross-attention (encoder-decoder attention) lets the decoder attend to the encoder output:

$$\text{CrossAttn}(Y, X) = \text{softmax}\left(\frac{(Y W_q)(X W_k)^T}{\sqrt{d_k}}\right) (X W_v)$$

Q from decoder (target), K and V from encoder (source).

## 2. Role in Seq2Seq
The decoder queries the source for relevant information at each generation step. Translating "I love you" -> "Je t\'aime": when generating "aime", the decoder attends to "love".

## 3. Q, K, V Assignment
- **Q**: Decoder hidden states (what does the decoder need?)
- **K**: Encoder outputs (where is it in the source?)
- **V**: Encoder outputs (what information to extract?)

## 4. vs Self-Attention
| Aspect | Self-Attention | Cross-Attention |
|---|---|---|
| Q source | Same sequence | Decoder |
| K,V source | Same sequence | Encoder |
| Shape | n x n | m x n |
| Purpose | Context within seq | Source-target alignment |

## 5. Modern Usage
Cross-attention is essential in encoder-decoder Transformers (T5, BART). Decoder-only models (GPT) omit cross-attention entirely, using only self-attention.
## 📝 Self-Assessment
<details class="mcq"><summary>What is cross-attention?</summary><ol type="A"><li>Same-sequence attention</li><li>Decoder attends to encoder output -- Q from decoder, K/V from encoder</li><li>Self-attention</li><li>Random attention</li></ol><div class="answer">✅ Answer: B -- Cross-attention connects decoder (target) to encoder (source). Key mechanism in encoder-decoder Transformers.</div></details>
<details class="mcq"><summary>Q, K, V sources in cross-attention?</summary><ol type="A"><li>All encoder</li><li>Q = decoder states, K/V = encoder outputs. Decoder queries the source for info</li><li>All decoder</li><li>Mixed</li></ol><div class="answer">✅ Answer: B -- Decoder asks (Q), encoder provides (K/V). Each decoder step queries for relevant source information.</div></details>
<details class="mcq"><summary>Cross-attention purpose in translation?</summary><ol type="A"><li>Not used</li><li>When generating target word, decoder attends to relevant source words -- "aime" attends to "love"</li><li>Grammar</li><li>Style</li></ol><div class="answer">✅ Answer: B -- Cross-attention learns soft alignments between source and target tokens during translation.</div></details>
<details class="mcq"><summary>Decoder-only models (GPT) use cross-attention?</summary><ol type="A"><li>Yes, always</li><li>No -- decoder-only models have no encoder to attend to. They use only causal self-attention</li><li>Sometimes</li><li>Required</li></ol><div class="answer">✅ Answer: B -- GPT and other decoder-only models omit cross-attention entirely. They rely solely on self-attention for context.</div></details>
<details class="mcq"><summary>Cross-attention shape?</summary><ol type="A"><li>n x n</li><li>m x n -- m decoder queries attend to n encoder keys. Typically m != n</li><li>n x d</li><li>d x d</li></ol><div class="answer">✅ Answer: B -- Cross-attention matrix is rectangular: decoder length x encoder length. Different from square self-attention.</div></details>
