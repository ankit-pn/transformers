<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Transformer Blocks</div>
# Input Embeddings

## 1. Token to Vector
Convert discrete token IDs to continuous vectors:

$$E \in \mathbb{R}^{|V| \times d_{model}}$$
$$\text{embed}(token\_id) = E[token\_id] \in \mathbb{R}^{d_{model}}$$

## 2. Scaling
The original Transformer scales embeddings by sqrt(d_model):

$$\text{input} = \text{embed}(tokens) \times \sqrt{d_{model}} + PE$$

This prevents the embeddings from being too small relative to positional encoding.

## 3. Vocabulary Size
| Model | |V| | d_model |
|---|---|---|---|
| Original Transformer | 37K | 512 |
| BERT | 30K | 768 |
| GPT-3 | 50K | 12288 |
| Llama | 32K | 4096 |

## 4. Parameter Impact
$$\text{Params}_{\text{embed}} = |V| \times d_{model}$$

For GPT-3: 50K x 12288 = 614M parameters -- ~20% of total 175B! Vocabulary size significantly impacts model size.

## 5. Subword Tokenization
Most models use subword tokenization (BPE, WordPiece, SentencePiece). Embeddings represent subwords, not whole words. This enables open vocabulary and handles rare words.
## 📝 Self-Assessment
<details class="mcq"><summary>Input embedding dimension?</summary><ol type="A"><li>|V|</li><li>|V| x d_model -- lookup table mapping vocabulary to d_model-dimensional vectors</li><li>d_model x d_model</li><li>n x d_model</li></ol><div class="answer">✅ Answer: B -- A lookup table: each token ID maps to a learned d_model-dimensional vector. Shape is vocab_size x d_model.</div></details>
<details class="mcq"><summary>Why scale embeddings by sqrt(d_model)?</summary><ol type="A"><li>Arbitrary</li><li>Prevents embeddings from being too small relative to positional encoding. Keeps contributions balanced</li><li>For speed</li><li>Required</li></ol><div class="answer">✅ Answer: B -- PE values are ~1.0. Without scaling, learned embeddings (init ~0.02) would be dwarfed by PE.</div></details>
<details class="mcq"><summary>GPT-3 embedding parameter count?</summary><ol type="A"><li>50K</li><li>~614M -- |V|=50K x d_model=12288. ~0.35% of 175B total, but 20% of non-FNN params</li><li>~175B</li><li>~12K</li></ol><div class="answer">✅ Answer: B -- 50K x 12288 = 614M. The embedding layer is one of the largest parameter contributors in large models.</div></details>
<details class="mcq"><summary>Subword tokenization benefit?</summary><ol type="A"><li>Smaller model</li><li>Open vocabulary -- handles any word by decomposing into subwords. No unknown tokens</li><li>Faster</li><li>Simpler</li></ol><div class="answer">✅ Answer: B -- Subwords handle rare/unknown words. "unbelievably" -> "un" + "believ" + "ably". All seen subwords.</div></details>
<details class="mcq"><summary>Embedding scaling formula?</summary><ol type="A"><li>embed / sqrt(d)</li><li>embed * sqrt(d_model) + PE. Scaling prevents PE from dominating the combined representation</li><li>embed + PE</li><li>PE * embed</li></ol><div class="answer">✅ Answer: B -- Multiplied by sqrt(d) to increase magnitude. PE values are ~1; unscaled embeddings would be much smaller initially.</div></details>
