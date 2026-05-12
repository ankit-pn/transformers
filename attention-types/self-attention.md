<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Types</div>
# Self-Attention

## 1. Attending Within a Sequence
Self-attention computes attention where queries, keys, and values all come from the SAME sequence:

$$\text{SelfAttention}(X) = \text{softmax}\left(\frac{(X W_q)(X W_k)^T}{\sqrt{d_k}}\right) (X W_v)$$

## 2. What It Learns
Each token attends to every other token (including itself), building context-aware representations. "The animal didn\'t cross the street because it was too tired" -- self-attention lets "it" attend to "animal".

## 3. Relationship Modeling
| Relationship | Attention Pattern |
|---|---|
| Syntactic (subject-verb) | Cross-position |
| Coreference (pronoun refs) | Cross-position |
| Semantic (related concepts) | Cross-position |
| Self-reference | Diagonal |

## 4. Computational Cost
$$\text{FLOPs} = O(n^2 d)$$ for sequence length n. The quadratic dependence on n is the main limitation.

## 5. Key Property
$$\text{Output}_i = f(\text{all tokens in sequence})$$
Each output position depends on the entire input. This is what enables rich context modeling but also creates the O(n^2) cost.
## 📝 Self-Assessment
<details class="mcq"><summary>What is self-attention?</summary><ol type="A"><li>Cross-sequence attention</li><li>Attention where Q, K, V all come from the SAME sequence -- each token attends to every token</li><li>Encoder-decoder</li><li>External attention</li></ol><div class="answer">✅ Answer: B -- Self-attention = same sequence for Q, K, V. Each token builds a context-aware representation from the full sequence.</div></details>
<details class="mcq"><summary>Self-attention output dependency?</summary><ol type="A"><li>Only adjacent tokens</li><li>Each output depends on ALL input tokens -- this is the source of both power and O(n^2) cost</li><li>Previous tokens only</li><li>Next tokens only</li></ol><div class="answer">✅ Answer: B -- Output_i = f(all tokens). This creates rich context but quadratic scaling with sequence length.</div></details>
<details class="mcq"><summary>What does self-attention enable linguistically?</summary><ol type="A"><li>Nothing special</li><li>Coreference resolution -- "it" can attend to "animal". Syntactic and semantic relationships across positions</li><li>Only translation</li><li>Only classification</li></ol><div class="answer">✅ Answer: B -- Self-attention learns token-token relationships. Pronoun resolution, syntactic dependencies, semantic associations.</div></details>
<details class="mcq"><summary>Computational complexity of self-attention?</summary><ol type="A"><li>O(n d)</li><li>O(n^2 d) -- quadratic in sequence length. Each of n queries attends to all n keys</li><li>O(n d^2)</li><li>O(n log n)</li></ol><div class="answer">✅ Answer: B -- n queries x n keys = n^2 attention scores. Each is a d-dimensional dot product. O(n^2 d) total.</div></details>
<details class="mcq"><summary>What dominates the attention matrix in early layers?</summary><ol type="A"><li>Random</li><li>Diagonal -- tokens attend strongly to themselves. Cross-position attention develops in deeper layers</li><li>Uniform</li><li>Off-diagonal</li></ol><div class="answer">✅ Answer: B -- Early layers: self-attention dominates. Deeper layers: cross-position attention for syntax/semantics develops.</div></details>
