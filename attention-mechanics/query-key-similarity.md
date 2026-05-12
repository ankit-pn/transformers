<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Mechanics</div>
# Query-Key Similarity Computation

## 1. Computing Relevance
$$\text{sim}(\mathbf{q}, \mathbf{k}) = f(\mathbf{q}, \mathbf{k})$$

## 2. Common Functions
| Function | Formula | Properties |
|---|---|---|
| Dot Product | q^T k | Simple, no params, O(d) |
| Cosine | q^Tk/(||q||||k||) | Bounded [-1,1], scale-invariant |
| Scaled Dot | q^Tk/sqrt(d_k) | Prevents large values |
| Additive | v^T tanh(W_q q + W_k k) | Learned nonlinearity |
| Euclidean | -||q - k||^2 | Distance-based |

## 3. Vectorized
$$\mathbf{E} = \mathbf{Q} \mathbf{K}^T \in \mathbb{R}^{n \times n}$$
Single matrix multiply. E_{ij} = q_i^T k_j. Fully GPU-parallelized.

## 4. Cost
FLOPs = 2 x n_q x n_k x d_k. For n=1024, d_k=64: ~134M. Tiny vs full transformer.
## 📝 Self-Assessment
<details class="mcq"><summary>Purpose of Q-K similarity?</summary><ol type="A"><li>Output gen</li><li>Determines attention weights -- how much each Key position contributes to output for each Query</li><li>Param init</li><li>Loss</li></ol><div class="answer">✅ Answer: B -- Higher similarity = more attention weight = more contribution from that position.</div></details>
<details class="mcq"><summary>Why compute QK^T as matrix multiply?</summary><ol type="A"><li>Accuracy</li><li>GPU efficiency -- one batched matmul computes all query-key pairs simultaneously on tensor cores</li><li>Memory</li><li>Required</li></ol><div class="answer">✅ Answer: B -- Q(n x d) x K^T(d x n) = E(n x n). One operation replaces n^2 sequential dot products.</div></details>
<details class="mcq"><summary>Cosine vs dot product?</summary><ol type="A"><li>Same</li><li>Cosine normalizes by magnitude (bounded [-1,1]). Dot grows with magnitude. Cosine is scale-invariant</li><li>Cosine better</li><li>Dot bounded</li></ol><div class="answer">✅ Answer: B -- Cosine = dot/(||q||||k||). Prevents large activations from dominating. Dot with scaling works in Transformers.</div></details>
<details class="mcq"><summary>Euclidean distance as similarity?</summary><ol type="A"><li>Not possible</li><li>-||q - k||^2 -- negative distance. Smaller distance = larger similarity (near 0). Valid but uncommon</li><li>Always zero</li><li>Same as dot</li></ol><div class="answer">✅ Answer: B -- Negative distance: when q and k are close, similarity is near 0 (max). Inverse of typical similarity.</div></details>
<details class="mcq"><summary>FLOPs for n=1024, d_k=64?</summary><ol type="A"><li>~1B</li><li>~134M -- 2 x 1024 x 1024 x 64. Negligible versus total transformer FLOPs</li><li>~10M</li><li>~1T</li></ol><div class="answer">✅ Answer: B -- O(n^2 d). For typical sizes, <1% of total transformer compute, which is dominated by the FFN block.</div></details>
