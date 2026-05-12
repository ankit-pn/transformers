<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Mechanics</div>
# Alignment Scores

## 1. Definition
$$e_{ij} = \text{score}(\mathbf{s}_i, \mathbf{h}_j)$$

## 2. Score Functions
| Name | Formula | Params |
|---|---|---|
| Additive (Bahdanau) | v^T tanh(W1 s_i + W2 h_j) | W1,W2,v |
| Dot Product (Luong) | s_i^T h_j | None |
| Scaled Dot Product | (s_i^T h_j) / sqrt(d_k) | None |
| General (Luong) | s_i^T W_a h_j | W_a |

## 3. Additive vs Dot
Additive: more expressive (learned nonlinearity), works regardless of dimension. Dot: simpler, faster (no params), but large dot products for high d. Scaled dot product fixes this.

## 4. Scores to Weights
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k} \exp(e_{ik})}$$

Softmax converts unbounded scores to probability distribution summing to 1.
## 📝 Self-Assessment
<details class="mcq"><summary>What are alignment scores?</summary><ol type="A"><li>Output probs</li><li>Compatibility scores between decoder and encoder states -- raw attention values before softmax</li><li>Loss values</li><li>Embeddings</li></ol><div class="answer">✅ Answer: B -- e_{ij} = score(s_i, h_j) measures relevance of encoder j for decoder step i.</div></details>
<details class="mcq"><summary>Bahdanau vs Luong attention?</summary><ol type="A"><li>Same</li><li>Bahdanau: additive (learned params). Luong: dot product (parameter-free). Additive more expressive</li><li>Luong additive</li><li>Bahdanau dot</li></ol><div class="answer">✅ Answer: B -- Bahdanau uses learned MLP for scoring. Luong simpler. Scaled dot in Transformers descends from Luong.</div></details>
<details class="mcq"><summary>Zero-parameter score function?</summary><ol type="A"><li>Additive</li><li>Dot product -- e_{ij} = s_i^T * h_j. Pure geometric similarity</li><li>General</li><li>Location-based</li></ol><div class="answer">✅ Answer: B -- Dot product attention has no learnable parameters in the scoring function itself.</div></details>
<details class="mcq"><summary>Why softmax on alignment scores?</summary><ol type="A"><li>Nonlinearity</li><li>Converts unbounded scores to probability distribution summing to 1 -- proper convex combination</li><li>Reduce dim</li><li>Speed</li></ol><div class="answer">✅ Answer: B -- Softmax normalizes to [0,1] with sum=1. Makes attention output a convex combination of values.</div></details>
<details class="mcq"><summary>Why scaled dot product divides by sqrt(d_k)?</summary><ol type="A"><li>Nonlinearity</li><li>Prevents softmax saturation for large d_k -- dot products grow with dimension</li><li>Numerical stability</li><li>Reduce compute</li></ol><div class="answer">✅ Answer: B -- For d_k=512, raw dot products reach +-23 std. Dividing by sqrt(d_k) normalizes to +-1.</div></details>
