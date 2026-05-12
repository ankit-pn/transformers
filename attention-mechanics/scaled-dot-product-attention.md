<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Mechanics</div>
# Scaled Dot Product Attention

## 1. The Formula
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

## 2. Why Scale?
$$\text{Var}(q^T k) = d_k \quad \text{Std} = \sqrt{d_k}$$
$$\text{Var}(q^T k / \sqrt{d_k}) = 1$$
Normalizes variance to 1. Prevents softmax saturation.

## 3. Gradient Problem
Without scaling, for large d_k: softmax([100,...]) ~ [1,0,0] -- nearly one-hot, near-zero gradients. Scaling keeps values where softmax has meaningful curvature.

## 4. Implementation
```python
d_k = Q.size(-1)
scores = Q @ K.T / math.sqrt(d_k)
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
return F.softmax(scores, dim=-1) @ V
```

## 5. Effect
| d_k | Unscaled | Scaled | Entropy |
|---|---|---|---|
| 64 | +-8 | +-1 | High |
| 512 | +-23 | +-1 | High |
| 1024 | +-32 | +-1 | High |
Scaling preserves diverse attention across all dimensions.
## 📝 Self-Assessment
<details class="mcq"><summary>Why scale by sqrt(d_k)?</summary><ol type="A"><li>Arbitrary</li><li>Normalizes dot product variance to 1 -- prevents softmax saturation for large d_k</li><li>Numerical stability</li><li>Speed</li></ol><div class="answer">✅ Answer: B -- Var(q^T k) = d_k. Dividing by sqrt(d_k) normalizes to 1. Without scaling, large d_k saturates softmax.</div></details>
<details class="mcq"><summary>Without scaling for d_k=512?</summary><ol type="A"><li>Nothing</li><li>Raw scores reach +-23 std -- softmax becomes nearly one-hot with near-zero gradients</li><li>Faster convergence</li><li>Better</li></ol><div class="answer">✅ Answer: B -- Scores outside [-3,3] have near-zero softmax curvature. Gradients vanish. Attention dies.</div></details>
<details class="mcq"><summary>Variance of scaled dot product?</summary><ol type="A"><li>d_k</li><li>1 -- scaling by sqrt(d_k) normalizes variance exactly to 1</li><li>sqrt(d_k)</li><li>d_k/2</li></ol><div class="answer">✅ Answer: B -- Var(q^T k / sqrt(d_k)) = d_k / d_k = 1. Distribution becomes approximately standard normal.</div></details>
<details class="mcq"><summary>What does masking do?</summary><ol type="A"><li>Not needed</li><li>Sets masked positions to -inf before softmax -- those get ~0 attention weight</li><li>Accuracy</li><li>Reduce params</li></ol><div class="answer">✅ Answer: B -- -1e9 in masked positions makes exp(-1e9)~0. Essential for causal and padding masks.</div></details>
<details class="mcq"><summary>Entropy: scaled vs unscaled for d_k=1024?</summary><ol type="A"><li>Same</li><li>Unscaled: low entropy (one position dominates). Scaled: high entropy (diverse attention)</li><li>Unscaled higher</li><li>Both low</li></ol><div class="answer">✅ Answer: B -- Scaling prevents single position from dominating. Higher entropy means richer information aggregation.</div></details>
