<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Mechanics</div>
# Softmax Normalization in Attention

## 1. The Softmax
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$

## 2. Why Softmax?
Non-negative (positive), sums to 1 (convex), differentiable (smooth gradients), exponential amplifies differences (focused), translation invariant.

## 3. Temperature
$$\alpha_i = \frac{\exp(e_i / \tau)}{\sum \exp(e_j / \tau)}$$
tau->0: argmax (hard). tau=1: standard. tau->inf: uniform (equal).

## 4. Numerical Stability
```python
x_shifted = x - x.max()   # Prevent overflow
return (x_shifted.exp() / x_shifted.exp().sum())
```

## 5. Gradient
$$\frac{\partial \alpha_i}{\partial e_j} = \alpha_i(\delta_{ij} - \alpha_j)$$
Interconnected: changing one score affects all weights.
## 📝 Self-Assessment
<details class="mcq"><summary>Why softmax for attention?</summary><ol type="A"><li>Simplicity</li><li>Non-negative, sums to 1, differentiable, exponential amplifies differences -- ideal for weighted average</li><li>Required</li><li>Historical</li></ol><div class="answer">✅ Answer: B -- Softmax produces proper distribution. Exponential amplifies differences for focused attention.</div></details>
<details class="mcq"><summary>Temperature tau < 1 effect?</summary><ol type="A"><li>Smooths</li><li>Sharpens -- more like argmax. High scores dominate; attention more focused</li><li>No effect</li><li>Widens</li></ol><div class="answer">✅ Answer: B -- Small tau divides scores, widening gaps. The max dominates; entropy decreases.</div></details>
<details class="mcq"><summary>Why subtract max before softmax?</summary><ol type="A"><li>Accuracy</li><li>Numerical stability -- prevents exp(1000)=inf. Mathematically equivalent: softmax(x)=softmax(x-max)</li><li>Required</li><li>Speed</li></ol><div class="answer">✅ Answer: B -- exp(overflow)=inf in float32. Subtracting max shifts down while preserving relative differences.</div></details>
<details class="mcq"><summary>Softmax gradient: one score change affects...</summary><ol type="A"><li>Only that weight</li><li>All weights -- increasing one decreases others proportionally. Total change sums to zero</li><li>Only adjacent</li><li>No gradient</li></ol><div class="answer">✅ Answer: B -- d(alpha_i)/d(e_j) = alpha_i*(delta_ij - alpha_j). Zero-sum: total weight change is zero.</div></details>
<details class="mcq"><summary>tau -> infinity effect?</summary><ol type="A"><li>Argmax</li><li>Uniform distribution -- all positions equal. Information from scoring is lost</li><li>One-hot</li><li>Random</li></ol><div class="answer">✅ Answer: B -- Flat scores = equal weights. Model cannot distinguish relevant from irrelevant positions.</div></details>
