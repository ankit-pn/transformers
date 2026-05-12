<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Mechanics</div>
# Weighted Sum Aggregation

## 1. The Aggregation
$$\mathbf{c}_t = \sum_{i=1}^{n} \alpha_{ti} \mathbf{v}_i$$

Output is convex combination of values, weighted by attention.

## 2. Properties
Convex (alpha_i >= 0, sum=1). Differentiable: d(c_t)/d(v_i) = alpha_i. Dimension: d_v independent of sequence length.

## 3. Vectorized
$$\mathbf{C} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{n_q \times d_v}$$
Where A = softmax(QK^T/sqrt(d_k)). Single matrix multiply.

## 4. Interpretation
Soft lookup: high alpha_i means value v_i contributes more. If alpha=[0,0,1,0], output = v_3 exactly. Smooth: small alpha changes = small output changes.

## 5. Cost
FLOPs = 2 x n_q x n_k x d_v. For self-attention (n_q=n_k=n): O(n^2 d_v). Second O(n^2) term (after scoring).
## 📝 Self-Assessment
<details class="mcq"><summary>What is weighted sum?</summary><ol type="A"><li>Element-wise</li><li>Weighted combination of values by attention weights: c_t = sum(alpha_{ti} * v_i)</li><li>Matmul only</li><li>Average</li></ol><div class="answer">✅ Answer: B -- Output is weighted average where each value contributes proportionally to its attention weight.</div></details>
<details class="mcq"><summary>Convex combination?</summary><ol type="A"><li>Any linear</li><li>Weights >= 0 and sum to 1. Output guaranteed to lie in convex hull of inputs</li><li>Equal weights</li><li>Negative</li></ol><div class="answer">✅ Answer: B -- Convex means output is between the value vectors. Smooth interpolation of inputs.</div></details>
<details class="mcq"><summary>Gradient of output w.r.t value?</summary><ol type="A"><li>Complex</li><li>Simply alpha_i -- the attention weight itself is the gradient. Beautiful simplicity</li><li>Depends on Q</li><li>Depends on K</li></ol><div class="answer">✅ Answer: B -- d(c_t)/d(v_i) = alpha_i. No chain rule. Attention weight IS the gradient for aggregation.</div></details>
<details class="mcq"><summary>One alpha_i=1, others=0?</summary><ol type="A"><li>Error</li><li>Output = v_i exactly -- hard attention selecting one value. Rare with softmax but possible</li><li>Zero output</li><li>Sum of all</li></ol><div class="answer">✅ Answer: B -- Perfect focus gives that values vector. Gradients flow to all positions via softmax.</div></details>
<details class="mcq"><summary>FLOPs for n=1024, d_v=64?</summary><ol type="A"><li>~1B</li><li>~134M -- 2 x 1024^2 x 64. Same order as scoring; both O(n^2 d)</li><li>~10M</li><li>~1T</li></ol><div class="answer">✅ Answer: B -- A @ V has identical complexity to Q @ K^T. Both O(n^2 d), source of quadratic bottleneck.</div></details>
