<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Optimization</div>
# Vanishing/Exploding Gradients vs RNNs

## 1. RNN Gradient Problem
$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$
Repeated multiplication: eigenvalues < 1 vanish, > 1 explode.

## 2. Transformer Advantage
O(1) path length between any positions:
$$\frac{\partial L}{\partial X_i} = \sum_j \alpha_{ji} \cdot \frac{\partial L}{\partial Y_j}$$
Attention weights provide direct connections -- no repeated multiplication.

## 3. Residual Mitigation
$$\frac{\partial L}{\partial X_l} = \frac{\partial L}{\partial X_{l+1}} \left(I + \frac{\partial \text{Sublayer}}{\partial X_l}\right)$$
Identity term ensures gradient flow through 100+ layers even when sublayer gradients are small.

## 4. Still Need Clipping
Despite better gradient flow, Transformers can still experience gradient spikes from attention softmax saturation or FFN activation extremes. Gradient clipping remains essential.
## 📝 Self-Assessment
<details class="mcq"><summary>RNN gradient vanishing cause?</summary><ol type="A"><li>Short sequences</li><li>Repeated multiplication of recurrent weight matrix T times -- eigenvalues < 1 cause exponential decay</li><li>Large batch</li><li>No cause</li></ol><div class="answer">✅ Answer: B -- Gradient through RNN = product of T Jacobians. Each with eigenvalues < 1 shrinks gradient exponentially.</div></details>
<details class="mcq"><summary>Transformer gradient advantage?</summary><ol type="A"><li>None</li><li>O(1) path length -- attention directly connects any positions. No repeated multiplication</li><li>Same as RNN</li><li>Worse</li></ol><div class="answer">✅ Answer: B -- Attention weight directly links input to output. Gradient path length constant regardless of distance.</div></details>
<details class="mcq"><summary>Residual gradient contribution?</summary><ol type="A"><li>Nothing</li><li>Identity term I -- guarantees gradient can flow through residual path, bypassing sublayer</li><li>Negative</li><li>Sublayer only</li></ol><div class="answer">✅ Answer: B -- dY/dX = I + dSublayer/dX. Even if dSublayer/dX -> 0, identity carries gradient.</div></details>
<details class="mcq"><summary>Why still need gradient clipping?</summary><ol type="A"><li>Not needed</li><li>Attention saturation and FFN extremes can cause gradient spikes -- clipping bounds max update</li><li>Historical</li><li>Only for RNNs</li></ol><div class="answer">✅ Answer: B -- Despite good average flow, transformers can spike. Clipping provides a safety net for outlier gradients.</div></details>
<details class="mcq"><summary>Long-range gradient: Transformer vs RNN?</summary><ol type="A"><li>Same</li><li>Transformer: undiminished. RNN: exponentially decayed. Transformers maintain strong gradients across any distance</li><li>Weaker</li><li>Stronger</li></ol><div class="answer">✅ Answer: B -- Attention creates direct connections. No attenuation with distance. Long-range dependencies get strong gradient signal.</div></details>
