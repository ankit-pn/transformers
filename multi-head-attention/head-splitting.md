<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Multi-Head Attention</div>
# Head Splitting

## 1. Splitting the Model Dimension
Given input X of shape [batch, n, d_model], split into h heads:

$$X \in \mathbb{R}^{n \times d_{model}} \rightarrow [X_1, ..., X_h] \text{ each } \in \mathbb{R}^{n \times d_k}$$

## 2. Implementation
```python
def split_heads(x, h):
    batch, n, d_model = x.shape
    d_k = d_model // h
    x = x.view(batch, n, h, d_k)   # Split last dim
    return x.transpose(1, 2)         # [batch, h, n, d_k]
```

## 3. Why Split?
Splitting enables parallel attention with different projections per head. Each head sees a different subspace of the full model dimension. Concatenation restores the full representation.

## 4. Projection Perspective
Rather than computing Q, K, V at full d_model then splitting, standard practice projects to d_model and reshapes:
```python
Q = x @ W_q           # [n, d_model]
Q = Q.view(n, h, d_k) # Reshape into heads
```
This is mathematically equivalent to separate per-head projections.
## 📝 Self-Assessment
<details class="mcq"><summary>Head splitting operation?</summary><ol type="A"><li>Random split</li><li>Reshape d_model into h groups of d_k -- each head gets a subspace of the full dimension</li><li>Concatenation</li><li>Pooling</li></ol><div class="answer">✅ Answer: B -- x.view(batch, n, h, d_k) splits the last dimension. Each head sees d_k = d_model/h dimensions.</div></details>
<details class="mcq"><summary>Why split rather than separate projections?</summary><ol type="A"><li>No reason</li><li>Efficiency -- single matrix multiply then reshape computes all heads simultaneously on GPU</li><li>Accuracy</li><li>Memory</li></ol><div class="answer">✅ Answer: B -- One large QKV projection is more GPU-efficient than h separate small ones. Reshape is zero-cost.</div></details>
<details class="mcq"><summary>After splitting, tensor shape?</summary><ol type="A"><li>[batch, n, d_model]</li><li>[batch, h, n, d_k] -- heads dimension added, sequence second, features split per head</li><li>[batch, h, d_k, n]</li><li>[h, batch, n, d_k]</li></ol><div class="answer">✅ Answer: B -- Standard shape for multi-head: [batch, heads, seq_len, d_k]. Each head processes all tokens independently.</div></details>
<details class="mcq"><summary>Are per-head projections independent?</summary><ol type="A"><li>They share weights</li><li>Yes -- Q = X * W_q reshapes into heads. Each head sees different feature subspace</li><li>No, fully shared</li><li>Only W_q shared</li></ol><div class="answer">✅ Answer: B -- Each head operates on different feature dimensions (split from d_model). Independent attention per head.</div></details>
<details class="mcq"><summary>Reshape computational cost?</summary><ol type="A"><li>Expensive</li><li>Zero -- view/reshape only changes metadata (strides), no data movement. Free operation</li><li>O(n d)</li><li>O(h n d)</li></ol><div class="answer">✅ Answer: B -- view() in PyTorch is O(1). It just reinterprets the existing memory layout without copying data.</div></details>
