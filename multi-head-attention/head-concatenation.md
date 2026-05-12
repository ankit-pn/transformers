<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Multi-Head Attention</div>
# Head Concatenation

## 1. Merging Head Outputs
After each head computes its attention output, all outputs are concatenated:

$$\text{Concat}(\text{head}_1, ..., \text{head}_h) \in \mathbb{R}^{n \times d_{model}}$$

## 2. Implementation
```python
def combine_heads(x):
    # x: [batch, h, n, d_k]
    batch, h, n, d_k = x.shape
    x = x.transpose(1, 2)          # [batch, n, h, d_k]
    return x.reshape(batch, n, h * d_k)  # [batch, n, d_model]
```

## 3. Restoring Dimension
Each head outputs d_v = d_k dimensions. Concatenating h heads restores d_model:
$$d_{model} = h \times d_k$$

## 4. Then W_o Projection
After concatenation, a final linear projection mixes head outputs:
$$\text{Output} = \text{Concat}(\text{head}_1, ..., \text{head}_h) \cdot W_o$$

W_o enables cross-head interaction -- the only place where head outputs influence each other.

## 5. Why Not Sum?
Concatenation preserves all head information. Summing would force heads to share output space, reducing expressiveness. Concatenation + W_o allows flexible mixing of head contributions.
## 📝 Self-Assessment
<details class="mcq"><summary>Head concatenation purpose?</summary><ol type="A"><li>Reduce dimension</li><li>Merge h head outputs back into d_model dimensions for the next layer</li><li>Split heads</li><li>Add heads</li></ol><div class="answer">✅ Answer: B -- Each head outputs d_k. Concatenating h heads = d_model. Restores the original model dimension.</div></details>
<details class="mcq"><summary>After concatenation, what operation follows?</summary><ol type="A"><li>Nothing</li><li>W_o linear projection -- the only point where head outputs interact</li><li>Dropout</li><li>LayerNorm</li></ol><div class="answer">✅ Answer: B -- W_o mixes concatenated head outputs. This is where cross-head information sharing happens.</div></details>
<details class="mcq"><summary>Why concatenate rather than sum?</summary><ol type="A"><li>Sum is better</li><li>Preserves all head information. Summing forces shared output space, losing per-head detail</li><li>Faster</li><li>Less memory</li></ol><div class="answer">✅ Answer: B -- Concatenation keeps all head contributions separate. W_o then learns an optimal linear combination.</div></details>
<details class="mcq"><summary>Output shape after concatenation?</summary><ol type="A"><li>[batch, h, n, d_k]</li><li>[batch, n, d_model] -- heads merged into model dimension, ready for next layer</li><li>[batch, n, h*d_k]</li><li>[n, d_model]</li></ol><div class="answer">✅ Answer: B -- Transpose + reshape: [batch,h,n,d_k] -> [batch,n,h,d_k] -> [batch,n,d_model]. Restores input shape.</div></details>
<details class="mcq"><summary>Is transpose computationally expensive?</summary><ol type="A"><li>Yes</li><li>Near-zero cost -- transpose only changes memory strides in PyTorch. No data movement</li><li>O(n d)</li><li>O(h n d)</li></ol><div class="answer">✅ Answer: B -- tensor.transpose() is O(1). It reinterprets memory layout without copying data. Only reshape may require contiguity.</div></details>
