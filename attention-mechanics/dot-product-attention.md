<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Mechanics</div>
# Dot Product Attention

## 1. Simplest Form
$$\text{Attention}(Q, K, V) = \text{softmax}(Q K^T) V$$

## 2. Step by Step
```python
scores = Q @ K.T        # (n_q, n_k) dot products
weights = softmax(scores) # probability distribution
output = weights @ V      # weighted sum
```

## 3. Why Dot Product?
Simple (no params), fast (single matmul on tensor cores), interpretable (larger dot = more similar direction), differentiable through softmax.

## 4. Scaling Problem
$$\text{Var}(q^T k) = d_k \cdot \sigma_q^2 \cdot \sigma_k^2$$
For d_k=512: variance ~512, std ~22.6. Raw dot products reach +-100, saturating softmax. Hence need for scaled version.

## 5. When It Works
Good for small d_k (< 64) where variance is manageable. For larger d_k, use scaled dot product.
## 📝 Self-Assessment
<details class="mcq"><summary>What is dot product attention?</summary><ol type="A"><li>Element-wise</li><li>Attention where compatibility = dot product of Q and K: softmax(Q K^T) V</li><li>Learned params</li><li>Cosine-based</li></ol><div class="answer">✅ Answer: B -- Simplest form: scores = Q @ K^T. No learned parameters in scoring. Pure geometric similarity.</div></details>
<details class="mcq"><summary>Why fail with large d_k?</summary><ol type="A"><li>Memory</li><li>Variance of dot product grows linearly with d_k -- extreme softmax inputs saturate gradients</li><li>Too many params</li><li>Not diff</li></ol><div class="answer">✅ Answer: B -- Var(q^T k) = d_k * sigma^4. For d_k=512, raw scores +-100 making softmax nearly one-hot.</div></details>
<details class="mcq"><summary>Computational pattern?</summary><ol type="A"><li>Element-wise</li><li>Single matrix multiply Q @ K^T -- highly optimized on GPU tensor cores</li><li>Sequential</li><li>Convolution</li></ol><div class="answer">✅ Answer: B -- The batched matmul computes all query-key pairs simultaneously. Workhorse of attention on GPUs.</div></details>
<details class="mcq"><summary>Parameter count of pure dot product attention?</summary><ol type="A"><li>d_k * d_v</li><li>Zero -- dot product attention has no learned parameters. It is purely a mathematical operation</li><li>n * d_k</li><li>d_k + d_v</li></ol><div class="answer">✅ Answer: B -- Pure dot product is parameter-free. Linear projections (W_q, W_k, W_v) in Transformers DO have params.</div></details>
<details class="mcq"><summary>Weight matrix after softmax?</summary><ol type="A"><li>Random</li><li>Each row sums to 1 -- a proper probability distribution over key positions per query</li><li>All zeros</li><li>Diagonal</li></ol><div class="answer">✅ Answer: B -- Softmax normalizes each row independently to a valid probability distribution.</div></details>
