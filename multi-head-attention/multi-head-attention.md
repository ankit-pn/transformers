<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Multi-Head Attention</div>
# Multi-Head Attention

## 1. Multiple Parallel Attentions
Instead of one attention function, multi-head attention runs h parallel attention operations:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_o$$
$$\text{head}_i = \text{Attention}(Q W_{q,i}, K W_{k,i}, V W_{v,i})$$

## 2. Why Multiple Heads?
Different heads learn different relationships: one might attend locally, another globally, one to syntax, another to semantics. Averaging them improves robustness.

## 3. Dimension Splitting
$$d_k = d_v = d_{model} / h$$
For d_model=512, h=8: each head works with d_k=64. Total parameter count stays same as single-head with d_model dimensions.

## 4. Computational Cost
$$\text{FLOPs}_{\text{MHA}} = 4 \cdot n \cdot d_{model}^2 + 2 \cdot n^2 \cdot d_{model}$$
Linear in n for projections, quadratic in n for attention. For long sequences, attention dominates.

## 5. Standard Configurations
| Model | d_model | h | d_k |
|---|---|---|---|
| Transformer (base) | 512 | 8 | 64 |
| Transformer (big) | 1024 | 16 | 64 |
| BERT-base | 768 | 12 | 64 |
| GPT-3 | 12288 | 96 | 128 |
## 📝 Self-Assessment
<details class="mcq"><summary>What is multi-head attention?</summary><ol type="A"><li>One big head</li><li>H parallel attention operations with different learned projections -- diverse views of the same sequence</li><li>Sequential attention</li><li>Single head</li></ol><div class="answer">✅ Answer: B -- Multi-head runs h independent attention functions in parallel, each with its own projection matrices.</div></details>
<details class="mcq"><summary>Why multiple heads?</summary><ol type="A"><li>Faster</li><li>Different heads learn different patterns -- local, global, syntactic, semantic. Diversity improves robustness</li><li>Less memory</li><li>Simplicity</li></ol><div class="answer">✅ Answer: B -- Each head captures different relationships. Averaging across diverse patterns improves representation quality.</div></details>
<details class="mcq"><summary>Dimension per head formula?</summary><ol type="A"><li>d_model</li><li>d_k = d_model / h. For 512/8=64 per head. Total param count equals single-head with d_model dims</li><li>d_model * h</li><li>Random</li></ol><div class="answer">✅ Answer: B -- The projection splits d_model across h heads. Each head works in lower dimension; concatenation restores d_model.</div></details>
<details class="mcq"><summary>Standard d_k across most Transformers?</summary><ol type="A"><li>128</li><li>64 -- remarkably consistent across architectures. Good balance of expressiveness and efficiency</li><li>32</li><li>256</li></ol><div class="answer">✅ Answer: B -- d_k=64 is the sweet spot: enough dimension for meaningful dot products without excessive variance.</div></details>
<details class="mcq"><summary>Multi-head FLOPs breakdown?</summary><ol type="A"><li>Attention only</li><li>O(n d_model^2) for projections, O(n^2 d_model) for attention. Projections dominate for short n; attention for long n</li><li>Equal</li><li>Projections always</li></ol><div class="answer">✅ Answer: B -- For n << d_model, projections dominate. For n >> d_model (long context), quadratic attention dominates.</div></details>
