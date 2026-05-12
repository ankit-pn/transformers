<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Multi-Head Attention</div>
# Parallel Attention Heads

## 1. Independent Parallel Operations
Each head computes attention independently:

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

All h heads compute simultaneously on GPU -- no sequential dependency.

## 2. GPU Parallelism
The batch and head dimensions combined provide massive parallelism:
```python
# All heads in one batched operation
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
# Q: [batch, h, n, d_k]  K: [batch, h, n, d_k]
# scores: [batch, h, n, n]
```

## 3. Why Parallel?
Attention operations are independent across heads -- no communication needed. This is ideal for GPU parallelism: all heads execute in parallel across SMs.

## 4. Head Independence
No interaction between heads during attention computation. Heads only interact at the concatenation + output projection (W_o) step. This independence is key to efficiency.

## 5. Scaling Heads
More heads = more parallel operations but smaller d_k per head. Diminishing returns beyond ~16 heads for most tasks. GPT-3 uses 96 heads at scale; smaller models use 8-12.
## 📝 Self-Assessment
<details class="mcq"><summary>Are attention heads computed sequentially?</summary><ol type="A"><li>Yes</li><li>No -- all heads compute simultaneously. Q,K,V for all heads are computed in one batched operation</li><li>Only for training</li><li>Depends on model</li></ol><div class="answer">✅ Answer: B -- Q,K,V projections compute all heads at once. Attention on each head is independent and GPU-parallel.</div></details>
<details class="mcq"><summary>GPU parallelism across heads?</summary><ol type="A"><li>Not used</li><li>Combined batch*head dimension gives massive parallelism. All heads run on GPU simultaneously</li><li>Sequential only</li><li>CPU</li></ol><div class="answer">✅ Answer: B -- The [batch, h, n, d_k] tensor processes all heads independently. GPU SMs handle multiple heads in parallel.</div></details>
<details class="mcq"><summary>When do heads interact?</summary><ol type="A"><li>During attention</li><li>Only at concatenation + W_o projection. Zero interaction during the attention computation itself</li><li>Never</li><li>At every step</li></ol><div class="answer">✅ Answer: B -- Heads are independent during QKV and attention. They only mix at the final linear projection (W_o).</div></details>
<details class="mcq"><summary>Optimal number of heads?</summary><ol type="A"><li>As many as possible</li><li>8-16 for most tasks. Diminishing returns beyond 16. Very large models use more (GPT-3: 96)</li><li>1</li><li>100+</li></ol><div class="answer">✅ Answer: B -- H=8-16 is sweet spot. More heads = smaller d_k per head. Trade-off: head diversity vs per-head dimension.</div></details>
<details class="mcq"><summary>Multi-head parallelism type?</summary><ol type="A"><li>Pipeline</li><li>Data parallelism -- each head processes the same input through different projections independently</li><li>Tensor</li><li>Model</li></ol><div class="answer">✅ Answer: B -- Heads are like data-parallel branches processing the same data with different parameters. Naturally parallel.</div></details>
