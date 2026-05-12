<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Optimization</div>
# KV Caching

## 1. The Core Optimization
Cache Key and Value tensors from previous steps:

$$\text{KV\_Cache}_t = \{(K_1,V_1), (K_2,V_2), ..., (K_t,V_t)\}$$

Step t+1 only computes Q for the new token and uses cached K,V.

## 2. Complexity Reduction
Without cache: O(N^3 d) total FLOPs for N tokens.
With cache: O(N^2 d) total FLOPs.
For N=100: ~33x savings.

## 3. Memory Cost
$$M_{KV} = 2 \cdot L \cdot h \cdot d_h \cdot N \cdot 2 \text{ bytes (FP16)}$$
For Llama-70B at N=4096: ~10 GB. Memory is the new bottleneck.

## 4. Implementation Pattern
```python
if kv_cache is None:
    kv_cache = (K, V)
else:
    K_cache, V_cache = kv_cache
    K = torch.cat([K_cache, K], dim=2)
    V = torch.cat([V_cache, V], dim=2)
```

## 5. Scaling with Batch
Each sequence in batch has its own KV cache. M_KV_total = B x M_KV_per_seq. This is the primary scaling constraint for large-batch, long-context inference.
## 📝 Self-Assessment
<details class="mcq"><summary>KV cache purpose?</summary><ol type="A"><li>Store input</li><li>Cache K,V from previous steps -- avoid O(t^2) recomputation at each decode step</li><li>Store output</li><li>Speed only</li></ol><div class="answer">✅ Answer: B -- Without cache, each step recomputes K,V for all previous tokens. Cache stores them for reuse.</div></details>
<details class="mcq"><summary>Without cache, total decode FLOPs?</summary><ol type="A"><li>O(N^2)</li><li>O(N^3) -- each step t recomputes O(t^2). Sum to N = O(N^3). With cache: O(N^2)</li><li>O(N)</li><li>O(N log N)</li></ol><div class="answer">✅ Answer: B -- sum_{t=1}^{N} O(t^2) = O(N^3). KV cache eliminates recomputation, reducing to O(N^2).</div></details>
<details class="mcq"><summary>KV cache memory for 70B model, N=4096?</summary><ol type="A"><li>~1 GB</li><li>~10 GB -- 2 x 80 layers x 64 heads x 128 x 4096 x 2 bytes</li><li>~50 GB</li><li>~100 GB</li></ol><div class="answer">✅ Answer: B -- M_KV = 2*L*h*d_h*N*bytes. For long contexts, KV cache dwarfs model weights.</div></details>
<details class="mcq"><summary>KV cache update?</summary><ol type="A"><li>Replace</li><li>Append -- new token K,V concatenated to existing cache along sequence dimension</li><li>Overwrite</li><li>Reset</li></ol><div class="answer">✅ Answer: B -- K_new = cat([K_cache, K_new], dim=2). Cache grows with each generated token.</div></details>
<details class="mcq"><summary>KV cache and batch size?</summary><ol type="A"><li>Independent</li><li>Linear -- each sequence has its own cache. M_KV_total = B x M_KV_per_seq</li><li>Constant</li><li>Sublinear</li></ol><div class="answer">✅ Answer: B -- Doubling batch size doubles KV cache memory. Primary scaling constraint for large-batch inference.</div></details>
