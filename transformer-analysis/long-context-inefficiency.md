<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Analysis</div>
# Long-Context Inefficiency

## 1. The Quadratic Curse
$$\text{Cost}(n) = c \cdot n^2$$
Going from 4K to 128K context: 1024x more compute and memory.

## 2. Practical Limits
| Context Length | Attention Memory (d=64, FP16) |
|---|---|
| 4K | 135 MB |
| 8K | 540 MB |
| 16K | 2.1 GB |
| 32K | 8.6 GB |
| 128K | 137 GB |
| 1M | 8.6 TB |

## 3. Memory Bandwidth Bottleneck
At long context, attention becomes memory-bandwidth-bound. The GPU spends most time reading/writing the attention matrix from HBM, not computing.

## 4. KV Cache Explosion
Decoder KV cache also scales with n. For Llama-70B at 128K: ~320 GB of KV cache alone. This is often the harder constraint than attention memory.

## 5. Solutions
| Approach | Context | Tradeoff |
|---|---|---|
| FlashAttention | Up to 64K | Exact attention, better memory |
| Sliding window | Unlimited | Loses long-range |
| Sparse attention | 2-4x baseline | Pattern-dependent |
| Ring attention | Scales with GPUs | Communication overhead |
| State space models | Unlimited | Different architecture |
## 📝 Self-Assessment
<details class="mcq"><summary>Cost multiplier: 4K to 128K context?</summary><ol type="A"><li>32x</li><li>1024x -- (128K/4K)^2 = 32^2 = 1024. Quadratic scaling is brutal</li><li>128x</li><li>4096x</li></ol><div class="answer">✅ Answer: B -- O(n^2). Going from 4K to 128K = 32x longer = 1024x more compute and memory.</div></details>
<details class="mcq"><summary>Attention memory for n=128K, d=64, FP16?</summary><ol type="A"><li>~1 GB</li><li>~137 GB -- 128K^2 x 64 x 2 bytes. Far beyond any single GPU VRAM</li><li>~10 GB</li><li>~500 GB</li></ol><div class="answer">✅ Answer: B -- 128K^2 = 16B elements x 64 heads x 2 bytes = ~2 TB. Per-head: ~137 GB. Requires distributed or sparse attention.</div></details>
<details class="mcq"><summary>KV cache for 70B at n=128K?</summary><ol type="A"><li>~10 GB</li><li>~320 GB -- 2 x 80 x 64 x 128 x 128K x 2 bytes. KV cache dwarfs model weights</li><li>~80 GB</li><li>~1 TB</li></ol><div class="answer">✅ Answer: B -- KV cache = 2*L*h*d_h*n*bytes. At 128K, this is a massive memory demand.</div></details>
<details class="mcq"><summary>FlashAttention context limit?</summary><ol type="A"><li>4K</li><li>Up to 64K+ on A100/H100 -- same O(n^2) compute but O(n) memory via tiling</li><li>Unlimited</li><li>2K</li></ol><div class="answer">✅ Answer: B -- FlashAttention computes exact attention with O(n) memory. Pushes context to 64K+ on modern GPUs.</div></details>
<details class="mcq"><summary>Ring attention solution?</summary><ol type="A"><li>Smaller model</li><li>Distributes attention across multiple GPUs -- each GPU holds a slice of the sequence. Context scales with GPU count</li><li>Faster GPU</li><li>Compression</li></ol><div class="answer">✅ Answer: B -- Ring attention splits the sequence across GPUs, communicating K,V blocks in a ring. Context = per_GPU_memory x num_GPUs.</div></details>
