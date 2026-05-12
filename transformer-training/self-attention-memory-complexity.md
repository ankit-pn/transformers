<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Training</div>
# Memory Complexity of Self-Attention

## 1. Memory Breakdown
$$M_{\text{attention}} = M_{QKV} + M_{scores} + M_{weights} + M_{output}$$

## 2. The Score Matrix
The n x n attention score matrix is the primary memory consumer:

$$M_{scores} = n^2 \times \text{bytes\_per\_elem}$$

| n | FP32 (4B) | FP16 (2B) |
|---|---|---|
| 1024 | 4 MB | 2 MB |
| 4096 | 67 MB | 34 MB |
| 8192 | 268 MB | 134 MB |
| 32768 | 4.3 GB | 2.1 GB |
| 131072 | 68.7 GB | 34.4 GB |

## 3. Memory vs Compute
For long sequences, memory becomes the bottleneck before compute. At n=32K, the score matrix alone is 4.3 GB -- exceeding many GPU memories. FlashAttention solves this by never materializing the full matrix.

## 4. Training Memory
Training requires storing activations for backprop: score matrix + Q,K,V + attention weights. Inference is lighter (no backprop), but KV cache grows with sequence length.
## 📝 Self-Assessment
<details class="mcq"><summary>Memory bottleneck in self-attention?</summary><ol type="A"><li>Q,K,V tensors</li><li>The n x n attention score matrix -- grows quadratically with sequence length</li><li>Output tensor</li><li>Weights</li></ol><div class="answer">✅ Answer: B -- n^2 values rapidly exceeds GPU memory. At n=32K: 4.3 GB FP32 for just the score matrix.</div></details>
<details class="mcq"><summary>Score matrix memory for n=8192 FP16?</summary><ol type="A"><li>~16 MB</li><li>~134 MB -- 8192^2 x 2 bytes = 134,217,728 bytes</li><li>~1 GB</li><li>~500 MB</li></ol><div class="answer">✅ Answer: B -- O(n^2). 67M elements x 2 bytes = 134 MB. This is why long-context inference is memory-constrained.</div></details>
<details class="mcq"><summary>How does FlashAttention reduce memory?</summary><ol type="A"><li>Compression</li><li>Never materializes the full n^2 matrix -- computes attention in tiles with O(n) memory</li><li>Lower precision</li><li>Smaller model</li></ol><div class="answer">✅ Answer: B -- FlashAttention tiling: O(n^2) compute but O(n) memory. The score matrix is never stored; it is consumed immediately.</div></details>
<details class="mcq"><summary>Training vs inference memory?</summary><ol type="A"><li>Same</li><li>Training: higher (activations for backprop). Inference: lower (no backprop), but KV cache grows with n</li><li>Inference higher</li><li>Always equal</li></ol><div class="answer">✅ Answer: B -- Training stores intermediate activations for backward pass. Inference discards them but accumulates KV cache.</div></details>
<details class="mcq"><summary>Maximum practical sequence length without FlashAttention?</summary><ol type="A"><li>Unlimited</li><li>~2-4K for typical GPU (16-24 GB). Beyond this, the n^2 score matrix exceeds VRAM</li><li>100K</li><li>1M</li></ol><div class="answer">✅ Answer: B -- Without memory-efficient attention, O(n^2) memory caps sequences at a few thousand tokens on most GPUs.</div></details>
