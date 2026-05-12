<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Training</div>
# Computational Complexity of Self-Attention

## 1. FLOPs Breakdown
$$\text{FLOPs}_{\text{attention}} = 4nd^2 + 2n^2 d$$
- Projections (Q,K,V,O): 4nd^2
- Attention scores (QK^T): 2n^2 d
- Weighted sum (AV): 2n^2 d

## 2. When Each Term Dominates
- Short sequences (n << d): Projections dominate (4nd^2)
- Long sequences (n >> d): Attention dominates (4n^2 d)
- Cross-over: n = 2d

## 3. Practical Examples
| n | d=1024 | Projections | Attention | Ratio |
|---|---|---|---|---|
| 128 | 1024 | 536M | 34M | 16:1 |
| 1024 | 1024 | 4.3B | 2.1B | 2:1 |
| 4096 | 1024 | 17B | 34B | 1:2 |
| 8192 | 1024 | 34B | 138B | 1:4 |

## 4. FLOPs vs Memory
Compute (FLOPs) tells half the story. Memory bandwidth often dominates for inference (especially decode).
## 📝 Self-Assessment
<details class="mcq"><summary>Self-attention FLOPs formula?</summary><ol type="A"><li>O(n d)</li><li>4nd^2 + 4n^2 d -- projections (O(nd^2)) + attention computation (O(n^2 d))</li><li>O(n^2 d^2)</li><li>O(n d^2)</li></ol><div class="answer">✅ Answer: B -- Two main terms: projections scale with nd^2, attention scores and aggregation scale with n^2 d.</div></details>
<details class="mcq"><summary>When does attention term dominate?</summary><ol type="A"><li>Always</li><li>When n >> 2d -- the n^2 term overtakes the nd^2 term. For d=1024, cross-over at n=2048</li><li>Never</li><li>Only for small n</li></ol><div class="answer">✅ Answer: B -- 4n^2d > 4nd^2 when n > d. For typical d=64 per head, attention dominates at n > 64.</div></details>
<details class="mcq"><summary>FLOPs for n=4096, d=1024?</summary><ol type="A"><li>~10B</li><li>~51B -- projections: 17B, attention: 34B. Attention dominates at this length</li><li>~100B</li><li>~1B</li></ol><div class="answer">✅ Answer: B -- 4*4096*1024^2 = 17B for projections. 4*4096^2*1024 = 68B for attention. Total ~85B for full MHA.</div></details>
<details class="mcq"><summary>FLOPs vs memory bandwidth bottleneck?</summary><ol type="A"><li>Same thing</li><li>FLOPs = compute cost. Memory bandwidth = data movement cost. For inference, bandwidth often the real bottleneck</li><li>FLOPs always</li><li>Memory always</li></ol><div class="answer">✅ Answer: B -- A GPU may have FLOPs to spare but be stalled waiting for data from HBM. Bandwidth-bound, not compute-bound.</div></details>
<details class="mcq"><summary>Per-head attention FLOPs with n=2048, d_k=64?</summary><ol type="A"><li>~1B</li><li>~2.1B -- 4*2048*64^2=33M projections + 4*2048^2*64=1.07B attention = ~1.1B per head</li><li>~10B</li><li>~500M</li></ol><div class="answer">✅ Answer: B -- Per-head cost is modest. Multi-head multiplies this by h. For h=16: ~17.6B total attention FLOPs.</div></details>
