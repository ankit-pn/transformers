<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Training</div>
# Quadratic Scaling Issue (O(n^2))

## 1. The Core Problem
Both compute AND memory of self-attention scale as O(n^2):

$$\text{FLOPs} = 4n^2 d, \quad \text{Memory} = n^2$$

## 2. Why It Matters
| n | Relative Cost | Feasibility |
|---|---|---|
| 512 | 1x | Trivial |
| 2048 | 16x | Easy |
| 8192 | 256x | Challenging |
| 32768 | 4096x | Hard (needs FlashAttention) |
| 131072 | 65536x | Extreme (needs sparse/linear attention) |

## 3. Solutions Landscape
| Approach | Complexity | Quality |
|---|---|---|
| FlashAttention | O(n^2) compute, O(n) memory | Exact |
| Sparse Attention | O(n * k) | Near-exact |
| Linear Attention | O(n * d^2) | Approximate |
| Sliding Window | O(n * W) | Local context |
| State Space Models | O(n * d^2) | Competitive |

## 4. The Reality
For most production models, context is limited to 4K-32K tokens. Longer contexts require architectural innovations or significant compute investment. The quadratic barrier is THE defining challenge of Transformer scaling.
## 📝 Self-Assessment
<details class="mcq"><summary>Self-attention complexity?</summary><ol type="A"><li>O(n log n)</li><li>O(n^2) in both FLOPs and memory. This is the fundamental Transformer bottleneck</li><li>O(n)</li><li>O(d^2)</li></ol><div class="answer">✅ Answer: B -- Attention computes n x n scores. Both compute and memory scale quadratically with sequence length.</div></details>
<details class="mcq"><summary>Cost multiplier from n=512 to n=8192?</summary><ol type="A"><li>16x</li><li>256x -- (8192/512)^2 = 256x increase in both compute and memory</li><li>64x</li><li>1024x</li></ol><div class="answer">✅ Answer: B -- O(n^2) scaling. 16x longer sequence = 256x more compute and memory. The curse of quadratic scaling.</div></details>
<details class="mcq"><summary>FlashAttention complexity?</summary><ol type="A"><li>O(n) compute</li><li>O(n^2) compute (exact attention) but O(n) memory (no n^2 matrix stored). I/O-aware algorithm</li><li>O(n log n)</li><li>O(n) both</li></ol><div class="answer">✅ Answer: B -- Same FLOPs as standard attention but O(n) memory via tiling. Computes exact attention without storing score matrix.</div></details>
<details class="mcq"><summary>Linear attention complexity?</summary><ol type="A"><li>O(n^2)</li><li>O(n d^2) -- approximates attention by changing computation order. Much faster for long sequences but approximate</li><li>O(n d)</li><li>O(n log n)</li></ol><div class="answer">✅ Answer: B -- Linear attention rearranges (QK^T)V to Q(K^T V). O(n d^2) instead of O(n^2 d). Approximates standard attention.</div></details>
<details class="mcq"><summary>Is O(n^2) THE defining Transformer challenge?</summary><ol type="A"><li>No</li><li>Yes -- it limits context length, drives memory cost, and motivates most efficient attention research</li><li>Minor</li><li>Solved</li></ol><div class="answer">✅ Answer: B -- The quadratic barrier is the primary constraint on Transformer scaling to longer contexts. Most research focuses on overcoming it.</div></details>
