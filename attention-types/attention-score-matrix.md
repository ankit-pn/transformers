<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Types</div>
# Attention Score Matrix

## 1. The Matrix
Before softmax, attention scores form an n x n matrix:

$$\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$$

Each element S_{ij} is the compatibility between query i and key j.

## 2. Structure
- **Diagonal**: Self-scores (Q_i · K_i). Usually high.
- **Off-diagonal**: Cross-position scores. Reveal token relationships.
- **Lower triangular**: Causal attention (decoder)
- **Full**: Bidirectional attention (encoder)

## 3. Properties
| Property | Encoder (Bidirectional) | Decoder (Causal) |
|---|---|---|
| Shape | n x n (full) | n x n (lower tri) |
| Diagonal | High self-attention | High self-attention |
| Information flow | All directions | Left to right only |

## 4. Visualization
Heatmaps of the attention score matrix reveal learned patterns: diagonal = self-focus, vertical lines = tokens many positions attend to, horizontal lines = positions attending broadly.

## 5. Memory
$$\text{Memory}_{\text{scores}} = n^2 \times 4 \text{ bytes (FP32)}$$
For n=2048: 16.8 MB. For n=8192: 268 MB. This O(n^2) memory is the primary bottleneck for long sequences.
## 📝 Self-Assessment
<details class="mcq"><summary>Attention score matrix shape?</summary><ol type="A"><li>n x d</li><li>n x n -- each of n queries scored against each of n keys</li><li>d x d</li><li>n x 1</li></ol><div class="answer">✅ Answer: B -- S = Q K^T / sqrt(d_k) produces an n x n matrix. Each element is a query-key compatibility score.</div></details>
<details class="mcq"><summary>Diagonal of score matrix?</summary><ol type="A"><li>Always zero</li><li>Self-scores -- Q_i · K_i. Usually highest values since token matches itself best</li><li>Random</li><li>Uniform</li></ol><div class="answer">✅ Answer: B -- A token naturally has high self-similarity. Diagonal typically dominates, especially in early layers.</div></details>
<details class="mcq"><summary>Encoder vs decoder score matrix?</summary><ol type="A"><li>Same</li><li>Encoder: full n x n. Decoder: lower triangular n x n (causal). Encoder sees all; decoder sees past only</li><li>Encoder smaller</li><li>Decoder full</li></ol><div class="answer">✅ Answer: B -- Bidirectional encoder has full attention. Autoregressive decoder uses causal mask = lower triangular.</div></details>
<details class="mcq"><summary>Memory for n=8192 score matrix (FP32)?</summary><ol type="A"><li>~64 MB</li><li>~268 MB -- 8192^2 x 4 bytes = 268,435,456 bytes</li><li>~16 MB</li><li>~1 GB</li></ol><div class="answer">✅ Answer: B -- O(n^2) memory. 8192^2 = 67M elements x 4 bytes = 268 MB. This quadratic growth limits context length.</div></details>
<details class="mcq"><summary>What do vertical lines in score heatmap mean?</summary><ol type="A"><li>Artifact</li><li>Positions that many queries attend to -- salient tokens that are important for many other positions</li><li>Error</li><li>Diagonal</li></ol><div class="answer">✅ Answer: B -- A column with consistently high scores means that specific token is important across many query positions.</div></details>
