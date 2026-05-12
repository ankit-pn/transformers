<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Analysis</div>
# Limitations of Vanilla Attention

## 1. Quadratic Complexity
$$\text{FLOPs} = O(n^2 d), \quad \text{Memory} = O(n^2)$$
Doubling sequence length quadruples compute and memory. This is THE fundamental limitation.

## 2. No Built-in Position Information
Attention is permutation-invariant. Positional encoding is bolted on but does not provide a strong inductive bias for sequential structure.

## 3. Fixed Context Window
Each token attends to ALL other tokens within the window. Cannot selectively attend to far-away tokens more than nearby ones without explicit position bias.

## 4. Over-parameterization
Attention maps are sparse in practice but dense attention computes ALL n^2 pairs. Most compute is wasted on near-zero attention weights. Sparse and linear attention methods address this.

## 5. Single Representation
Each token has one attention pattern per head. Cannot dynamically adjust behavior based on content. Mixture of Experts and dynamic routing address this.
## 📝 Self-Assessment
<details class="mcq"><summary>Vanilla attention primary limitation?</summary><ol type="A"><li>Too deep</li><li>Quadratic complexity O(n^2) in both compute and memory -- the fundamental scaling bottleneck</li><li>Too slow to train</li><li>Too shallow</li></ol><div class="answer">✅ Answer: B -- Every token attends to every other token. O(n^2) scaling limits practical sequence length.</div></details>
<details class="mcq"><summary>Attention and position information?</summary><ol type="A"><li>Built-in</li><li>Attention is permutation-invariant -- position encoding is added separately, not inherent to the mechanism</li><li>Position aware</li><li>Sequence-based</li></ol><div class="answer">✅ Answer: B -- Shuffling input gives same attention outputs (reordered). PE is a workaround, not a solution.</div></details>
<details class="mcq"><summary>Attention map density in practice?</summary><ol type="A"><li>Dense</li><li>Sparse -- most attention weights are near-zero. Dense computation wastes resources on negligible pairs</li><li>Uniform</li><li>Diagonal only</li></ol><div class="answer">✅ Answer: B -- 80%+ of attention weights are near-zero. The dense O(n^2) computation is wasteful given the inherent sparsity.</div></details>
<details class="mcq"><summary>Fixed context window issue?</summary><ol type="A"><li>No issue</li><li>Every token gets equal access to all positions -- cannot naturally prioritize nearby over distant tokens</li><li>Distant favored</li><li>Nearby ignored</li></ol><div class="answer">✅ Answer: B -- All positions are equally accessible. This is powerful but unnatural for language where locality matters.</div></details>
<details class="mcq"><summary>Single representation per head?</summary><ol type="A"><li>Head adapts</li><li>Each head produces one attention pattern regardless of input content. Cannot dynamically specialize per-token</li><li>Multiple rep</li><li>Dynamic</li></ol><div class="answer">✅ Answer: B -- Attention pattern is determined by QK similarity. No mechanism to dynamically change behavior based on token importance.</div></details>
