<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Training</div>
# Attention Map Sparsity Patterns

## 1. Not All Positions Are Equal
Attention maps are inherently sparse -- most tokens attend strongly to only a few positions:

$$\text{Sparsity} = \frac{\text{number of near-zero weights}}{n^2}$$

## 2. Common Patterns
| Pattern | Description | % of Attention |
|---|---|---|
| Diagonal | Self-attention | 10-20% |
| Local band | Nearby tokens | 30-50% |
| Vertical stripes | Salient tokens (punctuation, CLS) | 10-20% |
| Sparse long-range | Few distant connections | 5-10% |

## 3. Why Sparse?
Language has LOCAL structure (most words relate to nearby words) and selective LONG-RANGE structure (pronouns, syntactic dependencies). Full dense attention is wasteful -- most pairs are irrelevant.

## 4. Exploiting Sparsity
Sparse attention patterns (Longformer, BigBird) reduce O(n^2) to O(n * k) where k << n:

$$\text{Attention}_{sparse}(i) = \sum_{j \in S(i)} \alpha_{ij} v_j$$

Where S(i) is a sparse subset of all positions (typically 128-512 tokens).
## 📝 Self-Assessment
<details class="mcq"><summary>Are attention maps dense?</summary><ol type="A"><li>Yes</li><li>No -- inherently sparse. Most tokens attend to a small fraction of positions. 80%+ weights near-zero</li><li>Always dense</li><li>Depends</li></ol><div class="answer">✅ Answer: B -- Attention maps show strong diagonal/local patterns. Most token pairs have near-zero attention after softmax.</div></details>
<details class="mcq"><summary>Most common attention pattern?</summary><ol type="A"><li>Uniform</li><li>Local band -- nearby tokens dominate attention. Language has strong locality</li><li>Global</li><li>Random</li></ol><div class="answer">✅ Answer: B -- 30-50% of attention mass goes to tokens within 10-20 positions. Language is locally structured.</div></details>
<details class="mcq"><summary>What are vertical stripes in attention maps?</summary><ol type="A"><li>Artifact</li><li>Salient tokens many positions attend to -- often punctuation, [CLS], or key content words</li><li>Error</li><li>Padding</li></ol><div class="answer">✅ Answer: B -- When many rows show high values in the same column, that token is important across the sequence.</div></details>
<details class="mcq"><summary>Sparse attention complexity?</summary><ol type="A"><li>O(n^2)</li><li>O(n * k) where k is the number of attended positions (typically 128-512). k << n</li><li>O(n)</li><li>O(n log n)</li></ol><div class="answer">✅ Answer: B -- Instead of attending to all n positions, attend to k selected positions. Complexity drops from O(n^2) to O(nk).</div></details>
<details class="mcq"><summary>Why does language attention exhibit locality?</summary><ol type="A"><li>Random</li><li>Adjacent words form phrases and have strong syntactic/semantic relationships -- most dependencies are local</li><li>Model architecture</li><li>Training</li></ol><div class="answer">✅ Answer: B -- Natural language has local structure: adjectives modify nearby nouns, verbs relate to nearby subjects.</div></details>
