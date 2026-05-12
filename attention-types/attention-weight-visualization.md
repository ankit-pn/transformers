<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Types</div>
# Attention Weight Visualization

## 1. Visualizing What the Model Sees
Attention weights (after softmax) reveal which tokens influence each output:

$$\mathbf{A} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{n \times n}$$

## 2. Common Patterns
| Pattern | Meaning |
|---|---|
| **Diagonal** | Self-attention dominance |
| **Vertical stripes** | Salient tokens many positions attend to |
| **Horizontal bands** | Positions attending broadly |
| **Block diagonal** | Local attention within phrases |
| **Head specialization** | Different heads show different patterns |

## 3. Interpretation Caveats
Attention is NOT explanation (Jain & Wallace, 2019). High attention weight does not always mean causal importance. Attention can be misleading -- weights may not correspond to feature importance.

## 4. Tools
```python
import matplotlib.pyplot as plt
# Extract attention from a specific layer and head
attn = outputs.attentions[layer][0, head]  # [n, n]
plt.imshow(attn.detach().cpu(), cmap='Blues')
```

## 5. Multi-Head Diversity
Different heads capture different relationships: some attend locally, some globally, some focus on syntax, others on semantics. This is the power of multi-head attention -- parallel, diverse views of the same sequence.
## 📝 Self-Assessment
<details class="mcq"><summary>What do attention weights visualize?</summary><ol type="A"><li>Loss</li><li>Which tokens influence each output position -- softmax-normalized relevance scores</li><li>Gradients</li><li>Parameters</li></ol><div class="answer">✅ Answer: B -- The attention matrix shows token-token dependencies. Each row shows where that position focuses.</div></details>
<details class="mcq"><summary>Are attention weights = explanation?</summary><ol type="A"><li>Yes, always</li><li>No -- high attention does not mean causal importance. Attention can be misleading (Jain & Wallace 2019)</li><li>Yes for BERT</li><li>Only for small models</li></ol><div class="answer">✅ Answer: B -- Attention weights correlate with but do not equal feature importance. Counter-examples exist where high-attention tokens are not important.</div></details>
<details class="mcq"><summary>Vertical stripe pattern meaning?</summary><ol type="A"><li>Artifact</li><li>Salient tokens that many positions attend to -- key words/phrases important across the sequence</li><li>Padding</li><li>Error</li></ol><div class="answer">✅ Answer: B -- A column with dark values = that token influences many output positions. Often punctuation or key content words.</div></details>
<details class="mcq"><summary>Block diagonal attention?</summary><ol type="A"><li>Random</li><li>Local attention within phrases -- nearby tokens attend to each other strongly</li><li>Global attention</li><li>Uniform</li></ol><div class="answer">✅ Answer: B -- Block diagonal = localized attention. Tokens attend strongly to neighbors, forming attention blocks along the diagonal.</div></details>
<details class="mcq"><summary>Multi-head attention diversity?</summary><ol type="A"><li>All same</li><li>Different heads learn different patterns: local, global, syntactic, semantic. Parallel diverse views of sequence</li><li>Random</li><li>Only one pattern</li></ol><div class="answer">✅ Answer: B -- Multi-head diversity is key to Transformer power. 8-64 parallel attention operations each capturing different relationships.</div></details>
