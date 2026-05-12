<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Transformer Blocks</div>
# Encoder Stack

## 1. Stacked Encoder Blocks
$$H^{(l)} = \text{EncoderBlock}_l(H^{(l-1)}), \quad l = 1, ..., L$$

Input H^(0) = Token Embeddings + Positional Encoding.

## 2. Depth
| Model | Encoder Layers |
|---|---|
| Transformer Base | 6 |
| Transformer Big | 6 |
| BERT-base | 12 |
| BERT-large | 24 |
| T5 | 12-24 |

## 3. Information Flow
Each layer refines representations. Lower layers capture local syntax; higher layers capture semantics. Residual connections enable training deep stacks.

## 4. Parameter Distribution
$$\text{Params}_{\text{encoder}} = L \times (\text{Params}_{\text{MHA}} + \text{Params}_{\text{FFN}})$$

For BERT-base: 12 layers x ~7M per layer = ~85M (plus embedding params).

## 5. Output
Final encoder output: H^(L) of shape [batch, n, d_model]. Each token has been contextualized by the full sequence through L layers of bidirectional attention.
## 📝 Self-Assessment
<details class="mcq"><summary>Encoder stack depth (Transformer Base)?</summary><ol type="A"><li>12</li><li>6 layers. Each layer has self-attention + FFN</li><li>24</li><li>1</li></ol><div class="answer">✅ Answer: B -- Original Transformer Base: 6 encoder layers. BERT scaled this to 12-24 layers.</div></details>
<details class="mcq"><summary>Information flow across encoder layers?</summary><ol type="A"><li>Same representation</li><li>Progressive refinement -- lower layers capture local syntax; higher layers capture global semantics</li><li>No change</li><li>Random</li></ol><div class="answer">✅ Answer: B -- Each layer builds on previous. Shallow = syntax. Deep = semantics. Residuals make deep stacks trainable.</div></details>
<details class="mcq"><summary>Encoder output for each token?</summary><ol type="A"><li>Single vector</li><li>Contextualized representation -- enriched by all other tokens through bidirectional attention</li><li>Class label</li><li>Decoded text</li></ol><div class="answer">✅ Answer: B -- Each token output is informed by the entire input sequence through L layers of full self-attention.</div></details>
<details class="mcq"><summary>Parameter count per encoder layer (d=768)?</summary><ol type="A"><li>~1M</li><li>~7M -- MHA: 4x768^2=~2.4M, FFN: 2x768x3072=~4.7M. Total ~7M per layer</li><li>~14M</li><li>~3M</li></ol><div class="answer">✅ Answer: B -- FFN dominates: 8*d^2 for FFN vs 4*d^2 for attention. Per-layer params are mostly in FFN.</div></details>
<details class="mcq"><summary>Why stack multiple encoder layers?</summary><ol type="A"><li>Only need one</li><li>Progressive abstraction -- each layer captures increasingly complex relationships across tokens</li><li>For speed</li><li>For parameters</li></ol><div class="answer">✅ Answer: B -- Multiple layers build hierarchical representations. One layer cannot capture both local and global patterns.</div></details>
