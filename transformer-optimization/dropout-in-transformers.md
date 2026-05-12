<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Optimization</div>
# Dropout in Transformers

## 1. Regularization via Random Masking
During training, randomly zero out elements with probability p:

$$h_i' = h_i \cdot \frac{m_i}{1-p}, \quad m_i \sim \text{Bernoulli}(1-p)$$

Scaling by 1/(1-p) keeps the expected value unchanged.

## 2. Where Dropout Is Applied
| Location | Typical p |
|---|---|
| Attention weights | 0.1 |
| FFN hidden activations | 0.1 |
| Embedding dropout | 0.1 |
| Output logits | 0.1 |

## 3. Why Dropout in Transformers?
Transformers have massive capacity and can easily overfit, especially on smaller datasets. Dropout provides cheap, effective regularization. Each training forward pass uses a different random subset of the model.

## 4. Dropout at Inference
Dropout is DISABLED at inference. All weights are active. This is why the 1/(1-p) scaling is needed during training -- it calibrates the expected output to match inference behavior.

## 5. Modern Trends
Larger models trained on massive data use less dropout (or none). GPT-3 uses no dropout at 175B scale -- the data itself provides sufficient regularization.
## 📝 Self-Assessment
<details class="mcq"><summary>Dropout scaling factor?</summary><ol type="A"><li>1</li><li>1/(1-p) -- keeps expected output unchanged during training to match inference</li><li>p</li><li>1-p</li></ol><div class="answer">✅ Answer: B -- E[h * m / (1-p)] = h * (1-p) / (1-p) = h. Expected value preserved.</div></details>
<details class="mcq"><summary>Dropout at inference?</summary><ol type="A"><li>Enabled</li><li>DISABLED -- all connections active. Training scaling ensures output statistics match</li><li>Reduced</li><li>Random</li></ol><div class="answer">✅ Answer: B -- Inference uses the full model. Training dropout+scaling calibrates expected output to match the full model.</div></details>
<details class="mcq"><summary>Typical Transformer dropout rate?</summary><ol type="A"><li>0.5</li><li>0.1 (10%). Applied to attention weights, FFN activations, and embeddings</li><li>0.01</li><li>0.9</li></ol><div class="answer">✅ Answer: B -- p=0.1 is standard across attention, FFN, and embedding dropout in the original Transformer paper.</div></details>
<details class="mcq"><summary>Why less dropout in large models?</summary><ol type="A"><li>Slower training</li><li>Massive training data provides natural regularization -- overfitting is not a concern at scale</li><li>More params</li><li>Required</li></ol><div class="answer">✅ Answer: B -- GPT-3 (175B) uses zero dropout. The enormous training corpus provides implicit regularization.</div></details>
<details class="mcq"><summary>Dropout as ensemble?</summary><ol type="A"><li>No</li><li>Yes -- each dropout mask creates a different sub-network. Training approximates ensemble of exponentially many networks</li><li>Single model</li><li>Regularization only</li></ol><div class="answer">✅ Answer: B -- Each forward pass samples different sub-network via random masking. Ensemble effect key to dropout power.</div></details>
