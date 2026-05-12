<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Transformer Blocks</div>
# Layer Normalization

## 1. Normalizing Across Features
$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where mu and sigma are computed across the feature dimension for EACH token independently:

$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$$

## 2. Per-Token Normalization
Each token is normalized independently -- no dependency on batch or sequence. This is crucial for variable-length sequences.

## 3. Learnable Parameters
Gamma (scale) and beta (shift) are learned per feature dimension. They restore representational capacity after normalization.

## 4. Why LayerNorm over BatchNorm?
| Aspect | BatchNorm | LayerNorm |
|---|---|---|
| Normalization axis | Batch | Features |
| Batch size dependency | Yes (bad for small batch) | No |
| Sequence length dependency | Yes | No |
| Training/inference gap | Yes (running stats) | No (same) |

## 5. RMSNorm (Modern Alternative)
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2}} \cdot \gamma$$

Simpler: no mean subtraction, no beta. Used in Llama, T5. Slightly faster, similar quality.
## 📝 Self-Assessment
<details class="mcq"><summary>What does LayerNorm normalize across?</summary><ol type="A"><li>Batch</li><li>Feature dimension -- each token is normalized independently. Mu and sigma computed over d_model</li><li>Sequence</li><li>All dimensions</li></ol><div class="answer">✅ Answer: B -- LN normalizes across features for each token. No dependency on batch size or sequence length.</div></details>
<details class="mcq"><summary>Why LayerNorm over BatchNorm for Transformers?</summary><ol type="A"><li>BatchNorm faster</li><li>No batch/sequence dependency. Works identically for any batch size and sequence length</li><li>Better accuracy</li><li>Historical</li></ol><div class="answer">✅ Answer: B -- BatchNorm needs batch statistics; bad for variable-length sequences and small batches. LN is independent.</div></details>
<details class="mcq"><summary>LayerNorm learnable parameters?</summary><ol type="A"><li>None</li><li>Gamma (scale) and beta (shift) per feature. Learned during training; restore capacity after normalization</li><li>Mu and sigma</li><li>Just gamma</li></ol><div class="answer">✅ Answer: B -- gamma * normalized + beta. gamma=1, beta=0 at init. Learned to adjust the normalized distribution.</div></details>
<details class="mcq"><summary>What is RMSNorm?</summary><ol type="A"><li>Same as LayerNorm</li><li>Simpler: no mean centering, only scaling by RMS. Faster computation, similar quality (used in Llama)</li><li>BatchNorm variant</li><li>Post-LN</li></ol><div class="answer">✅ Answer: B -- RMSNorm(x) = x / RMS(x) * gamma. No mean subtraction, no beta. Simpler, faster, equally effective.</div></details>
<details class="mcq"><summary>LayerNorm computation per token?</summary><ol type="A"><li>Batch-dependent</li><li>Independent -- mu and sigma are computed from the token own features. Same at train and inference</li><li>Sequence-dependent</li><li>Global</li></ol><div class="answer">✅ Answer: B -- Each token has its own mu,sigma. No running statistics. Same computation at training and inference time.</div></details>
