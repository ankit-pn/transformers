<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Transformer Blocks</div>
# Residual Connections

## 1. Skip Connections
$$\text{Output} = \text{LayerNorm}(X + \text{Sublayer}(X))$$

The input X is added to the sublayer output before LayerNorm. This creates an identity shortcut that helps gradients flow.

## 2. Why Residuals?
Without residuals, deep networks suffer from vanishing gradients and degradation. Residuals provide a "gradient highway" -- the gradient can flow through the identity path unimpeded:

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot \left(I + \frac{\partial \text{Sublayer}}{\partial X}\right)$$

The identity term I prevents the gradient from vanishing even when d(Sublayer)/dX is small.

## 3. Pre-LN vs Post-LN
| Variant | Order | Stability |
|---|---|---|
| Post-LN (original) | X + Sublayer(LN(X)) | Needs warmup |
| Pre-LN (modern) | X + Sublayer(LN(X))... no wait | |

Actually the standard Transformer uses: LN(X + Sublayer(X)) for each block. Modern variants (Pre-LN) use X + Sublayer(LN(X)) which is more stable.

## 4. Where Used
Every sublayer in the Transformer has a residual connection: Self-attention sublayer AND feedforward sublayer. Each block: residual -> LayerNorm.
## 📝 Self-Assessment
<details class="mcq"><summary>Transformer residual connection formula?</summary><ol type="A"><li>X + Sublayer(X)</li><li>LN(X + Sublayer(X)) -- add input to output, then normalize. Two residuals per block</li><li>Sublayer(X) + LN</li><li>X * Sublayer(X)</li></ol><div class="answer">✅ Answer: B -- Each sublayer (attention, FFN) has a residual: output = LN(input + Sublayer(input)).</div></details>
<details class="mcq"><summary>Why are residuals critical?</summary><ol type="A"><li>Speed</li><li>Gradient highway -- the identity term in dY/dX = I + dSublayer/dX prevents vanishing gradients in deep networks</li><li>Accuracy</li><li>Memory</li></ol><div class="answer">✅ Answer: B -- Even when the sublayer gradient is small, the identity term ensures gradients flow backward. Enables training 100+ layer Transformers.</div></details>
<details class="mcq"><summary>How many residuals per transformer block?</summary><ol type="A"><li>1</li><li>2 -- one for self-attention sublayer, one for feedforward sublayer</li><li>3</li><li>4</li></ol><div class="answer">✅ Answer: B -- Each block has two sublayers (attention + FFN), each with its own residual connection.</div></details>
<details class="mcq"><summary>Pre-LN vs Post-LN stability?</summary><ol type="A"><li>Same</li><li>Pre-LN (X + Sublayer(LN(X))) is more stable without learning rate warmup. Modern default</li><li>Post-LN better</li><li>No difference</li></ol><div class="answer">✅ Answer: B -- Pre-LN normalizes the input, making gradient flow more stable. Most modern implementations use Pre-LN.</div></details>
<details class="mcq"><summary>Residual connection gradient?</summary><ol type="A"><li>Only Sublayer</li><li>I + dSublayer/dX -- the identity matrix guarantees gradient does not vanish even in deep layers</li><li>Zero</li><li>Negative</li></ol><div class="answer">✅ Answer: B -- d(LN(X+Sublayer))/dX includes identity term I. Even if Sublayer gradients vanish, identity carries signal.</div></details>
