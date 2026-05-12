<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Optimization</div>
# Initialization Strategies

## 1. Xavier/Glorot Initialization
$$W \sim \mathcal{U}\left(-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right)$$

## 2. Kaiming/He Initialization
For ReLU activations:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

The factor of 2 accounts for ReLU zeroing half the activations.

## 3. Transformer-Specific
Original Transformer paper:
$$W \sim \mathcal{N}(0, 0.02) \text{ for most parameters}$$
Embeddings scaled by sqrt(d_model).

## 4. Why Good Initialization Matters
Poor initialization causes vanishing gradients, exploding gradients, slow convergence, and training instability in deep networks.

## 5. Residual Scaling
Some implementations scale residual branch output or initialize the last linear layer in each sublayer to zero for better early-training stability.
## 📝 Self-Assessment
<details class="mcq"><summary>Xavier initialization formula?</summary><ol type="A"><li>N(0,1)</li><li>Uniform[-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))] -- balances forward/backward variance</li><li>N(0, 2/n_in)</li><li>N(0, 0.01)</li></ol><div class="answer">✅ Answer: B -- Xavier keeps variance stable through the network for tanh-like activations.</div></details>
<details class="mcq"><summary>He vs Xavier?</summary><ol type="A"><li>Same</li><li>He: 2/n_in (ReLU halves activations). Xavier: 2/(n_in+n_out) (tanh preserves). He for ReLU</li><li>He for tanh</li><li>Xavier for ReLU</li></ol><div class="answer">✅ Answer: B -- He accounts for ReLU zeroing negative values, requiring 2x variance to compensate.</div></details>
<details class="mcq"><summary>Original Transformer initialization?</summary><ol type="A"><li>Xavier</li><li>N(0, 0.02) -- a simple small-variance normal. Embeddings scaled by sqrt(d_model)</li><li>He</li><li>Uniform</li></ol><div class="answer">✅ Answer: B -- The original paper used a straightforward small normal initialization.</div></details>
<details class="mcq"><summary>Poor initialization consequence?</summary><ol type="A"><li>Nothing</li><li>Vanishing/exploding gradients, slow convergence, training instability in deep networks</li><li>Slightly slower</li><li>Minor</li></ol><div class="answer">✅ Answer: B -- Wrong init scale cascades through deep networks. Devastating for deep Transformers.</div></details>
<details class="mcq"><summary>Zero-initializing last linear layer?</summary><ol type="A"><li>Never used</li><li>Forces each sublayer to initially act as identity. Improves early training stability</li><li>Always harmful</li><li>No benefit</li></ol><div class="answer">✅ Answer: B -- Zero-init of final projection makes residual path dominate initially, providing stable early training.</div></details>
