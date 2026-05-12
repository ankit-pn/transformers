<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Optimization</div>
# Gradient Flow in Transformers

## 1. Why Transformers Train Well
Residual connections + LayerNorm = excellent gradient flow:

$$\frac{\partial L}{\partial X_l} = \frac{\partial L}{\partial Y_L} \prod_{k=l+1}^{L} \left(I + \frac{\partial \text{Sublayer}_k}{\partial X_{k-1}}\right)$$

## 2. The Identity Highway
The I term ensures gradients never fully vanish. Even if all sublayer gradients go to zero, the identity carries the training signal to the earliest layers.

## 3. Gradient Scale
LayerNorm maintains activations at consistent scale throughout the network, preventing gradient explosion or vanishing.

## 4. Attention Gradient Flow
Attention creates direct paths between any positions:

$$\frac{\partial L}{\partial X_i} = \sum_j \alpha_{ji} \cdot \frac{\partial L}{\partial Y_j}$$

Each output contributes gradient to each input, weighted by attention.

## 5. Deep Transformers
Up to ~100 layers trainable with Pre-LN, warmup, and proper initialization. Beyond that, stability becomes challenging.
## 📝 Self-Assessment
<details class="mcq"><summary>Transformer gradient flow advantage?</summary><ol type="A"><li>No advantage</li><li>Residuals + LayerNorm create excellent gradient highways -- the I term prevents vanishing in deep networks</li><li>Same as RNNs</li><li>Worse than RNNs</li></ol><div class="answer">✅ Answer: B -- dL/dX_l includes I term from residual. Even if sublayer gradients vanish, identity carries signal.</div></details>
<details class="mcq"><summary>LayerNorm effect on gradient scale?</summary><ol type="A"><li>None</li><li>Maintains consistent activation variance across layers, preventing gradient explosion/vanishing</li><li>Increases variance</li><li>Decreases variance</li></ol><div class="answer">✅ Answer: B -- LN normalizes each layer output to similar statistics. Gradients flow through consistent-scale activations.</div></details>
<details class="mcq"><summary>Attention gradient connectivity?</summary><ol type="A"><li>Sparse</li><li>Dense -- each output contributes gradient to ALL inputs, weighted by attention</li><li>Local only</li><li>Diagonal</li></ol><div class="answer">✅ Answer: B -- dL/dX_i = sum(alpha_{ji} * dL/dY_j). Rich information flow between all positions.</div></details>
<details class="mcq"><summary>Maximum trainable Transformer depth?</summary><ol type="A"><li>10</li><li>~100 layers with Pre-LN, warmup, and proper initialization</li><li>Unlimited</li><li>50</li></ol><div class="answer">✅ Answer: B -- GPT-3 uses 96 layers. Deep Transformers are trainable with careful architectural choices.</div></details>
<details class="mcq"><summary>Why does Pre-LN improve gradient flow?</summary><ol type="A"><li>No effect</li><li>Normalizing INPUT to sublayers keeps activations in controlled range regardless of depth</li><li>Post-LN better</li><li>Same</li></ol><div class="answer">✅ Answer: B -- X + Sublayer(LN(X)). Input normalized before sublayer, preventing activation drift across depth.</div></details>
