<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Training</div>
# Training Objective

## 1. Cross-Entropy Loss
$$\mathcal{L} = -\frac{1}{N}\sum_{n=1}^{N} \sum_{t=1}^{T_n} \log P(y_{n,t}^* | y_{n,<t}^*, x_n)$$

## 2. Perplexity
Exponentiated cross-entropy:

$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t} \log P(y_t^*)\right)$$

Interpretation: the model is as confused as if it were choosing uniformly among PPL equally likely options.

## 3. Optimization
Adam optimizer with learning rate schedule:
- Warmup: linear increase for first N steps
- Decay: inverse square root or cosine decay
- Gradient clipping: prevent exploding gradients

## 4. Label Smoothing
Replace one-hot targets with smoothed distribution:
$$P_{smooth}(y) = (1 - \epsilon) \cdot \mathbb{1}[y = y^*] + \frac{\epsilon}{|V|}$$

Prevents overconfidence and improves generalization.
## 📝 Self-Assessment
<details class="mcq"><summary>Transformer training loss?</summary><ol type="A"><li>MSE</li><li>Cross-entropy: -sum(log P(correct token | context)). Standard for classification</li><li>L1</li><li>Cosine</li></ol><div class="answer">✅ Answer: B -- CE loss maximizes probability of correct token. Equivalent to minimizing KL divergence between predicted and true distribution.</div></details>
<details class="mcq"><summary>Perplexity of 10 means?</summary><ol type="A"><li>10% accuracy</li><li>Model is as uncertain as choosing uniformly from 10 equally likely options. Lower = better</li><li>10 tokens</li><li>10 layers</li></ol><div class="answer">✅ Answer: B -- PPL = exp(cross-entropy). PPL=10 means the model average uncertainty is equivalent to a 1/10 chance per token.</div></details>
<details class="mcq"><summary>Adam warmup purpose?</summary><ol type="A"><li>Not needed</li><li>Prevents early training instability -- gradients can be huge early. Warmup gradually increases LR</li><li>Speed</li><li>Memory</li></ol><div class="answer">✅ Answer: B -- Transformer gradients are large early in training. Warmup prevents destructive parameter updates before adaptive moments stabilize.</div></details>
<details class="mcq"><summary>Label smoothing effect?</summary><ol type="A"><li>No effect</li><li>Prevents overconfidence -- softens target distribution. Model learns to be less certain about predictions</li><li>Worse accuracy</li><li>Faster</li></ol><div class="answer">✅ Answer: B -- Instead of P(target)=1.0, gives P(target)=0.9, P(others)=0.1/|V|. Improves calibration and generalization.</div></details>
<details class="mcq"><summary>Gradient clipping in Transformers?</summary><ol type="A"><li>Not needed</li><li>Prevents gradient explosion -- clips grad norm to max value (typically 1.0). Essential for training stability</li><li>Rarely used</li><li>Only inference</li></ol><div class="answer">✅ Answer: B -- Large gradients cause training collapse. Clipping bounds the update magnitude, preventing destructive parameter changes.</div></details>
