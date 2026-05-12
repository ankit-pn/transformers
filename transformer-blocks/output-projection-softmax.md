<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Transformer Blocks</div>
# Output Projection + Softmax Layer

## 1. Final Step
The decoder output passes through a linear projection and softmax to produce token probabilities:

$$\mathbf{z} = \mathbf{h} \mathbf{W}^T + \mathbf{b} \quad \text{(logits)}$$
$$P(y_i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)} \quad \text{(softmax)}$$

## 2. Architecture
```
Decoder Output [n, d_model]
     |
  Linear(|V|) -- projects to vocabulary size
     |
  Logits [n, |V|]
     |
  Softmax -- normalizes to probabilities
     |
  Token Probabilities [n, |V|]
```

## 3. Temperature
At inference, temperature modifies the distribution sharpness:

$$P_T(y_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

T < 1: sharper (more deterministic). T > 1: flatter (more random).

## 4. Training Loss
Cross-entropy loss between predicted distribution and true one-hot target:

$$\mathcal{L} = -\sum_{t} \log P(y_t^* | y_{<t}, x)$$

## 5. Inference
At each step, the model selects the next token via sampling or greedy decoding from this distribution. The selected token is fed back as input for the next step.
## 📝 Self-Assessment
<details class="mcq"><summary>Output projection dimension change?</summary><ol type="A"><li>d_model -> d_model</li><li>d_model -> |V|. Projects hidden state to vocabulary-sized logits</li><li>d_model -> d_ff</li><li>d_model -> 1</li></ol><div class="answer">✅ Answer: B -- Linear layer maps from model dimension to vocabulary size. Each output is a logit for one token.</div></details>
<details class="mcq"><summary>Temperature T < 1 effect?</summary><ol type="A"><li>Flatter</li><li>Sharper -- high-probability tokens become even more likely. More deterministic output</li><li>No effect</li><li>Random</li></ol><div class="answer">✅ Answer: B -- Dividing logits by small T amplifies differences. Peak gets sharper; entropy decreases.</div></details>
<details class="mcq"><summary>Training loss function?</summary><ol type="A"><li>MSE</li><li>Cross-entropy: -sum(log P(y_t* | context)). Maximizes probability of correct token</li><li>L1</li><li>Hinge</li></ol><div class="answer">✅ Answer: B -- CE loss = negative log-likelihood of the correct token. Standard for classification over large vocabularies.</div></details>
<details class="mcq"><summary>Output probability for all tokens sums to?</summary><ol type="A"><li>|V|</li><li>1 -- softmax ensures a valid probability distribution over the vocabulary</li><li>d_model</li><li>n</li></ol><div class="answer">✅ Answer: B -- Softmax normalizes so sum of probabilities = 1. Each value = P(this token | previous tokens).</div></details>
<details class="mcq"><summary>Inference: selected token used as?</summary><ol type="A"><li>Ignored</li><li>Fed back as input for the NEXT step -- autoregressive loop</li><li>Output only</li><li>Training signal</li></ol><div class="answer">✅ Answer: B -- The generated token is appended to the sequence and used as input to generate the following token.</div></details>
