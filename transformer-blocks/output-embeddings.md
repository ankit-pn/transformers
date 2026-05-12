<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Transformer Blocks</div>
# Output Embeddings

## 1. Token Prediction
The output embedding (often called the projection layer) maps decoder hidden states to vocabulary logits:

$$\text{logits} = H W_{out}^T + b_{out}$$

Where W_out has shape |V| x d_model (or d_model x |V| depending on convention).

## 2. Weight Tying
Many models tie input and output embeddings:

$$W_{out} = E$$

This saves |V| x d_model parameters and provides a useful inductive bias: tokens with similar input embeddings have similar output projections.

## 3. Logits to Probabilities
$$P(y | x) = \text{softmax}(\text{logits})$$

Probability distribution over the entire vocabulary for the next token.

## 4. Dimension Flow
```
Hidden state [n, d_model] -> Linear(|V|) -> Logits [n, |V|] -> Softmax -> Probs [n, |V|]
```

## 5. Top-k/Top-p
In practice, sampling from the full 50K vocabulary is expensive and noisy. Top-k and top-p truncation focus on the most likely tokens:

$$P_{filtered}(y) = \frac{P(y) \cdot \mathbb{1}[\text{valid}]}{\sum_{valid} P(y')}$$
## 📝 Self-Assessment
<details class="mcq"><summary>Output projection shape?</summary><ol type="A"><li>d_model x d_model</li><li>|V| x d_model -- maps hidden state to vocabulary-sized logits</li><li>n x d_model</li><li>|V| x n</li></ol><div class="answer">✅ Answer: B -- Projects from model dimension to vocabulary size. Each output dimension corresponds to one token.</div></details>
<details class="mcq"><summary>What is weight tying?</summary><ol type="A"><li>Tying attention weights</li><li>Sharing input and output embedding matrices -- W_out = E. Saves params, provides inductive bias</li><li>Tying layer weights</li><li>Tying biases</li></ol><div class="answer">✅ Answer: B -- The same embedding matrix used for input lookup and output projection. Saves |V| x d_model parameters.</div></details>
<details class="mcq"><summary>Why tie embeddings?</summary><ol type="A"><li>Required</li><li>Parameter efficiency + inductive bias: semantically similar tokens have similar input AND output representations</li><li>For speed</li><li>No reason</li></ol><div class="answer">✅ Answer: B -- Saves ~600M params in GPT-3. Similar tokens get similar input embeddings AND output projections.</div></details>
<details class="mcq"><summary>Logits to probabilities?</summary><ol type="A"><li>Argmax</li><li>Softmax(logits) -- converts to proper probability distribution summing to 1 over vocabulary</li><li>Sigmoid</li><li>ReLU</li></ol><div class="answer">✅ Answer: B -- Softmax normalizes logits to [0,1] with sum=1. Each value = P(token_i | context).</div></details>
<details class="mcq"><summary>Why top-k/top-p for sampling?</summary><ol type="A"><li>Full vocab better</li><li>Full 50K vocabulary sampling is computationally expensive and produces low-quality tokens. Truncation helps</li><li>Faster only</li><li>Required</li></ol><div class="answer">✅ Answer: B -- Most vocabulary tokens have near-zero probability. Filtering to reasonable candidates improves quality and speed.</div></details>
