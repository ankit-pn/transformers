<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Training</div>
# Next-Token Prediction

## 1. The Core Task
Given a prefix of tokens, predict the NEXT token:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})$$

## 2. Self-Supervised Learning
No labeled data needed! Any text corpus becomes training data. The text itself provides the targets. This is why LLMs can be trained on internet-scale data.

## 3. Causal Language Modeling
This task is called Causal Language Modeling (CLM). The model learns the conditional probability distribution of language.

## 4. Training Data Preparation
```python
# Input:  [The, cat, sat, on, the, mat]
# Target: [cat, sat, on, the, mat, EOS]
# The model predicts target[i] given input[:i+1]
```

## 5. Scaling
Next-token prediction scales to trillions of tokens. The loss decreases predictably with compute (scaling laws). More data + bigger model = better next-token prediction = more capable model.
## 📝 Self-Assessment
<details class="mcq"><summary>Next-token prediction training signal?</summary><ol type="A"><li>Labels</li><li>The text itself -- each token is the target for the prefix before it. Self-supervised</li><li>Human annotation</li><li>Rewards</li></ol><div class="answer">✅ Answer: B -- The next token in the sequence IS the label. No human annotation needed. Any text is training data.</div></details>
<details class="mcq"><summary>Why is this called self-supervised?</summary><ol type="A"><li>No supervision</li><li>The data provides its own labels. Token t+1 is the label for tokens 1..t. No human labeling</li><li>Weak labels</li><li>Semi-supervised</li></ol><div class="answer">✅ Answer: B -- Supervision signal is inherent in the data structure. The next word IS the correct answer.</div></details>
<details class="mcq"><summary>Scaling laws for next-token prediction?</summary><ol type="A"><li>No pattern</li><li>Loss decreases predictably with compute -- L = a * C^(-b). More data + bigger model = better prediction</li><li>Random</li><li>Linear</li></ol><div class="answer">✅ Answer: B -- Kaplan et al. (2020): cross-entropy loss follows a power law with compute. Predictable improvement with scaling.</div></details>
<details class="mcq"><summary>Training data format?</summary><ol type="A"><li>Input = Target</li><li>Input: tokens[0:n-1], Target: tokens[1:n]. Model predicts each next token</li><li>Input only</li><li>Random split</li></ol><div class="answer">✅ Answer: B -- Shifted by one: the model sees tokens 0..t and must predict token t+1. Standard for autoregressive LMs.</div></details>
<details class="mcq"><summary>Why is next-token prediction so powerful?</summary><ol type="A"><li>Artificial</li><li>Language fluency requires understanding -- to predict the next word well, the model must understand context, facts, reasoning</li><li>Trivial task</li><li>Limited</li></ol><div class="answer">✅ Answer: B -- Predicting next token well requires internalizing grammar, world knowledge, and reasoning patterns from training data.</div></details>
