<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Mechanics</div>
# Introduction to Attention

## 1. The Core Idea
Attention allows a model to dynamically focus on relevant parts of the input when producing each output:

$$\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{ti} \mathbf{h}_i$$

## 2. Intuition
When translating "The cat sat on the mat" to French, generating "chat" focuses on "cat", generating "assis" focuses on "sat". Attention learns soft alignments automatically.

## 3. Bahdanau Attention (Additive)
$$e_{ti} = \mathbf{v}^T \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_i)$$
$$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^{T} \exp(e_{tj})}$$

## 4. Key Benefit
$$\text{Path Length}_{attention} = O(1) \text{ vs } \text{Path Length}_{RNN} = O(T)$$
## 📝 Self-Assessment
<details class="mcq"><summary>Core idea of attention?</summary><ol type="A"><li>Compress into one vector</li><li>Dynamically weight different input positions per output step -- the decoder focuses on relevant inputs</li><li>Process sequentially</li><li>Ignore input</li></ol><div class="answer">✅ Answer: B -- Weighted sum where weights vary per output, focusing on relevant source positions.</div></details>
<details class="mcq"><summary>Who introduced first attention?</summary><ol type="A"><li>Vaswani (2017)</li><li>Bahdanau et al. (2015) -- additive attention for NMT</li><li>Hochreiter</li><li>Sutskever</li></ol><div class="answer">✅ Answer: B -- Bahdanau introduced attention to MT in 2015. Vaswani later contributed self-attention and Transformers in 2017.</div></details>
<details class="mcq"><summary>Path length attention vs RNN?</summary><ol type="A"><li>Same O(T)</li><li>O(1) for attention vs O(T) for RNN -- direct access between any positions</li><li>O(log T)</li><li>O(T^2)</li></ol><div class="answer">✅ Answer: B -- Attention creates direct connections. Any query can access any key in one step versus T sequential RN steps.</div></details>
<details class="mcq"><summary>What does alpha_{ti} represent?</summary><ol type="A"><li>Input token</li><li>Attention weight -- how much decoder attends to source position i at step t</li><li>Output token</li><li>Hidden state</li></ol><div class="answer">✅ Answer: B -- Alpha weights are normalized scores summing to 1. They represent per-position importance for current decoding step.</div></details>
<details class="mcq"><summary>Global vs local attention?</summary><ol type="A"><li>Same</li><li>Global: all positions. Local: window around predicted position. Local cheaper for long sequences</li><li>Global faster</li><li>Local more memory</li></ol><div class="answer">✅ Answer: B -- Global attention is O(T). Local attention restricts to window of size W << T, reducing cost to O(W).</div></details>
