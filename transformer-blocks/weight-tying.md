<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Transformer Blocks</div>
# Weight Tying

## 1. Sharing Embeddings
Input embedding matrix E and output projection matrix W_out are shared:

$$E = W_{out} \in \mathbb{R}^{|V| \times d_{model}}$$

$$\text{input\_embed}(id) = E[id]$$
$$\text{output\_logits} = H \cdot E^T$$

## 2. Motivation
1. Parameter efficiency: saves |V| x d_model parameters (~30% of total in some models)
2. Semantic consistency: similar tokens treated similarly in input and output
3. Regularization: fewer parameters reduces overfitting

## 3. Transformer Original Paper
The original Transformer paper explicitly mentions weight tying between the two embedding layers and the pre-softmax linear transformation.

## 4. When NOT to Tie
When |V| is very large relative to d_model, or when input and output token distributions differ significantly (e.g., multilingual translation with different source/target vocabularies).

## 5. Modern Practice
Most models tie weights: GPT, BERT, T5, Llama. It is the default unless there is a specific reason not to.
## 📝 Self-Assessment
<details class="mcq"><summary>What is weight tying in Transformers?</summary><ol type="A"><li>Sharing attention weights</li><li>Sharing input embedding E and output projection W_out. E = W_out</li><li>Sharing FFN weights</li><li>Tying biases</li></ol><div class="answer">✅ Answer: B -- The same matrix maps tokens -> vectors (input) and vectors -> logits (output).</div></details>
<details class="mcq"><summary>Parameter savings from weight tying?</summary><ol type="A"><li>No savings</li><li>|V| x d_model -- e.g., 50K x 768 = 38M for BERT. Significant fraction of total</li><li>Negligible</li><li>Doubles params</li></ol><div class="answer">✅ Answer: B -- Input embedding + output projection = 2 x |V| x d_model. Tying halves this to |V| x d_model.</div></details>
<details class="mcq"><summary>Why tie weights beyond saving params?</summary><ol type="A"><li>No other reason</li><li>Semantic consistency -- similar tokens have similar input AND output representations. Inductive bias</li><li>Speed</li><li>Required</li></ol><div class="answer">✅ Answer: B -- Tokens with nearby input embeddings also get nearby output projections. Coherent semantic space.</div></details>
<details class="mcq"><summary>Original Transformer weight tying?</summary><ol type="A"><li>Not used</li><li>Yes -- the paper explicitly mentions sharing embedding and pre-softmax weights</li><li>Only encoder</li><li>Only decoder</li></ol><div class="answer">✅ Answer: B -- The original Transformer paper describes this as a design choice for parameter efficiency.</div></details>
<details class="mcq"><summary>When NOT to tie weights?</summary><ol type="A"><li>Always tie</li><li>When source and target vocabularies differ significantly (e.g., multilingual with different tokenizers)</li><li>Never</li><li>Only for small models</li></ol><div class="answer">✅ Answer: B -- Separate vocabularies require separate embeddings. Tying assumes same |V| for input and output.</div></details>
