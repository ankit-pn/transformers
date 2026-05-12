<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Multi-Head Attention</div>
# Linear Projection Matrices

## 1. The Four Projections
Multi-head attention uses four learnable projection matrices:

$$Q = X W_q, \quad K = X W_k, \quad V = X W_v, \quad \text{Output} = \text{Concat} \cdot W_o$$

## 2. Dimensions
| Matrix | Shape | Purpose |
|---|---|---|
| W_q | d_model x d_model | Projects input to query space |
| W_k | d_model x d_model | Projects input to key space |
| W_v | d_model x d_model | Projects input to value space |
| W_o | d_model x d_model | Mixes concatenated head outputs |

## 3. Parameter Count
For d_model=512, each projection is 512 x 512 = 262K. Four = ~1M parameters per attention layer. Negligible vs FFN block.

$$\text{Params}_{\text{MHA}} = 4 \cdot d_{model}^2 \text{ (plus bias: } 4 \cdot d_{model})$$

## 4. Why Separate Projections?
Q, K, V serve different roles: Q encodes queries (what to look for), K encodes keys (what to match against), V encodes values (what to retrieve). Separate projections allow optimization for each role.

## 5. Weight Sharing
Some architectures share W_k and W_v (ALBERT) or use tied weights to reduce parameters. This slightly reduces quality but significantly reduces model size.
## 📝 Self-Assessment
<details class="mcq"><summary>How many projection matrices in MHA?</summary><ol type="A"><li>2</li><li>4 -- W_q, W_k, W_v (input projections) and W_o (output projection)</li><li>3</li><li>1</li></ol><div class="answer">✅ Answer: B -- Three projections for Q,K,V from input, plus W_o to mix concatenated head outputs. Four total learnable matrices.</div></details>
<details class="mcq"><summary>W_q, W_k, W_v dimensions?</summary><ol type="A"><li>d_model x d_k</li><li>d_model x d_model -- project full model dim, then split into heads via reshape</li><li>d_k x d_k</li><li>n x d_model</li></ol><div class="answer">✅ Answer: B -- Each projects from d_model to d_model. The head-splitting happens via reshape (view), not projection.</div></details>
<details class="mcq"><summary>Parameter count for MHA projections (d=512)?</summary><ol type="A"><li>~128K</li><li>~1M -- 4 x (512 x 512) = 4 x 262K = 1,048,576 parameters</li><li>~4M</li><li>~256K</li></ol><div class="answer">✅ Answer: B -- Four d_model x d_model matrices. 512^2=262K each x 4 = ~1M. Small vs FFN block (4*d_model^2).</div></details>
<details class="mcq"><summary>Why separate Q, K, V projections?</summary><ol type="A"><li>Not needed</li><li>Different roles: Q for querying, K for matching, V for retrieving. Separate projections optimize for each role</li><li>Historical</li><li>Speed</li></ol><div class="answer">✅ Answer: B -- Q learns what to ask, K learns how to be found, V learns what to contribute. Asymmetric projections for asymmetric roles.</div></details>
<details class="mcq"><summary>ALBERT weight sharing?</summary><ol type="A"><li>Shares all</li><li>Shares W_k and W_v (or all attention params) across layers -- reduces parameters 10-20x with small quality drop</li><li>No sharing</li><li>Only W_o</li></ol><div class="answer">✅ Answer: B -- ALBERT ties attention weights across layers. Dramatic parameter reduction enables deeper models with same memory.</div></details>
