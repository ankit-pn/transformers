<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Tokenization</div>
# Top-K Sampling

## 1. Truncate to K Best
Only sample from the k tokens with highest probability:

$$V_k = \{y_{(1)}, ..., y_{(k)}\} \text{ (sorted by probability)}$$
$$P'(y_i) = \frac{P(y_i)}{\sum_{j \in V_k} P(y_j)} \text{ if } i \in V_k \text{ else } 0$$

## 2. Why Top-K?
Prevents sampling from the long tail of near-zero-probability tokens. Without truncation, the model occasionally selects nonsensical tokens from the 50K vocabulary.

## 3. k Selection
| k | Behavior |
|---|---|
| 1 | Greedy (no randomness) |
| 10-20 | Conservative (code, structured tasks) |
| 40-50 | Balanced (most common) |
| 100+ | Creative (more diversity) |
| |V| | Full distribution (no truncation) |

## 4. Top-K Problem
Rigid cutoff -- k is fixed regardless of the distribution shape. On flat distributions, cuts off legitimate options. On peaked distributions, includes irrelevant tokens with near-zero probability. Top-P solves this.
## 📝 Self-Assessment
<details class="mcq"><summary>Top-k sampling?</summary><ol type="A"><li>All tokens</li><li>Only sample from k highest-probability tokens. Truncates the long tail of unlikely tokens</li><li>Bottom tokens</li><li>Random tokens</li></ol><div class="answer">✅ Answer: B -- Top-k limits the sampling pool to the k most probable tokens, renormalizing within that set.</div></details>
<details class="mcq"><summary>Typical k value for balanced sampling?</summary><ol type="A"><li>1</li><li>40-50 -- enough tokens for diversity, few enough to exclude low-probability noise</li><li>10</li><li>1000</li></ol><div class="answer">✅ Answer: B -- k=40-50 is the common default. Wide enough for diverse output; narrow enough to avoid nonsense tokens.</div></details>
<details class="mcq"><summary>Top-k=1 behavior?</summary><ol type="A"><li>Random</li><li>Greedy decoding -- only the single highest-probability token is considered. Deterministic</li><li>Uniform</li><li>Top-p</li></ol><div class="answer">✅ Answer: B -- With k=1, only the argmax token is in V_k. Sampling from a set of 1 = deterministic greedy decoding.</div></details>
<details class="mcq"><summary>Main limitation of top-k?</summary><ol type="A"><li>Too slow</li><li>Rigid cutoff -- always includes exactly k tokens regardless of distribution shape. May include or exclude too many</li><li>Too random</li><li>Too deterministic</li></ol><div class="answer">✅ Answer: B -- For peaked distribution (one token 0.95), k=50 includes 49 near-zero tokens. For flat, k=50 may cut off viable options.</div></details>
<details class="mcq"><summary>Top-k vs top-p: which adapts to distribution?</summary><ol type="A"><li>Top-k</li><li>Top-p -- it includes however many tokens needed to reach cumulative probability p. Top-k is fixed</li><li>Both adapt</li><li>Neither</li></ol><div class="answer">✅ Answer: B -- Top-p dynamically adjusts the number of considered tokens based on the probability distribution. Top-k is rigid.</div></details>
