<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Tokenization</div>
# Top-P (Nucleus) Sampling

## 1. Dynamic Truncation
Sample from the smallest set of tokens whose cumulative probability >= p:

$$\text{Smallest } V_p : \sum_{y \in V_p} P(y) \geq p$$

## 2. Why Top-P?
Adapts to the distribution shape. On peaked distributions (one token 0.95), only 1-2 tokens included. On flat distributions, many tokens included. No rigid cutoff.

## 3. p Selection
| p | Behavior |
|---|---|
| 0.5 | Narrow -- only the most confident tokens |
| 0.9 | Balanced (common default) |
| 0.95 | Moderate diversity |
| 1.0 | No truncation (full distribution) |

## 4. Top-K + Top-P Combined
Common practice: apply BOTH for defense in depth

```python
probs = top_k_filter(probs, k=50)  # Hard ceiling
probs = top_p_filter(probs, p=0.9)  # Dynamic floor
next_token = torch.multinomial(probs, 1)
```

## 5. Nucleus Sampling Name
The selected set V_p is the "nucleus" of the distribution -- the smallest set containing most of the probability mass. The tail (outside the nucleus) is discarded.
## 📝 Self-Assessment
<details class="mcq"><summary>Top-p (nucleus) sampling?</summary><ol type="A"><li>Fixed k tokens</li><li>Dynamically selects smallest token set with cumulative prob >= p. Adapts to distribution shape</li><li>Random k</li><li>All tokens</li></ol><div class="answer">✅ Answer: B -- Top-p includes however many tokens are needed to reach p probability mass. Adapts: 2 tokens for peaked, 100+ for flat.</div></details>
<details class="mcq"><summary>Why nucleus in the name?</summary><ol type="A"><li>Random</li><li>The smallest set containing mass p is the nucleus of the distribution. The unreliable tail is discarded</li><li>From paper author</li><li>Historical</li></ol><div class="answer">✅ Answer: B -- Like an atomic nucleus containing most mass, V_p contains most of the probability mass. The tail is pruned.</div></details>
<details class="mcq"><summary>p=0.9 on highly peaked distribution (top token=0.95)?</summary><ol type="A"><li>50 tokens</li><li>~1-2 tokens -- only need top token(s) to reach 0.9 cumulative probability. Efficient truncation</li><li>10 tokens</li><li>All tokens</li></ol><div class="answer">✅ Answer: B -- With one token at 0.95, cumulative prob already exceeds 0.9. Only that token (or maybe a second) is included.</div></details>
<details class="mcq"><summary>Top-p vs top-k adaptability?</summary><ol type="A"><li>Top-k adapts</li><li>Top-p adapts to distribution shape; top-k is rigid. Top-p is generally preferred for this reason</li><li>Both rigid</li><li>Neither adapts</li></ol><div class="answer">✅ Answer: B -- Top-p dynamic token count makes it better suited for diverse distributions. Top-k fixed count is simpler but less flexible.</div></details>
<details class="mcq"><summary>Combined top-k + top-p?</summary><ol type="A"><li>Cannot combine</li><li>Common practice -- k=50 provides hard ceiling; p=0.9 provides dynamic floor. Defense in depth</li><li>Only one</li><li>Incompatible</li></ol><div class="answer">✅ Answer: B -- Top-k prevents too many tokens; top-p prevents including near-zero-prob tokens when distribution is peaked. Best of both.</div></details>
