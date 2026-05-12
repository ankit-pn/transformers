<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Tokenization</div>
# Greedy Decoding

## 1. Always Pick the Best
At each step, select the single most probable token:

$$y_t = \argmax_y P(y | y_{<t}, x)$$

## 2. Properties
- **Fast**: O(n) instead of O(kn) for beam search
- **Deterministic**: Same input always produces same output (at temperature=0)
- **Myopic**: Cannot recover from early mistakes
- **Repetitive**: Tends to get stuck in loops without penalties

## 3. When Greedy Works
Short-answer QA, classification, factual extraction -- tasks where only one output is correct and diversity is not needed.

## 4. Greedy vs Sampling
```python
# Greedy
next_token = torch.argmax(logits, dim=-1)

# Sampling (with temperature)
probs = F.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, 1)
```

## 5. Degeneracy
Without repetition penalty or temperature, greedy decoding often produces degenerate repetitive output: "I think that I think that I think that..."
## 📝 Self-Assessment
<details class="mcq"><summary>Greedy decoding?</summary><ol type="A"><li>Random selection</li><li>Always pick the highest-probability token at each step. Deterministic, fast, but can be repetitive</li><li>Multiple candidates</li><li>Sampling</li></ol><div class="answer">✅ Answer: B -- argmax over logits each step. Simple but myopic -- early mistakes cannot be corrected later.</div></details>
<details class="mcq"><summary>Greedy output diversity?</summary><ol type="A"><li>High</li><li>Zero -- deterministic. Same input always gives same output. No variation</li><li>Moderate</li><li>Depends</li></ol><div class="answer">✅ Answer: B -- At temperature=0 (or argmax), the model produces identical output for identical input. No randomness.</div></details>
<details class="mcq"><summary>Greedy degeneracy problem?</summary><ol type="A"><li>Too diverse</li><li>Gets stuck in repetitive loops -- "I think I think I think..." Model keeps selecting same high-prob token</li><li>Too slow</li><li>Memory heavy</li></ol><div class="answer">✅ Answer: B -- Without randomness or penalty, greedy often cycles. Repetition penalty or sampling breaks the cycle.</div></details>
<details class="mcq"><summary>When is greedy decoding appropriate?</summary><ol type="A"><li>Creative writing</li><li>Factual QA, classification -- tasks with one correct answer where diversity is not needed</li><li>Story generation</li><li>Chat</li></ol><div class="answer">✅ Answer: B -- When there is a single correct output, greedy is appropriate. For open-ended tasks, sampling is better.</div></details>
<details class="mcq"><summary>Greedy vs beam search (beam=1)?</summary><ol type="A"><li>Different</li><li>Same -- beam=1 is greedy decoding. Beam search with k=1 reduces to argmax at each step</li><li>Beam faster</li><li>Beam better</li></ol><div class="answer">✅ Answer: B -- With beam_width=1, beam search is identical to greedy decoding. Only one candidate maintained.</div></details>
