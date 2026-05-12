<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Tokenization</div>
# Beam Search Decoding

## 1. Exploring Multiple Hypotheses
Maintain k candidate sequences (the beam) and expand all at each step:

$$\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i | y_{<i}, x)$$

## 2. Algorithm
```python
beams = [([], 0.0)]  # (tokens, score)
for t in range(max_len):
    candidates = []
    for tokens, score in beams:
        logits = model.decode(tokens[-1])
        top_k = torch.topk(logits, k=beam_width)
        for tok, logprob in zip(top_k.indices, top_k.values):
            candidates.append((tokens + [tok], score + logprob))
    beams = sorted(candidates, key=lambda x: x[1])[-beam_width:]
```

## 3. Beam Width Tradeoff
| Width | Quality | Speed | Memory |
|---|---|---|---|
| 1 (greedy) | Baseline | 1x | 1x |
| 4 | Better | ~4x | ~4x |
| 8 | Good | ~8x | ~8x |
| 32 | Best | ~32x | ~32x |

## 4. Length Normalization
Prevent short-sequence bias:

$$\text{score}_{norm} = \frac{\sum \log P(y_i)}{|y|^\alpha}$$
## 📝 Self-Assessment
<details class="mcq"><summary>Beam search vs greedy?</summary><ol type="A"><li>Same</li><li>Beam explores k candidates in parallel; greedy takes only the best. Beam better for structured output (translation)</li><li>Greedy better</li><li>No difference</li></ol><div class="answer">✅ Answer: B -- Beam search maintains multiple hypotheses. Better for tasks with clear correctness criteria (MT, code).</div></details>
<details class="mcq"><summary>Beam width = 4 means?</summary><ol type="A"><li>4 layers</li><li>4 candidate sequences maintained and expanded at each step. 4x compute and memory of greedy</li><li>4 tokens</li><li>4 models</li></ol><div class="answer">✅ Answer: B -- The beam contains 4 top-scoring partial sequences. Each is expanded to generate new candidates per step.</div></details>
<details class="mcq"><summary>Length normalization purpose?</summary><ol type="A"><li>Speed</li><li>Prevents bias toward short sequences -- raw log-prob sum favors shorter outputs. Dividing by length^alpha corrects</li><li>Accuracy</li><li>Memory</li></ol><div class="answer">✅ Answer: B -- sum(log P) decreases with length. Length normalization makes scores comparable across different-length candidates.</div></details>
<details class="mcq"><summary>When NOT to use beam search?</summary><ol type="A"><li>Always use</li><li>Creative writing, chat, open-ended generation -- sampling produces more natural, diverse output</li><li>Translation</li><li>Code</li></ol><div class="answer">✅ Answer: B -- Beam search optimizes for likelihood, producing bland/repetitive text in open-ended tasks. Sampling is preferred.</div></details>
<details class="mcq"><summary>Beam search output selection?</summary><ol type="A"><li>Random</li><li>Highest cumulative log-probability sequence after length normalization. Deterministic given beam width</li><li>First beam</li><li>Shortest</li></ol><div class="answer">✅ Answer: B -- The sequence with the highest normalized total score is selected. Deterministic output for a given beam width.</div></details>
