<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Training</div>
# Autoregressive Decoding

## 1. One Token at a Time
Generates output sequentially, feeding each predicted token back as input:

$$y_t \sim P(y_t | y_1, ..., y_{t-1}, \mathbf{x})$$

## 2. Process
```python
generated = []
for t in range(max_len):
    logits = model(generated, encoder_output)
    next_token = sample(logits[:, -1, :])
    generated.append(next_token)
    if next_token == eos_token:
        break
```

## 3. Why Sequential?
Each token depends on all previous tokens. Parallel generation would require knowing future tokens in advance. The causal structure enforces the left-to-right constraint.

## 4. Inference Cost
$$T_{inference} = N_{tokens} \times T_{per\_token}$$
Linear in output length. Each step processes the growing prefix. KV caching reduces per-step cost to near-constant.
## 📝 Self-Assessment
<details class="mcq"><summary>Autoregressive means?</summary><ol type="A"><li>Parallel</li><li>Tokens generated one at a time, each conditioned on all previously generated tokens</li><li>Bidirectional</li><li>Random</li></ol><div class="answer">✅ Answer: B -- y_t depends on y_1..y_{t-1}. Sequential generation with feedback loop.</div></details>
<details class="mcq"><summary>Why cannot generate all tokens at once?</summary><ol type="A"><li>Too slow</li><li>Causal constraint -- each token depends on previous tokens which do not exist yet at generation time</li><li>Memory</li><li>Not parallel</li></ol><div class="answer">✅ Answer: B -- Tokens are sequentially dependent. Token 5 needs token 4, which needs token 3... Cannot be parallelized.</div></details>
<details class="mcq"><summary>Inference time for N tokens?</summary><ol type="A"><li>Constant</li><li>Linear in N -- N x T_per_token. KV caching makes per-token cost nearly constant</li><li>O(N^2)</li><li>O(log N)</li></ol><div class="answer">✅ Answer: B -- Each token requires one forward pass. With KV cache, each pass is ~constant cost. So total = O(N).</div></details>
<details class="mcq"><summary>What happens when EOS is generated?</summary><ol type="A"><li>Continues</li><li>Generation stops -- EOS signals natural completion. max_tokens provides a safety limit</li><li>Error</li><li>Restarts</li></ol><div class="answer">✅ Answer: B -- EOS token indicates the model considers the output complete. Generation halts; no further tokens produced.</div></details>
<details class="mcq"><summary>Autoregressive generation feeding?</summary><ol type="A"><li>Encoder output</li><li>Previous generated tokens -- each new token is appended to the sequence and used for the next step</li><li>Random noise</li><li>Ground truth</li></ol><div class="answer">✅ Answer: B -- The output at step t becomes part of the input at step t+1. This is the autoregressive feedback loop.</div></details>
