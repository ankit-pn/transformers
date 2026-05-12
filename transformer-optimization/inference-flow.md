<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Optimization</div>
# Transformer Inference Flow

## 1. Encoder Inference
```python
x = token_embed(input_ids) + position_embed(positions)
for layer in encoder_layers:
    x = layer(x)  # Self-attention + FFN with residuals
encoder_output = x  # Contextualized representations
```
One forward pass, fully parallel. All tokens processed simultaneously.

## 2. Decoder Inference (Autoregressive)
```python
generated = [start_token]; encoder_kv = precompute_encoder_kv(encoder_output)
for t in range(max_tokens):
    logits = decoder_step(generated[-1], self_kv_cache, encoder_kv)
    next_token = sample(logits)
    generated.append(next_token); update_kv_cache(next_token)
```

## 3. Encoder KV Cache
Encoder output is static -- compute once, cache, reused for every decoder step via cross-attention.

## 4. Decoder KV Cache
Without cache: O(N^3) total. With cache: O(N^2) total. KV cache eliminates recomputation of attention for past tokens.
## 📝 Self-Assessment
<details class="mcq"><summary>Encoder inference parallelism?</summary><ol type="A"><li>Sequential</li><li>Fully parallel -- all input tokens processed in one forward pass</li><li>Semi-parallel</li><li>Token by token</li></ol><div class="answer">✅ Answer: B -- Encoder has no causal constraint. All tokens attend to all others simultaneously.</div></details>
<details class="mcq"><summary>Encoder output in decoder?</summary><ol type="A"><li>Ignored</li><li>Cached -- computed once and fed to ALL decoder layers via cross-attention</li><li>Recomputed</li><li>Only once</li></ol><div class="answer">✅ Answer: B -- Encoder output is static. Compute once, reuse for entire autoregressive generation.</div></details>
<details class="mcq"><summary>Inference without KV cache complexity?</summary><ol type="A"><li>O(N^2)</li><li>O(N^3) -- each step recomputes attention for all previous tokens</li><li>O(N)</li><li>O(N log N)</li></ol><div class="answer">✅ Answer: B -- Step t recomputes O(t) attention over t tokens. Sum = O(N^3). KV cache reduces to O(N^2).</div></details>
<details class="mcq"><summary>Decoder inference loop?</summary><ol type="A"><li>One pass</li><li>Generate -> append -> update KV cache -> repeat until EOS. Autoregressive</li><li>Parallel</li><li>Batch only</li></ol><div class="answer">✅ Answer: B -- Each iteration: forward pass for one new token, sample, update cache, check stop.</div></details>
<details class="mcq"><summary>Why cache encoder output?</summary><ol type="A"><li>Cannot cache</li><li>Encoder output is constant across all decoder steps. One-time compute saves redundant passes</li><li>Dynamic</li><li>Rules</li></ol><div class="answer">✅ Answer: B -- Encoder processes input once. Its output never changes during generation. Caching eliminates redundancy.</div></details>
