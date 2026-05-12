<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Transformer Blocks</div>
# Decoder Stack

## 1. Stacked Decoder Blocks
$$H_d^{(l)} = \text{DecoderBlock}_l(H_d^{(l-1)}, H_e^{(L)})$$

Where H_e^(L) is the final encoder output (used in cross-attention at every decoder layer).

## 2. Depth
Mirrors encoder: 6 layers in base Transformer, 12-96 in modern models. GPT-3 has 96 decoder layers.

## 3. Cross-Attention Input
Every decoder layer attends to the SAME encoder output. This means the encoder output is accessed at multiple levels of abstraction in the decoder.

## 4. Autoregressive Generation
The decoder stack produces output one token at a time:

```python
for t in range(max_len):
    output = decoder_stack(prev_tokens, encoder_output)
    next_token = sample(output[:, -1, :])
    prev_tokens = torch.cat([prev_tokens, next_token], dim=1)
```

## 5. KV Caching
At inference, the decoder caches K,V from self-attention to avoid recomputing for previous tokens. This is the key optimization that makes autoregressive decoding practical.
## 📝 Self-Assessment
<details class="mcq"><summary>Decoder layers access encoder output?</summary><ol type="A"><li>Only last layer</li><li>ALL decoder layers -- cross-attention at every layer attends to the same encoder output</li><li>First layer only</li><li>Random layer</li></ol><div class="answer">✅ Answer: B -- Each decoder layer has cross-attention. All layers access the full encoder output at different abstraction levels.</div></details>
<details class="mcq"><summary>Decoder stack depth (GPT-3)?</summary><ol type="A"><li>12</li><li>96 layers -- decoder-only models stack many blocks. Depth is primary scaling dimension</li><li>6</li><li>24</li></ol><div class="answer">✅ Answer: B -- GPT-3 uses 96 decoder layers. Depth is the primary scaling dimension for decoder-only models.</div></details>
<details class="mcq"><summary>Autoregressive generation loop?</summary><ol type="A"><li>All at once</li><li>One token per step: decode full prefix, sample last position, append to prefix. Repeat until EOS</li><li>Batch generation</li><li>Random</li></ol><div class="answer">✅ Answer: B -- The decoder generates sequentially. Each step produces one new token based on all previous tokens.</div></details>
<details class="mcq"><summary>KV caching purpose?</summary><ol type="A"><li>Memory</li><li>Avoid recomputing K,V for previous tokens at each generation step. Critical for efficient autoregressive decoding</li><li>Speed only</li><li>Training</li></ol><div class="answer">✅ Answer: B -- Without KV cache, each step would recompute attention for all previous tokens. With cache: O(n) total vs O(n^2).</div></details>
<details class="mcq"><summary>Decoder vs encoder depth relationship?</summary><ol type="A"><li>Always same</li><li>Can differ -- modern models often use asymmetric depths or decoder-only architectures</li><li>Decoder always deeper</li><li>Encoder always deeper</li></ol><div class="answer">✅ Answer: B -- Original Transformer had equal depths. Modern decoder-only models (GPT) have zero encoder layers.</div></details>
