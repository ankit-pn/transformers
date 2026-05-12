<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Positional Encoding</div>
# Learned Positional Embeddings

## 1. Trainable Position Vectors
Instead of fixed sinusoidal functions, learn an embedding for each position:

$$PE \in \mathbb{R}^{max\_len \times d_{model}}$$

## 2. BERT/GPT Approach
```python
self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
# During forward pass
position_ids = torch.arange(seq_len, device=input_ids.device)
position_embeds = self.position_embeddings(position_ids)
hidden_states = token_embeds + position_embeds
```

## 3. Advantages
- Task-adaptive: learns position patterns relevant to the task
- Can learn non-sinusoidal patterns (e.g., document structure, section boundaries)
- Simple to implement (just another embedding)

## 4. Limitations
- Fixed max length: cannot handle sequences longer than training max
- No built-in relative position info
- More parameters than sinusoidal (though negligible vs model size)
- No extrapolation capability

## 5. Hybrid and Modern Approaches
**RoPE** (Rotary Position Embedding): Encodes position by rotating Q,K vectors. Combines benefits of absolute and relative. Used in Llama, Mistral, GPT-NeoX.

**ALiBi**: Adds linear bias to attention scores (no embeddings). Simple, excellent extrapolation.
## 📝 Self-Assessment
<details class="mcq"><summary>Learned vs sinusoidal PE?</summary><ol type="A"><li>Same</li><li>Learned: trainable, task-adaptive but fixed max length. Sinusoidal: fixed function, extrapolates to any length</li><li>Learned always better</li><li>Sinusoidal faster</li></ol><div class="answer">✅ Answer: B -- Learned adapts to task but has max length limit. Sinusoidal extrapolates but is fixed. RoPE bridges the gap.</div></details>
<details class="mcq"><summary>BERT/GPT position embedding type?</summary><ol type="A"><li>Sinusoidal</li><li>Learned -- nn.Embedding(max_len, d_model). Positions 0..511 for BERT, 0..1024 for GPT</li><li>RoPE</li><li>ALiBi</li></ol><div class="answer">✅ Answer: B -- Early Transformers (BERT, GPT-1/2) used learned position embeddings trained from scratch with max length limit.</div></details>
<details class="mcq"><summary>Main limitation of learned PEs?</summary><ol type="A"><li>Slow</li><li>Fixed max sequence length -- cannot process sequences longer than training max without interpolation</li><li>Too many params</li><li>Low accuracy</li></ol><div class="answer">✅ Answer: B -- Learned PEs are defined only for positions 0..max_len. Extrapolation requires interpolation or new training.</div></details>
<details class="mcq"><summary>What is RoPE?</summary><ol type="A"><li>Sinusoidal variant</li><li>Rotary Position Embedding -- encodes position by rotating Q,K vectors. Relative position captured implicitly</li><li>Learned variant</li><li>Absolute only</li></ol><div class="answer">✅ Answer: B -- RoPE applies rotation matrices to Q and K based on position. Dot product Q^T K naturally captures relative position.</div></details>
<details class="mcq"><summary>ALiBi position encoding?</summary><ol type="A"><li>Embedding</li><li>Adds linear bias to attention scores (no embeddings). Simple, excellent extrapolation to 10x training length</li><li>Sinusoidal</li><li>Learned</li></ol><div class="answer">✅ Answer: B -- ALiBi adds bias = -m * |i-j| to attention scores. Encourages local attention. Extrapolates remarkably well.</div></details>
