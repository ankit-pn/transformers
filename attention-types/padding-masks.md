<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Types</div>
# Padding Masks

## 1. Handling Variable Lengths
Sequences in a batch have different lengths. Shorter sequences are padded to the max length. Padding masks prevent attention to these padding tokens:

$$\text{Mask}_{ij} = \begin{cases} 0 & \text{if token j is real} \\ -\infty & \text{if token j is padding} \end{cases}$$

## 2. Implementation
```python
# input_ids: [2, 5] -- batch of 2, seq_len 5
# seq: [101, 2003, 1037, 3231, 102] -- real tokens
# seq: [101, 2054, 2003, 0, 0]      -- last 2 are padding
padding_mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
# Shape: [batch, 1, 1, seq_len]
```

## 3. Combined Mask
Causal AND padding masks are combined:
```python
combined_mask = causal_mask & padding_mask
```

## 4. Why Not Just Truncate?
Batching requires uniform tensor shapes. Padding enables batching while ensuring padding tokens do not affect the output. Without padding masks, zero vectors would participate in attention, degrading quality.

## 5. Efficiency
For padded sequences, the effective sequence length is shorter. The padding mask prevents wasted computation on padding tokens.
## 📝 Self-Assessment
<details class="mcq"><summary>What are padding masks?</summary><ol type="A"><li>Causal masks</li><li>Prevents attention to padding tokens in batched sequences of different lengths</li><li>Dropout masks</li><li>Position masks</li></ol><div class="answer">✅ Answer: B -- Shorter sequences are padded to match batch max length. Padding mask ensures padding tokens get zero attention.</div></details>
<details class="mcq"><summary>Why not just truncate?</summary><ol type="A"><li>Always truncate</li><li>Batching requires uniform tensor shapes. Padding enables batching while ignoring filler tokens</li><li>Faster</li><li>Better</li></ol><div class="answer">✅ Answer: B -- Different-length sequences need padding for batch tensor. Padding mask prevents padded zeros from affecting attention.</div></details>
<details class="mcq"><summary>How to create padding mask?</summary><ol type="A"><li>Random</li><li>input_ids != pad_token_id -- boolean mask where True=real token, False=padding. Applied as -inf to False positions</li><li>All ones</li><li>Based on length</li></ol><div class="answer">✅ Answer: B -- Simple check: where input_ids match PAD token. Those positions get -inf attention, others get normal scores.</div></details>
<details class="mcq"><summary>Combined causal + padding mask?</summary><ol type="A"><li>Cannot combine</li><li>Multiply/AND the masks. Position j gets -inf if j>i (causal) OR if j is padding</li><li>Only one</li><li>Add masks</li></ol><div class="answer">✅ Answer: B -- Both masks serve different purposes. Combined ensures attention is only to valid previous tokens.</div></details>
<details class="mcq"><summary>Effect of no padding mask?</summary><ol type="A"><li>Nothing</li><li>Padding zeros participate in attention and contribute to context vector -- degrades quality and wastes compute</li><li>Better</li><li>Faster</li></ol><div class="answer">✅ Answer: B -- Zero vectors in attention would shift the context toward zero. Padding mask prevents this degradation.</div></details>
