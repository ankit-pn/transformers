<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Transformer Blocks</div>
# Decoder Block Architecture

## 1. Structure
```
Input -> Masked Self-Attention -> Add & Norm -> Cross-Attention -> Add & Norm -> FFN -> Add & Norm -> Output
```

## 2. Three Sublayers
1. **Masked** Multi-Head Self-Attention (causal)
2. Multi-Head Cross-Attention (attends to encoder output)
3. Position-wise FFN with residual + LN

## 3. Causal Self-Attention
Uses causal mask: token i attends only to tokens 1..i. Enables autoregressive generation.

## 4. Cross-Attention
Q from decoder, K/V from encoder output. This is where the decoder "reads" the source sequence.

## 5. Decoder-Only Variants
GPT-style models omit the cross-attention sublayer entirely. They have only two sublayers: masked self-attention + FFN. All context is captured through self-attention over the prefix.
## 📝 Self-Assessment
<details class="mcq"><summary>Decoder sublayers vs encoder?</summary><ol type="A"><li>Same</li><li>Decoder adds cross-attention (3 sublayers) vs encoder 2. Third sublayer attends to encoder output</li><li>Encoder has more</li><li>Decoder has fewer</li></ol><div class="answer">✅ Answer: B -- Decoder: masked self-attention + cross-attention + FFN. Encoder: self-attention + FFN. Cross-attention connects to encoder.</div></details>
<details class="mcq"><summary>Decoder self-attention type?</summary><ol type="A"><li>Full</li><li>Masked/causal -- each token sees only previous tokens. Enables autoregressive generation</li><li>Bidirectional</li><li>Local</li></ol><div class="answer">✅ Answer: B -- Causal mask ensures token i only attends to positions <= i. Essential for left-to-right generation.</div></details>
<details class="mcq"><summary>Cross-attention in decoder?</summary><ol type="A"><li>Not present</li><li>Q from decoder, K/V from encoder output. The decoder reads the source sequence</li><li>Self-attention</li><li>FFN</li></ol><div class="answer">✅ Answer: B -- Cross-attention is the decoder link to the encoder. It retrieves relevant source information for each target token.</div></details>
<details class="mcq"><summary>GPT-style decoder sublayers?</summary><ol type="A"><li>3</li><li>2 -- masked self-attention + FFN. No cross-attention. All context from prefix via self-attention</li><li>1</li><li>4</li></ol><div class="answer">✅ Answer: B -- Decoder-only models omit cross-attention. Context is captured entirely through self-attention over the prefix.</div></details>
<details class="mcq"><summary>Decoder training vs inference difference?</summary><ol type="A"><li>Same</li><li>Training: teacher forcing (parallel). Inference: autoregressive (sequential). Causal mask handles both</li><li>Fully different</li><li>No parallel</li></ol><div class="answer">✅ Answer: B -- Causal mask enables parallel training with teacher forcing. Same mask ensures correct autoregressive behavior at inference.</div></details>
