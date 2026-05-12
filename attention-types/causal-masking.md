<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Types</div>
# Causal Masking

## 1. Preventing Information Leakage
In autoregressive generation, token i must NOT attend to tokens j > i (future tokens). Causal masking enforces this:

$$\text{Mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

## 2. Implementation
```python
def causal_mask(n):
    mask = torch.triu(torch.ones(n, n), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))
# Result: lower triangular (including diagonal) = 0, upper = -inf
```

## 3. Attention Matrix
After applying mask and softmax, the attention matrix is lower triangular:
```
[alpha_11   0       0       0   ]
[alpha_21  alpha_22 0       0   ]
[alpha_31  alpha_32 alpha_33 0   ]
[alpha_41  alpha_42 alpha_43 alpha_44]
```

## 4. Why Causal?
Without masking, the model could "cheat" by looking at future tokens during training. This would make autoregressive generation impossible -- the model would depend on information it does not have at inference time.

## 5. Training vs Inference
During training, causal masking allows parallel processing of the entire sequence (teacher forcing). At inference, tokens are generated one at a time -- the causal constraint is naturally satisfied.
## 📝 Self-Assessment
<details class="mcq"><summary>What is causal masking?</summary><ol type="A"><li>Random masking</li><li>Prevents token i from attending to token j > i (future tokens). Enforces autoregressive constraint</li><li>Padding mask</li><li>Dropout</li></ol><div class="answer">✅ Answer: B -- Lower triangular attention matrix. Each token can only see itself and previous tokens, never future ones.</div></details>
<details class="mcq"><summary>Why is causal masking necessary?</summary><ol type="A"><li>Performance</li><li>Without it, model would cheat by looking at future tokens -- makes autoregressive inference impossible</li><li>Memory</li><li>Speed</li></ol><div class="answer">✅ Answer: B -- Training with future information creates train-inference mismatch. At inference, future tokens do not exist yet.</div></details>
<details class="mcq"><summary>Shape of causal mask?</summary><ol type="A"><li>Full matrix</li><li>Upper triangular is -inf; lower triangular (including diagonal) is 0. Lower triangular attention</li><li>All zeros</li><li>Diagonal only</li></ol><div class="answer">✅ Answer: B -- Mask prevents upper triangle attention. Token i sees tokens 1..i, never i+1..n.</div></details>
<details class="mcq"><summary>Training with causal masking?</summary><ol type="A"><li>Sequential only</li><li>Parallel -- teacher forcing with causal mask processes all tokens simultaneously with correct masking</li><li>Slow</li><li>Not possible</li></ol><div class="answer">✅ Answer: B -- Causal mask enables parallel training: all tokens processed at once, but each sees only valid context.</div></details>
<details class="mcq"><summary>Without causal mask at inference?</summary><ol type="A"><li>Faster</li><li>Impossible -- future tokens do not exist yet. Model would need oracle access to future</li><li>Same</li><li>Better</li></ol><div class="answer">✅ Answer: B -- At inference, tokens generated one-by-one. The causal constraint is structural: future tokens simply do not exist.</div></details>
