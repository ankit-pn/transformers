<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Transformer Blocks</div>
# Encoder Block Architecture

## 1. Structure
```
Input -> Multi-Head Self-Attention -> Add & Norm -> FeedForward -> Add & Norm -> Output
```

## 2. Bidirectional Self-Attention
The encoder uses FULL (non-causal) self-attention. Each token attends to ALL other tokens. No masking (except padding).

## 3. Two Sublayers
1. Multi-Head Self-Attention with residual + LN
2. Position-wise FFN with residual + LN

## 4. PyTorch Implementation
```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff):
        self.attention = MultiHeadAttention(d_model, h)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Self-attention sublayer
        attn_out = self.attention(x, x, x, mask)
        x = self.ln1(x + attn_out)
        # FFN sublayer
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x
```

## 📝 Self-Assessment
<details class="mcq"><summary>Encoder self-attention type?</summary><ol type="A"><li>Causal</li><li>Full/bidirectional -- each token attends to ALL other tokens. No future masking</li><li>Local</li><li>Random</li></ol><div class="answer">✅ Answer: B -- Encoder uses unmasked self-attention. Every token sees every other token for full context understanding.</div></details>
<details class="mcq"><summary>How many sublayers in encoder block?</summary><ol type="A"><li>1</li><li>2 -- multi-head self-attention + position-wise FFN. Each with residual + LayerNorm</li><li>3</li><li>4</li></ol><div class="answer">✅ Answer: B -- Two sublayers: attention and FFN. Both have residual connections and layer normalization.</div></details>
<details class="mcq"><summary>Encoder block input/output shape?</summary><ol type="A"><li>Different</li><li>Same -- [batch, n, d_model]. The encoder preserves sequence length and dimension</li><li>Halved</li><li>Doubled</li></ol><div class="answer">✅ Answer: B -- Encoder maintains sequence shape throughout. Same n and d_model from input to output of each block.</div></details>
<details class="mcq"><summary>What does the encoder produce?</summary><ol type="A"><li>Class label</li><li>Contextualized representations for each token -- a rich, bidirectional understanding of the input</li><li>Summary vector</li><li>Translation</li></ol><div class="answer">✅ Answer: B -- Encoder output is per-token hidden states that encode both the token identity and its full context.</div></details>
<details class="mcq"><summary>Purpose of the encoder?</summary><ol type="A"><li>Text generation</li><li>Understand the input sequence -- produce context-aware representations used by the decoder for generation</li><li>Classification only</li><li>Speed</li></ol><div class="answer">✅ Answer: B -- Encoder builds rich bidirectional representations. Decoder uses these via cross-attention to generate output.</div></details>
