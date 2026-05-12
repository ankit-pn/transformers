<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Analysis</div>
# Attention Is All You Need

## 1. The Revolutionary Paper
Vaswani et al. (2017), "Attention Is All You Need", introduced the Transformer architecture. Rejected recurrence and convolution entirely in favor of pure attention.

## 2. Key Contributions
- Self-attention as the core mechanism
- Multi-head attention for parallel diverse representations
- Positional encoding to inject sequence order
- Scaled dot-product attention preventing softmax saturation
- Residual connections + LayerNorm enabling deep training

## 3. Architecture (Base Model)
| Component | Specification |
|---|---|
| Encoder layers | 6 |
| Decoder layers | 6 |
| d_model | 512 |
| Attention heads | 8 |
| d_ff | 2048 |
| Total params | 65M |

## 4. WMT 2014 Results
| Task | BLEU Score |
|---|---|
| EN->DE (base) | 27.3 (SOTA) |
| EN->DE (big) | 28.4 (SOTA) |
| EN->FR (big) | 41.0 (SOTA) |

Surpassed all existing models including ensembles while being faster to train.

## 5. Legacy
The paper has >100,000 citations. It spawned BERT, GPT, T5, and the entire modern NLP landscape. "Attention Is All You Need" is arguably the most influential ML paper of the decade.
## 📝 Self-Assessment
<details class="mcq"><summary>What did Attention Is All You Need replace?</summary><ol type="A"><li>CNN</li><li>Recurrence and convolution -- Transformer uses ONLY attention, no RNN or CNN components</li><li>Attention</li><li>Linear layers</li></ol><div class="answer">✅ Answer: B -- The paper title is literal: the architecture uses attention exclusively, rejecting recurrence and convolution entirely.</div></details>
<details class="mcq"><summary>Original Transformer base model params?</summary><ol type="A"><li>65M</li><li>65M -- 6 encoder + 6 decoder layers, d_model=512, h=8. Big model: 213M</li><li>1B</li><li>10M</li></ol><div class="answer">✅ Answer: B -- 65M parameters for the base model. Modest by today standards but revolutionary in 2017.</div></details>
<details class="mcq"><summary>Positional encoding purpose in the paper?</summary><ol type="A"><li>Speed</li><li>Inject sequence order into permutation-invariant self-attention. Sinusoidal encoding chosen</li><li>Memory</li><li>Accuracy</li></ol><div class="answer">✅ Answer: B -- Self-attention is permutation-invariant. Positional encoding adds order information, critical for language.</div></details>
<details class="mcq"><summary>WMT 2014 EN-DE BLEU score (base)?</summary><ol type="A"><li>20.0</li><li>27.3 -- state of the art, surpassing all existing recurrent and convolutional models</li><li>35.0</li><li>15.0</li></ol><div class="answer">✅ Answer: B -- 27.3 BLEU was SOTA in 2017, achieved with a purely attention-based model trained in 3.5 days on 8 GPUs.</div></details>
<details class="mcq"><summary>How many citations for the paper?</summary><ol type="A"><li>~1,000</li><li>>100,000 -- it is one of the most cited papers in ML history. Foundation of modern NLP</li><li>~10,000</li><li>~50,000</li></ol><div class="answer">✅ Answer: B -- The Transformer paper is among the most-cited scientific papers of all time. It launched the modern LLM era.</div></details>
