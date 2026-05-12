<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Optimization</div>
# Encoder-Only vs Decoder-Only vs Encoder-Decoder

## 1. Three Architectures
| Type | Examples | Attention | Use Case |
|---|---|---|---|
| Encoder-Only | BERT, RoBERTa | Bidirectional self | Understanding |
| Decoder-Only | GPT, Llama | Causal self | Generation |
| Encoder-Decoder | T5, BART | Bidirectional + cross | Translation, summarization |

## 2. Encoder-Only
Each position sees full context. Good for understanding, cannot autoregressively generate text.

## 3. Decoder-Only
Each position sees prefix only. Natural for generation. Simpler architecture (no cross-attention).

## 4. Encoder-Decoder
Encoder: full source encoding. Decoder: autoregressive generation with cross-attention to source. Most flexible, most complex.

## 5. Modern Convergence
Architecture choice is increasingly about efficiency and scale. Decoder-only models handle understanding via prompting. All architectures can perform most tasks given sufficient scale.
## 📝 Self-Assessment
<details class="mcq"><summary>BERT architecture type?</summary><ol type="A"><li>Decoder-only</li><li>Encoder-only -- bidirectional self-attention for understanding tasks</li><li>Encoder-decoder</li><li>All three</li></ol><div class="answer">✅ Answer: B -- BERT uses only the Transformer encoder. Understands text but does not generate autoregressively.</div></details>
<details class="mcq"><summary>GPT architecture type?</summary><ol type="A"><li>Encoder-only</li><li>Decoder-only -- causal self-attention for autoregressive text generation</li><li>Encoder-decoder</li><li>All three</li></ol><div class="answer">✅ Answer: B -- GPT uses only the Transformer decoder. Generates text token by token.</div></details>
<details class="mcq"><summary>Decoder-only models and cross-attention?</summary><ol type="A"><li>Must have</li><li>No cross-attention -- decoder-only models have no encoder to attend to</li><li>Required</li><li>Optional</li></ol><div class="answer">✅ Answer: B -- Decoder-only omits the encoder entirely. No cross-attention sublayer in any block.</div></details>
<details class="mcq"><summary>Best for translation?</summary><ol type="A"><li>Encoder-only</li><li>Encoder-decoder -- encodes source language, generates target with cross-attention alignment</li><li>Decoder-only</li><li>Any</li></ol><div class="answer">✅ Answer: B -- Translation needs to understand source AND generate target. Encoder-decoder has specialized components for both.</div></details>
<details class="mcq"><summary>Modern architecture convergence?</summary><ol type="A"><li>Diverging</li><li>Converging -- decoder-only models handle understanding via prompting. Scale matters more than architecture type</li><li>No change</li><li>Opposite trend</li></ol><div class="answer">✅ Answer: B -- GPT-4 (decoder-only) handles tasks once requiring encoder-only. Architecture matters less at scale.</div></details>
