<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Training</div>
# Sequence Length Limitations

## 1. The Practical Limit
Most Transformers are trained with a fixed maximum sequence length:

| Model | Max Context |
|---|---|
| Original Transformer | 512 |
| BERT | 512 |
| GPT-3 | 2048 |
| GPT-4 | 8192-32768 |
| Llama-2 | 4096 |
| Claude-3 | 200K |
| Gemini 1.5 | 1M (claimed) |

## 2. Why Limits?
1. O(n^2) memory: GPU VRAM limits
2. O(n^2) compute: training time
3. Position encoding: learned PEs capped at max length
4. Training data: most text chunks are short

## 3. Position Interpolation
Extend context beyond training length:
$$PE(pos) = PE\left(pos \cdot \frac{L_{train}}{L_{target}}\right)$$

Linearly interpolating positions enables 2-4x context extension with minimal fine-tuning.

## 4. Memory as the Hard Limit
At inference, KV cache scales as O(n):
$$M_{KV} = 2 \cdot L \cdot h \cdot d_h \cdot n \cdot 2 \text{ bytes}$$
For 70B model at n=128K: ~320 GB KV cache alone. This is the ultimate constraint.
## 📝 Self-Assessment
<details class="mcq"><summary>Original Transformer max context?</summary><ol type="A"><li>2048</li><li>512 tokens. Early Transformers were limited by O(n^2) memory and compute</li><li>4096</li><li>1024</li></ol><div class="answer">✅ Answer: B -- The original paper used n=512. This was practical for 2017 GPU memory, before FlashAttention.</div></details>
<details class="mcq"><summary>Why are Transformer contexts limited?</summary><ol type="A"><li>No reason</li><li>O(n^2) compute and memory, position encoding caps, training data chunk sizes -- multiple constraints</li><li>Only memory</li><li>Only compute</li></ol><div class="answer">✅ Answer: B -- Multiple factors: quadratic complexity, learned PE limits, and typical training data chunk sizes.</div></details>
<details class="mcq"><summary>Position interpolation?</summary><ol type="A"><li>Adding positions</li><li>pos_new = pos * (L_train/L_target). Linearly map target positions into trained range</li><li>Removing positions</li><li>Random positions</li></ol><div class="answer">✅ Answer: B -- Interpolation stretches position encodings. A model trained on 2K can handle 8K with position interpolation + fine-tuning.</div></details>
<details class="mcq"><summary>KV cache for 70B model at n=128K?</summary><ol type="A"><li>~10 GB</li><li>~320 GB -- 2 x 80 layers x 64 heads x 128 x 128K x 2 bytes. KV cache dwarfs model weights</li><li>~80 GB</li><li>~1 TB</li></ol><div class="answer">✅ Answer: B -- KV cache = 2 x L x h x d_h x n x bytes_per_elem. For long contexts, KV cache is the primary memory consumer.</div></details>
<details class="mcq"><summary>Which model claims 1M token context?</summary><ol type="A"><li>GPT-4</li><li>Gemini 1.5 -- Google claims 1M token context window, though effective utilization may vary</li><li>Llama</li><li>Claude</li></ol><div class="answer">✅ Answer: B -- Gemini 1.5 Pro pushes the context frontier. 1M tokens = ~750K words, roughly the length of the entire Harry Potter series.</div></details>
