<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Training</div>
# Parallelization Advantages Over RNNs

## 1. The RNN Bottleneck
RNNs process tokens sequentially: h_t depends on h_{t-1}. This creates T sequential dependencies. GPU cannot parallelize across time steps.

## 2. Transformer Parallelism
Self-attention computes all positions simultaneously:
$$\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$$
All n^2 dot products computed in ONE matrix multiply.

## 3. Training Speed
| Model | Time per 1000 tokens | Sequential ops |
|---|---|---|
| LSTM (n=512) | ~500ms | 512 |
| Transformer (n=512) | ~50ms | O(1) |
| Transformer (n=2048) | ~200ms | O(1) |

## 4. Scaling Benefits
Transformer training time grows sublinearly with sequence length (bottlenecked by GPU memory bandwidth, not sequential compute). RNN training time grows linearly with sequence length. For long sequences, the gap is enormous.
## 📝 Self-Assessment
<details class="mcq"><summary>Transformer parallelism over RNNs?</summary><ol type="A"><li>None</li><li>All tokens processed simultaneously via matrix multiply vs RNN sequential processing</li><li>Slight</li><li>RNN parallel</li></ol><div class="answer">✅ Answer: B -- Self-attention O(1) sequential ops vs RNN O(n). Each token can attend to all others in one matrix multiply.</div></details>
<details class="mcq"><summary>Training speed: Transformer vs LSTM (n=512)?</summary><ol type="A"><li>Same</li><li>~10x faster -- Transformer ~50ms vs LSTM ~500ms. All positions in one forward pass</li><li>2x</li><li>100x</li></ol><div class="answer">✅ Answer: B -- LSTM requires 512 sequential RNN steps. Transformer computes all 512 positions in parallel with O(1) sequential ops.</div></details>
<details class="mcq"><summary>Transformer scaling with sequence length?</summary><ol type="A"><li>Linear</li><li>Sublinear -- bottleneck becomes GPU memory bandwidth, not sequential compute. Much better than RNN O(n)</li><li>Exponential</li><li>Same as RNN</li></ol><div class="answer">✅ Answer: B -- Attention cost = O(n^2) FLOPs but O(1) sequential steps. RNN cost = O(n) FLOPs AND O(n) sequential steps.</div></details>
<details class="mcq"><summary>GPU utilization: Transformer vs RNN?</summary><ol type="A"><li>RNN higher</li><li>Transformer higher -- massive parallelism enables full GPU utilization. RNN serializes, underutilizing GPU SMs</li><li>Same</li><li>Depends</li></ol><div class="answer">✅ Answer: B -- GPU tensor cores excel at batched matrix multiplies. Transformer attention = one big matmul = near-100% utilization.</div></details>
<details class="mcq"><summary>Key enabler of Transformer parallelism?</summary><ol type="A"><li>Deeper networks</li><li>Self-attention as matrix multiply -- QK^T computes all n^2 pairs in one operation. No sequential dependency</li><li>Residuals</li><li>LayerNorm</li></ol><div class="answer">✅ Answer: B -- The matrix formulation of attention is the key. QK^T is a single, highly parallelizable GPU operation.</div></details>
