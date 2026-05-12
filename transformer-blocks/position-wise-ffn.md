<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Transformer Blocks</div>
# Position-Wise FFN

## 1. Two-Layer MLP
Applied independently to each position:

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

## 2. Dimensions
$$d_{model} \xrightarrow{W_1} d_{ff} \xrightarrow{\text{GELU}} d_{ff} \xrightarrow{W_2} d_{model}$$

Where d_ff is typically 4 x d_model. For d_model=512: d_ff=2048.

## 3. Parameter Dominance
$$\text{Params}_{\text{FFN}} = 2 \cdot d_{model} \cdot d_{ff} = 8 \cdot d_{model}^2$$
$$\text{Params}_{\text{MHA}} = 4 \cdot d_{model}^2$$

The FFN has TWICE the parameters of the attention block. FFN = 2/3 of total parameters.

## 4. Activation Functions
| Activation | Formula | Used By |
|---|---|---|
| ReLU | max(0, x) | Original Transformer |
| GELU | x * Phi(x) | BERT, GPT |
| SwiGLU | x * sigmoid(beta*x) * W | Llama, PaLM |
| GeGLU | GELU(xW) * xV | Modern variants |

## 5. Why FFN?
Attention captures token-token interactions. FFN applies non-linear transformations per token. Together they create rich representations: attention for context, FFN for transformation.
## 📝 Self-Assessment
<details class="mcq"><summary>FFN dimensions?</summary><ol type="A"><li>d_model -> d_model</li><li>d_model -> d_ff -> d_model. Expand to 4x (d_ff=2048 for d_model=512), then project back</li><li>d_model -> d_model/4</li><li>Constant</li></ol><div class="answer">✅ Answer: B -- The bottleneck structure: expand, apply non-linearity, compress. 4x expansion is standard.</div></details>
<details class="mcq"><summary>FFN vs attention parameter ratio?</summary><ol type="A"><li>Same</li><li>FFN has 2x attention params: 8*d^2 vs 4*d^2. FFN is 2/3 of total transformer parameters</li><li>Attention larger</li><li>FFN smaller</li></ol><div class="answer">✅ Answer: B -- W1,W2: 2 * d * d_ff = 8d^2. Wq,Wk,Wv,Wo: 4d^2. FFN dominates parameter count.</div></details>
<details class="mcq"><summary>GELU activation?</summary><ol type="A"><li>ReLU variant</li><li>Gaussian Error Linear Unit: x * Phi(x) where Phi is standard normal CDF. Smoother than ReLU</li><li>Same as ReLU</li><li>Sigmoid</li></ol><div class="answer">✅ Answer: B -- GELU = x * P(X <= x) for X ~ N(0,1). Smooth, non-monotonic, better gradients than ReLU. BERT/GPT default.</div></details>
<details class="mcq"><summary>What is SwiGLU?</summary><ol type="A"><li>ReLU variant</li><li>Swish-Gated Linear Unit: (xW * sigmoid(xV)) * W_out. Gated activation with learnable gate. Used in Llama</li><li>Same as GELU</li><li>Older activation</li></ol><div class="answer">✅ Answer: B -- SwiGLU adds a gating mechanism. xW1 is gated by sigmoid(xW2). Better quality, slightly more parameters.</div></details>
<details class="mcq"><summary>Attention captures __, FFN captures __?</summary><ol type="A"><li>Both context</li><li>Attention: token-token interactions (context). FFN: per-token non-linear transformations (feature extraction)</li><li>Both features</li><li>Neither</li></ol><div class="answer">✅ Answer: B -- Attention builds context-aware representations. FFN applies position-wise transformations. Complementary roles.</div></details>
