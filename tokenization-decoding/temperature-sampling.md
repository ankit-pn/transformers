<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Tokenization</div>
# Temperature Sampling

## 1. Controlling Randomness
$$P_T(y_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

## 2. Temperature Effects
| T | Behavior | Use Case |
|---|---|---|
| 0.0 | Greedy (deterministic) | Factual QA |
| 0.3-0.5 | Conservative | Code, structured output |
| 0.7-0.8 | Balanced (default) | General chat |
| 0.9-1.2 | Creative | Storytelling, brainstorming |
| >1.5 | Chaotic | Degenerate, rarely useful |

## 3. Mathematical Effect
Temperature scales the logits before softmax:
$$z_i' = z_i / T$$

T < 1: amplifies differences (peaked distribution). T > 1: flattens differences (uniform distribution). T = 1: raw model distribution.

## 4. Entropy Control
$$H(P_T) = -\sum P_T(y_i) \log P_T(y_i)$$

Higher T -> higher entropy -> more diverse output. But too high -> nonsensical output.
## 📝 Self-Assessment
<details class="mcq"><summary>Temperature T=0.7 effect?</summary><ol type="A"><li>Makes uniform</li><li>Moderate sharpening -- high-prob tokens get slightly higher, low-prob slightly lower. Balanced diversity</li><li>Makes greedy</li><li>No effect</li></ol><div class="answer">✅ Answer: B -- T<1 sharpens the distribution. 0.7 is a sweet spot: enough focus for coherence, enough spread for diversity.</div></details>
<details class="mcq"><summary>T -> 0 behavior?</summary><ol type="A"><li>Uniform</li><li>Approaches greedy/argmax. The highest logit dominates entirely. Deterministic output</li><li>Random</li><li>Same as T=1</li></ol><div class="answer">✅ Answer: B -- As T approaches 0, the softmax becomes more peaked. At T=0 exactly, it would be one-hot at the argmax position.</div></details>
<details class="mcq"><summary>T=2.0 effect?</summary><ol type="A"><li>Sharpens</li><li>Flattens -- makes all tokens more equally probable. Output becomes increasingly random and incoherent</li><li>No effect</li><li>Same as T=1</li></ol><div class="answer">✅ Answer: B -- T>1 spreads probability mass across more tokens. At T=2, the distribution is significantly flatter, producing diverse but potentially incoherent output.</div></details>
<details class="mcq"><summary>Temperature vs top-k/top-p?</summary><ol type="A"><li>Same thing</li><li>Temperature controls distribution shape (continuous). Top-k/top-p truncate the distribution (discrete). Often combined</li><li>Only one needed</li><li>Incompatible</li></ol><div class="answer">✅ Answer: B -- Temperature adjusts the probability landscape; top-k/p limit which tokens can be sampled. Complementary; typically used together.</div></details>
<details class="mcq"><summary>Entropy and temperature relationship?</summary><ol type="A"><li>Inverse</li><li>Higher T -> higher entropy -> more diverse output. Entropy measures uncertainty of the distribution</li><li>No relationship</li><li>Constant</li></ol><div class="answer">✅ Answer: B -- Temperature directly controls entropy. T<1 decreases entropy (more certain); T>1 increases entropy (more uncertain).</div></details>
