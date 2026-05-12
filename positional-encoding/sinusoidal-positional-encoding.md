<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Positional Encoding</div>
# Sinusoidal Positional Encoding

## 1. The Formula
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

pos = position (0, 1, 2, ...), i = dimension index (0 to d/2-1).

## 2. Multi-Scale Representation
Lower dimensions (small i) oscillate rapidly (high frequency) -- capture local position. Higher dimensions oscillate slowly (low frequency) -- capture global position.

$$\lambda_i = 2\pi \cdot 10000^{2i/d}$$

Wave lengths range from 2pi (~6.3) to 2pi*10000 (~62832).

## 3. Key Property: Relative Position
$$PE_{pos+k} = f(PE_{pos}) \text{ via learned linear transformation}$$
The sinusoid at position pos+k can be expressed as a linear function of the sinusoid at pos. This enables learning relative position relationships.

## 4. Extrapolation
Sinusoidal PEs generalize to any position -- the formula works for pos beyond training length. This is their key advantage over learned embeddings.

## 5. Limitations
Fixed, not learned -- cannot adapt to task-specific position patterns. Modern models often prefer learned PEs or RoPE for better task adaptation.
## 📝 Self-Assessment
<details class="mcq"><summary>Sinusoidal PE formula variables?</summary><ol type="A"><li>Random</li><li>pos = token position, i = dimension index. Even dims use sin, odd dims use cos, frequency controlled by 10000^(2i/d)</li><li>pos only</li><li>i only</li></ol><div class="answer">✅ Answer: B -- Pair of sin/cos per dimension pair. Frequency decreases with i, giving multi-scale position encoding.</div></details>
<details class="mcq"><summary>Why multiple frequencies?</summary><ol type="A"><li>Arbitrary</li><li>Different dimensions encode position at different scales -- some capture local (high freq), others global (low freq)</li><li>For speed</li><li>For memory</li></ol><div class="answer">✅ Answer: B -- Low i = high frequency (local). High i = low frequency (global). Multi-scale enables both fine and coarse position info.</div></details>
<details class="mcq"><summary>Sinusoidal PE key advantage?</summary><ol type="A"><li>Learned</li><li>Extrapolation -- formula works for any position, even beyond training length. No retraining for longer sequences</li><li>Faster</li><li>Smaller</li></ol><div class="answer">✅ Answer: B -- Unlike learned embeddings (max length fixed), sinusoidal PEs can encode arbitrarily long sequences.</div></details>
<details class="mcq"><summary>Relative position property?</summary><ol type="A"><li>Not present</li><li>PE(pos+k) can be expressed as linear function of PE(pos) -- enables learning relative position relationships</li><li>Only absolute</li><li>Random</li></ol><div class="answer">✅ Answer: B -- The linear relationship property allows the model to learn that position 5 is 3 away from position 2.</div></details>
<details class="mcq"><summary>PE added to embeddings magnitude?</summary><ol type="A"><li>Large</li><li>Same order of magnitude (~1). Kept comparable to token embeddings so neither dominates</li><li>100x larger</li><li>Negligible</li></ol><div class="answer">✅ Answer: B -- Sin/cos range [-1,1], same scale as learned embeddings (typically ~0.02 std). Balanced contribution.</div></details>
