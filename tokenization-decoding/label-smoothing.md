<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Tokenization</div>
# Label Smoothing

## 1. Softening the Target
Instead of one-hot targets, use a smoothed distribution:

$$P_{smooth}(y) = (1 - \epsilon) \cdot \mathbb{1}[y = y^*] + \frac{\epsilon}{|V|}$$

## 2. Why?
- Reduces overconfidence: model learns uncertainty
- Improves calibration: predicted probabilities match empirical frequencies
- Regularization: prevents model from assigning probability 1.0 to training tokens
- Better generalization: especially for small datasets

## 3. Effect on Loss
$$\mathcal{L}_{smooth} = -(1-\epsilon) \log P(y^*) - \frac{\epsilon}{|V|} \sum_{y} \log P(y)$$

First term: maximize correct token. Second term: keep other token probabilities non-zero, preventing overconfidence.

## 4. Typical Epsilon
0.1 is standard for most Transformer training. Larger for smaller datasets; smaller or zero for very large datasets where overfitting is not a concern.
## 📝 Self-Assessment
<details class="mcq"><summary>Label smoothing?</summary><ol type="A"><li>Making labels longer</li><li>Replacing one-hot targets with smoothed distribution -- target token gets (1-e), others get e/|V|</li><li>Removing labels</li><li>Data augmentation</li></ol><div class="answer">✅ Answer: B -- Instead of P(target)=1.0, gives P(target)=0.9, P(others)=0.1/|V|. Softens the training signal.</div></details>
<details class="mcq"><summary>Why use label smoothing?</summary><ol type="A"><li>Speed</li><li>Reduces overconfidence -- model learns to be less certain, improving calibration and generalization</li><li>Memory</li><li>Required</li></ol><div class="answer">✅ Answer: B -- Overconfident models assign probability >0.99 to predictions. Label smoothing encourages more calibrated probabilities.</div></details>
<details class="mcq"><summary>Typical epsilon value?</summary><ol type="A"><li>0.5</li><li>0.1 -- 10% probability mass distributed across other tokens. Standard for most Transformer training</li><li>0.01</li><li>0.9</li></ol><div class="answer">✅ Answer: B -- epsilon=0.1 is the most common value. 10% uncertainty distributed across the vocabulary. Standard in Transformer papers.</div></details>
<details class="mcq"><summary>Label smoothing loss decomposition?</summary><ol type="A"><li>Cross-entropy only</li><li>-(1-e)*log(P(correct)) - e/|V|*sum(log(P)). Maximize correct token while preventing others from going to zero</li><li>MSE</li><li>KL only</li></ol><div class="answer">✅ Answer: B -- Two terms: maximize target probability (weighted by 1-e), and keep other tokens non-zero (e/|V| term).</div></details>
<details class="mcq"><summary>Label smoothing and calibration?</summary><ol type="A"><li>No effect</li><li>Improves calibration -- model predicted probabilities better match actual correctness rates</li><li>Worsens calibration</li><li>Same</li></ol><div class="answer">✅ Answer: B -- Without smoothing, models are overconfident (P=0.99 but 70% correct). Smoothing aligns confidence with accuracy.</div></details>
