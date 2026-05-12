<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Training</div>
# Teacher Forcing

## 1. Training with Ground Truth
During training, the model receives the CORRECT previous token (from training data), not its own prediction:

$$y_t \sim P(y_t | y_1^*, ..., y_{t-1}^*, \mathbf{x})$$

## 2. Why Teacher Forcing?
Without it, early mistakes cascade -- a wrong prediction at step 1 makes step 2 harder, which makes step 3 even harder. Teacher forcing provides stable training signal.

## 3. Parallel Training
With teacher forcing, all positions can be trained simultaneously:

```python
# All timesteps in one forward pass (with causal mask)
logits = model(target_sequence, encoder_output)
loss = cross_entropy(logits[:, :-1], target_sequence[:, 1:])
```

The causal mask ensures each position only sees previous tokens despite parallel computation.

## 4. Exposure Bias
Downside: train-inference mismatch. At training, model sees perfect history. At inference, it sees its own (potentially flawed) output. Scheduled sampling (mix of teacher forcing and self-feeding) partially addresses this.
## 📝 Self-Assessment
<details class="mcq"><summary>What is teacher forcing?</summary><ol type="A"><li>Student learns alone</li><li>Training with ground-truth previous tokens instead of model own predictions</li><li>Forced learning</li><li>No training</li></ol><div class="answer">✅ Answer: B -- At step t, input is the correct y*_{t-1} from training data, not the model predicted y_{t-1}.</div></details>
<details class="mcq"><summary>Why use teacher forcing?</summary><ol type="A"><li>Faster</li><li>Prevents error cascade -- early mistakes would compound if model fed its own erroneous predictions</li><li>Required</li><li>Memory</li></ol><div class="answer">✅ Answer: B -- Without teacher forcing, a wrong prediction at step 1 degrades all subsequent steps. Training becomes unstable.</div></details>
<details class="mcq"><summary>Parallel training with teacher forcing?</summary><ol type="A"><li>Not possible</li><li>All positions trained simultaneously with causal mask -- each position sees correct prefix despite parallel computation</li><li>Sequential only</li><li>Only GPT</li></ol><div class="answer">✅ Answer: B -- Causal mask + teacher forcing = parallel training. Each position sees its correct history without sequential generation.</div></details>
<details class="mcq"><summary>What is exposure bias?</summary><ol type="A"><li>Too much data</li><li>Train-inference mismatch -- model trained on perfect history but inference uses its own (imperfect) output</li><li>Overfitting</li><li>Underfitting</li></ol><div class="answer">✅ Answer: B -- At training, input history is always correct. At inference, errors accumulate. Scheduled sampling partially addresses this.</div></details>
<details class="mcq"><summary>Scheduled sampling?</summary><ol type="A"><li>Always teacher forcing</li><li>Gradually transition from teacher forcing to self-feeding during training -- reduces exposure bias</li><li>No training</li><li>Random</li></ol><div class="answer">✅ Answer: B -- Start with teacher forcing; over training, increasingly use model own predictions. Bridges train-inference gap.</div></details>
