<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Optimization</div>
# Optimization Challenges

## 1. Training Instability
Key challenges and solutions:

| Challenge | Symptom | Solution |
|---|---|---|
| Large initial gradients | Loss spikes, divergence | LR warmup |
| Gradient explosion | NaN loss | Gradient clipping |
| Attention collapse | Uniform attention | Better init, warmup |
| Overfitting | Train << Val loss | Dropout, data augmentation |
| Slow convergence | Plateauing loss | Better LR schedule |

## 2. Learning Rate Sensitivity
The optimal LR is narrow (often 10x smaller than for CNNs). Too high: divergence. Too low: slow convergence. Warmup is essential for the first ~4000 steps.

## 3. Mixed Precision Training
```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    loss = model(input)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
FP16 half memory and bandwidth with loss scaling to prevent gradient underflow.
## 📝 Self-Assessment
<details class="mcq"><summary>Most common Transformer training issue?</summary><ol type="A"><li>Memory</li><li>Training instability -- loss spikes/divergence especially early. Solved by LR warmup and gradient clipping</li><li>Speed</li><li>Accuracy</li></ol><div class="answer">✅ Answer: B -- Transformers are sensitive. Without warmup, early gradients are too large and cause divergence.</div></details>
<details class="mcq"><summary>LR warmup duration?</summary><ol type="A"><li>100 steps</li><li>~4000 steps -- linear increase from 0 to peak LR. Critical for training stability</li><li>1 step</li><li>100K steps</li></ol><div class="answer">✅ Answer: B -- The first few thousand steps are most unstable. Warmup allows Adam moments to stabilize before full LR.</div></details>
<details class="mcq"><summary>Gradient clipping threshold?</summary><ol type="A"><li>None</li><li>Typically 1.0 -- clips gradient norm to prevent explosions</li><li>10.0</li><li>100.0</li></ol><div class="answer">✅ Answer: B -- Clip grad norm to 1.0 is standard. Prevents single batches with pathological gradients from destroying training.</div></details>
<details class="mcq"><summary>Why mixed precision training?</summary><ol type="A"><li>Accuracy</li><li>FP16 half the memory and bandwidth, ~2x speedup on tensor cores. Loss scaling prevents FP16 underflow</li><li>Required</li><li>No benefit</li></ol><div class="answer">✅ Answer: B -- FP16 is faster and uses less memory. Loss scaling keeps small gradients representable in FP16.</div></details>
<details class="mcq"><summary>FP16 loss scaling?</summary><ol type="A"><li>Not needed</li><li>Multiplies loss by large factor before backward -- prevents small gradients from underflowing to zero in FP16</li><li>Division</li><li>Dynamic only</li></ol><div class="answer">✅ Answer: B -- FP16 minimum is ~6e-8. Small gradients would underflow. Scaling moves them into representable range.</div></details>
