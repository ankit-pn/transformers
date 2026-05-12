<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Tokenization</div>
# Training Pipeline

## 1. End-to-End Process
1. Collect massive text corpus
2. Train tokenizer (BPE/WordPiece/SentencePiece)
3. Preprocess: tokenize, pack sequences, create batches
4. Initialize model (random or from pretrained)
5. Train with Adam + cross-entropy loss + learning rate schedule
6. Evaluate perplexity on held-out data
7. Deploy for inference

## 2. Data Pipeline
```python
# Typical training data preparation
tokens = tokenizer.encode(text)  # Tokenize
chunks = chunk_tokens(tokens, max_seq_len)  # Split to context windows
batches = create_batches(chunks, batch_size)  # Batch
```

## 3. Learning Rate Schedule
```python
def lr_schedule(step, d_model, warmup_steps):
    arg1 = step ** (-0.5)
    arg2 = step * (warmup_steps ** (-1.5))
    return (d_model ** (-0.5)) * min(arg1, arg2)
```

## 4. Hardware Scale
| Model | GPUs | Training Time | Data |
|---|---|---|---|
| BERT-base | 16 TPUs | 4 days | 16 GB |
| GPT-3 175B | ~10K V100s | Months | 570 GB |
| Llama-2 70B | 2K A100s | ~1.7M GPU-hours | 2T tokens |

## 5. Key Hyperparameters
d_model, layers, heads, d_ff, learning rate, warmup steps, batch size (in tokens), dropout, weight decay, label smoothing.
## 📝 Self-Assessment
<details class="mcq"><summary>Transformer training data format?</summary><ol type="A"><li>Raw text</li><li>Tokenize -> chunk to max_seq_len -> batch. Text transformed to integer token IDs</li><li>Images</li><li>Audio</li></ol><div class="answer">✅ Answer: B -- Raw text is tokenized into integer sequences, chunked to context length, and batched for training.</div></details>
<details class="mcq"><summary>Learning rate warmup?</summary><ol type="A"><li>Constant LR</li><li>LR increases linearly from 0 to peak over warmup steps -- prevents early training instability</li><li>Decreasing</li><li>No schedule</li></ol><div class="answer">✅ Answer: B -- Transformer gradients are large and noisy early. Warmup prevents destructive early updates before Adam moments stabilize.</div></details>
<details class="mcq"><summary>GPT-3 training scale?</summary><ol type="A"><li>Single GPU</li><li>~10K V100 GPUs over months. Industrial-scale training</li><li>100 GPUs</li><li>1K GPUs</li></ol><div class="answer">✅ Answer: B -- GPT-3 175B required massive distributed training. One of the largest training runs at its time.</div></details>
<details class="mcq"><summary>Data preprocessing for training?</summary><ol type="A"><li>None needed</li><li>Tokenize, chunk to max_seq_len, pack multiple documents, create batches with padding/attention masks</li><li>Just tokenize</li><li>No prep</li></ol><div class="answer">✅ Answer: B -- Raw text must be tokenized, split into context windows, optionally packed for efficiency, and batched.</div></details>
<details class="mcq"><summary>Batch size in Transformer training?</summary><ol type="A"><li>In sequences</li><li>Typically measured in tokens (not sequences). 512-2048 tokens per batch for small; millions for large</li><li>In bytes</li><li>In words</li></ol><div class="answer">✅ Answer: B -- Token-based batching accounts for variable sequence length. Large models use batch sizes of millions of tokens.</div></details>
