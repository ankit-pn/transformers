<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"><style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem}th{background:#f6f8fa}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Attention Mechanics</div>
# Query Key Value Intuition

## 1. Database Analogy
| Component | Database | Attention |
|---|---|---|
| Query (Q) | What you seek | Current state s_t |
| Key (K) | Index of items | Encoder states h_i |
| Value (V) | Actual content | Encoder states (transformed) |
| Weight | Relevance score | softmax(score(Q, K_i)) |
| Output | Retrieved value | sum(alpha_i * V_i) |

## 2. Process
1. Match: e_i = score(q, k_i)
2. Normalize: alpha_i = softmax(e_i)
3. Aggregate: output = sum(alpha_i * v_i)

## 3. Intuitions
Query: "what information do I need?" Key: "what information do I have?" Value: "what do I contribute?" Weight = similarity(query, key) = relevance.

## 4. Separate K and V
Basic: key=value=encoder state. Separate enables asymmetric reps: Keys for matching, Values for content. Enables KV cache compression (MQA/GQA).
## 📝 Self-Assessment
<details class="mcq"><summary>Database analogy for Q,K,V?</summary><ol type="A"><li>Hash table</li><li>Query=what you seek, Key=item index, Value=item content. Soft key-value lookup</li><li>SQL query</li><li>Sequential scan</li></ol><div class="answer">✅ Answer: B -- Like retrieving from KV store, attention matches queries against keys, retrieves values with soft weights.</div></details>
<details class="mcq"><summary>What does Query encode?</summary><ol type="A"><li>Stored data</li><li>What information is needed -- current state defining the information requirement</li><li>Retrieved result</li><li>Index</li></ol><div class="answer">✅ Answer: B -- Query encodes the information need. Each token asks which other tokens are relevant to it.</div></details>
<details class="mcq"><summary>Why separate Key from Value?</summary><ol type="A"><li>Always same</li><li>Enables asymmetric representations. Keys for efficient matching; Values for content. Critical for MQA/GQA</li><li>Reduce params</li><li>Parallel</li></ol><div class="answer">✅ Answer: B -- Keys can be compressed for matching. Values stay full precision for aggregation. Enables memory-efficient inference.</div></details>
<details class="mcq"><summary>What if Query equals Key?</summary><ol type="A"><li>No attention</li><li>Max dot product -- token attends most strongly to itself, creating diagonal in attention matrix</li><li>Random</li><li>Uniform</li></ol><div class="answer">✅ Answer: B -- Self-attention: Q=K. Diagonal (i,i) has highest score since Q_i perfectly matches K_i.</div></details>
<details class="mcq"><summary>In cross-attention, what are Q, K, V?</summary><ol type="A"><li>All from encoder</li><li>Q from decoder, K and V from encoder. Decoder queries encoder for source information</li><li>All from decoder</li><li>Random</li></ol><div class="answer">✅ Answer: B -- Cross-attention: decoder queries encoder output. Q=decoder state, K,V=encoder states.</div></details>
