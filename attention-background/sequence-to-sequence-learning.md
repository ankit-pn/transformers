<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<style>body{min-width:320px;max-width:860px;margin:0 auto;padding:1rem;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;color:#1a1a1a}h1{font-size:1.6rem;border-bottom:2px solid #e36209;padding-bottom:.5rem}h2{font-size:1.25rem;margin-top:2rem;color:#e36209}h3{font-size:1.05rem;margin-top:1.5rem}pre{background:#f6f8fa;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.85rem}code{background:#f0f0f0;padding:.15em .3em;border-radius:3px;font-size:.9em}pre code{background:0;padding:0}.breadcrumb{font-size:.85rem;margin-bottom:1rem;color:#666}.breadcrumb a{color:#e36209;text-decoration:none}.mcq{background:#fff8f0;border-left:4px solid #e36209;padding:1rem;margin:1.5rem 0;border-radius:0 8px 8px 0}.mcq summary{font-weight:bold;cursor:pointer;color:#e36209;font-size:1rem}.mcq ol{margin-top:.5rem;padding-left:1.5rem}.answer{background:#e6ffed;padding:.5rem 1rem;margin-top:.5rem;border-radius:4px;font-weight:bold}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d0d7de;padding:.5rem .75rem;text-align:left}th{background:#f6f8fa}blockquote{border-left:3px solid #d0d7de;padding-left:1rem;color:#555;margin:1rem 0}@media(max-width:600px){body{padding:.75rem}h1{font-size:1.35rem}}</style><div class="breadcrumb">← <a href="../index.md">Transformers Knowledge Base</a> / Seq2Seq & RNN</div>
# Sequence-to-Sequence Learning

## 1. The Seq2Seq Paradigm

Sequence-to-sequence (seq2seq) models map an input sequence $\mathbf{x} = (x_1, ..., x_T)$ to an output sequence $\mathbf{y} = (y_1, ..., y_{T'})$ of possibly different length:

$$\mathbf{y} = f(\mathbf{x}; \theta)$$

This is the foundation for machine translation (English → French), summarization (article → summary), and dialogue generation.

## 2. Encoder-Decoder Architecture

The classic seq2seq model (Sutskever et al., 2014; Cho et al., 2014) has two components:

**Encoder**: Processes the input sequence into a fixed-length context vector $\mathbf{c}$:
$$\mathbf{h}_t = \text{RNN}(x_t, \mathbf{h}_{t-1}), \quad \mathbf{c} = \mathbf{h}_T$$

**Decoder**: Generates output autoregressively from the context vector:
$$\mathbf{s}_t = \text{RNN}(y_{t-1}, \mathbf{s}_{t-1}, \mathbf{c}), \quad P(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_o \mathbf{s}_t)$$

## 3. Mathematical Formulation

The full conditional probability:
$$P(\mathbf{y} | \mathbf{x}) = \prod_{t=1}^{T'} P(y_t | y_1, ..., y_{t-1}, \mathbf{c})$$

## 4. Applications

- Machine Translation (English → German)
- Text Summarization (article → headline)
- Speech Recognition (audio → transcript)
- Image Captioning (CNN encoder + RNN decoder)
- Conversational Models (utterance → response)

## 5. Training

Trained end-to-end with teacher forcing: ground-truth previous token is fed as input during training rather than the model's own prediction.

$$\mathcal{L} = -\sum_{t=1}^{T'} \log P(y_t^* | y_{<t}^*, \mathbf{x})$$

## 📝 Self-Assessment
<details class="mcq"><summary>What is the core idea of seq2seq?</summary><ol type="A"><li>Map fixed-length vectors to fixed-length vectors</li><li>Map variable-length input sequences to variable-length output sequences using encoder-decoder architecture</li><li>Classify sequences into categories</li><li>Generate random sequences</li></ol><div class="answer">✅ Answer: B — Seq2seq handles input/output sequences of DIFFERENT lengths. The encoder compresses input; decoder generates output autoregressively.</div></details>
<details class="mcq"><summary>What is the context vector c?</summary><ol type="A"><li>Random initialization</li><li>The final hidden state of the encoder RNN, encoding the entire input sequence into a fixed-length vector</li><li>The first decoder state</li><li>A trainable parameter</li></ol><div class="answer">✅ Answer: B — The context vector c = h_T summarizes the entire input. It is the only information passed from encoder to decoder in basic seq2seq.</div></details>
<details class="mcq"><summary>How is seq2seq trained?</summary><ol type="A"><li>Reinforcement learning only</li><li>Teacher forcing — ground-truth previous token fed as input, cross-entropy loss over output tokens</li><li>Unsupervised learning</li><li>Adversarial training</li></ol><div class="answer">✅ Answer: B — Teacher forcing feeds the correct previous token during training, avoiding error propagation. Loss is cross-entropy over target tokens.</div></details>
<details class="mcq"><summary>Which is NOT a seq2seq application?</summary><ol type="A"><li>Machine translation</li><li>Image classification (fixed output, not a sequence)</li><li>Text summarization</li><li>Speech recognition</li></ol><div class="answer">✅ Answer: B — Image classification has a fixed output (class label), not a variable-length sequence. Seq2seq requires sequence output.</div></details>
<details class="mcq"><summary>Who proposed the classic seq2seq model?</summary><ol type="A"><li>Vaswani et al.</li><li>Sutskever et al. (2014) and Cho et al. (2014) — two independent papers introducing encoder-decoder with RNNs</li><li>Hochreiter & Schmidhuber</li><li>Bengio et al.</li></ol><div class="answer">✅ Answer: B — The seq2seq paradigm was independently proposed by Sutskever (Google) and Cho (Montreal) in 2014, using LSTM and GRU respectively.</div></details>
