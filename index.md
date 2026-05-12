# 🤖 Transformers & Attention — Knowledge Base

> A comprehensive, research-backed reference on Transformer architecture and attention mechanisms. From seq2seq foundations through modern efficient attention variants. Every topic includes mathematical derivations, code examples, and self-assessment MCQs.

<style>
  body { min-width: 320px; font-family: system-ui, -apple-system, sans-serif; }
  .toc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; margin: 1.5rem 0; }
  .toc-card { background: var(--bg-secondary, #f6f8fa); border-radius: 10px; padding: 1rem; border: 1px solid var(--border-color, #d0d7de); }
  .toc-card h3 { margin: 0 0 0.5rem 0; font-size: 1.05rem; }
  .toc-card ul { margin: 0; padding-left: 1.2rem; font-size: 0.88rem; }
  .toc-card a { text-decoration: none; }
  .toc-card a:hover { text-decoration: underline; }
  @media (max-width: 600px) { .toc-grid { grid-template-columns: 1fr; } }
</style>

<div class="toc-grid">

<div class="toc-card">
<h3>📜 Seq2Seq & RNN Background</h3>
<ul>
<li><a href="attention-background/sequence-to-sequence-learning.md">Sequence-to-Sequence Learning</a></li>
<li><a href="attention-background/rnn-encoder-decoder-limitations.md">RNN Encoder-Decoder Limitations</a></li>
<li><a href="attention-background/bottleneck-problem.md">Bottleneck Problem in Seq2Seq</a></li>
</ul>
</div>

<div class="toc-card">
<h3>🔍 Attention Fundamentals</h3>
<ul>
<li><a href="attention-mechanics/introduction-to-attention.md">Introduction to Attention</a></li>
<li><a href="attention-mechanics/alignment-scores.md">Alignment Scores</a></li>
<li><a href="attention-mechanics/query-key-value-intuition.md">Query, Key, Value Intuition</a></li>
<li><a href="attention-mechanics/query-key-similarity.md">Query-Key Similarity Computation</a></li>
<li><a href="attention-mechanics/dot-product-attention.md">Dot Product Attention</a></li>
<li><a href="attention-mechanics/scaled-dot-product-attention.md">Scaled Dot Product Attention</a></li>
<li><a href="attention-mechanics/softmax-normalization.md">Softmax Normalization in Attention</a></li>
<li><a href="attention-mechanics/weighted-sum-aggregation.md">Weighted Sum Aggregation</a></li>
</ul>
</div>

<div class="toc-card">
<h3>🎯 Attention Types & Masking</h3>
<ul>
<li><a href="attention-types/self-attention.md">Self-Attention</a></li>
<li><a href="attention-types/cross-attention.md">Cross-Attention (Encoder-Decoder)</a></li>
<li><a href="attention-types/causal-masking.md">Causal / Self-Regressive Masking</a></li>
<li><a href="attention-types/padding-masks.md">Padding Masks</a></li>
<li><a href="attention-types/attention-score-matrix.md">Attention Score Matrix</a></li>
<li><a href="attention-types/attention-weight-visualization.md">Attention Weight Visualization</a></li>
</ul>
</div>

<div class="toc-card">
<h3>🔀 Multi-Head Attention</h3>
<ul>
<li><a href="multi-head-attention/multi-head-attention.md">Multi-Head Attention</a></li>
<li><a href="multi-head-attention/head-splitting.md">Head Splitting</a></li>
<li><a href="multi-head-attention/parallel-attention-heads.md">Parallel Attention Heads</a></li>
<li><a href="multi-head-attention/head-concatenation.md">Head Concatenation</a></li>
<li><a href="multi-head-attention/projection-matrices.md">Linear Projection Matrices (Wq, Wk, Wv, Wo)</a></li>
</ul>
</div>

<div class="toc-card">
<h3>📍 Positional Encoding</h3>
<ul>
<li><a href="positional-encoding/positional-encoding-motivation.md">Positional Encoding Motivation</a></li>
<li><a href="positional-encoding/sinusoidal-positional-encoding.md">Sinusoidal Positional Encoding</a></li>
<li><a href="positional-encoding/learned-positional-embeddings.md">Learned Positional Embeddings</a></li>
</ul>
</div>

<div class="toc-card">
<h3>🧱 Transformer Blocks</h3>
<ul>
<li><a href="transformer-blocks/residual-connections.md">Residual Connections</a></li>
<li><a href="transformer-blocks/layer-normalization.md">Layer Normalization</a></li>
<li><a href="transformer-blocks/position-wise-ffn.md">Position-Wise FFN / MLP Block</a></li>
<li><a href="transformer-blocks/encoder-block.md">Encoder Block Architecture</a></li>
<li><a href="transformer-blocks/decoder-block.md">Decoder Block Architecture</a></li>
<li><a href="transformer-blocks/encoder-stack.md">Encoder Stack</a></li>
<li><a href="transformer-blocks/decoder-stack.md">Decoder Stack</a></li>
<li><a href="transformer-blocks/input-embeddings.md">Input Embeddings</a></li>
<li><a href="transformer-blocks/output-embeddings.md">Output Embeddings</a></li>
<li><a href="transformer-blocks/weight-tying.md">Weight Tying</a></li>
<li><a href="transformer-blocks/output-projection-softmax.md">Output Projection + Softmax Layer</a></li>
</ul>
</div>

<div class="toc-card">
<h3>🏋️ Transformer Training</h3>
<ul>
<li><a href="transformer-training/autoregressive-decoding.md">Autoregressive Decoding</a></li>
<li><a href="transformer-training/teacher-forcing.md">Teacher Forcing</a></li>
<li><a href="transformer-training/next-token-prediction.md">Next-Token Prediction</a></li>
<li><a href="transformer-training/training-objective.md">Training Objective (Cross Entropy Loss)</a></li>
<li><a href="transformer-training/parallelization-advantages.md">Parallelization Advantages Over RNNs</a></li>
<li><a href="transformer-training/self-attention-computational-complexity.md">Computational Complexity of Self-Attention</a></li>
<li><a href="transformer-training/self-attention-memory-complexity.md">Memory Complexity of Self-Attention</a></li>
<li><a href="transformer-training/quadratic-scaling.md">Quadratic Scaling Issue (O(n²))</a></li>
<li><a href="transformer-training/attention-map-sparsity.md">Attention Map Sparsity Patterns</a></li>
<li><a href="transformer-training/sequence-length-limitations.md">Sequence Length Limitations</a></li>
</ul>
</div>

<div class="toc-card">
<h3>🔤 Tokenization & Decoding</h3>
<ul>
<li><a href="tokenization-decoding/training-pipeline.md">Transformer Training Pipeline</a></li>
<li><a href="tokenization-decoding/tokenization-basics.md">Tokenization Basics</a></li>
<li><a href="tokenization-decoding/byte-pair-encoding.md">Byte Pair Encoding (BPE)</a></li>
<li><a href="tokenization-decoding/beam-search.md">Beam Search Decoding</a></li>
<li><a href="tokenization-decoding/greedy-decoding.md">Greedy Decoding</a></li>
<li><a href="tokenization-decoding/temperature-sampling.md">Temperature Sampling</a></li>
<li><a href="tokenization-decoding/top-k-sampling.md">Top-K Sampling</a></li>
<li><a href="tokenization-decoding/top-p-sampling.md">Top-P Sampling</a></li>
<li><a href="tokenization-decoding/label-smoothing.md">Label Smoothing</a></li>
</ul>
</div>

<div class="toc-card">
<h3>⚙️ Optimization & Inference</h3>
<ul>
<li><a href="transformer-optimization/dropout-in-transformers.md">Dropout in Transformers</a></li>
<li><a href="transformer-optimization/initialization-strategies.md">Initialization Strategies</a></li>
<li><a href="transformer-optimization/gradient-flow.md">Gradient Flow in Transformers</a></li>
<li><a href="transformer-optimization/optimization-challenges.md">Transformer Optimization Challenges</a></li>
<li><a href="transformer-optimization/vanishing-exploding-gradients.md">Vanishing/Exploding Gradients vs RNNs</a></li>
<li><a href="transformer-optimization/inference-flow.md">Transformer Inference Flow</a></li>
<li><a href="transformer-optimization/kv-caching.md">KV Caching (Decoder Inference)</a></li>
<li><a href="transformer-optimization/encoder-decoder-distinction.md">Encoder-Only vs Decoder-Only vs Encoder-Decoder</a></li>
</ul>
</div>

<div class="toc-card">
<h3>📊 Analysis & Limitations</h3>
<ul>
<li><a href="transformer-analysis/attention-is-all-you-need.md">Original Paper: Attention Is All You Need</a></li>
<li><a href="transformer-analysis/vanilla-attention-limitations.md">Limitations of Vanilla Attention</a></li>
<li><a href="transformer-analysis/long-context-inefficiency.md">Long-Context Inefficiency</a></li>
<li><a href="transformer-analysis/attention-interpretability.md">Attention Interpretability Issues</a></li>
<li><a href="transformer-analysis/foundation-efficient-variants.md">Foundation for Efficient Attention Variants</a></li>
</ul>
</div>

</div>

---

<p style="text-align:center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
    Each page includes detailed mathematical derivations (LaTeX), practical code snippets, and self-assessment MCQs with answers.
    Best viewed on any device — fully responsive layout. Built following the <em>Attention Is All You Need</em> lineage.
</p>
