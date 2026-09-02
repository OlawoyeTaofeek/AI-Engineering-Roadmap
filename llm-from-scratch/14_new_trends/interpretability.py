"""
interpretability.py
=======================

Tools for understanding WHAT a trained model has actually learned,
beyond aggregate loss/benchmark numbers -- inspecting attention patterns
and individual neuron/feature behavior.

TODO
-------
1. Implement `visualize_attention_weights(model, input_ids, layer_idx,
   head_idx)` -- runs a forward pass, extracts the attention weight
   matrix from a SPECIFIC layer and head (you'll need to modify
   MultiHeadAttention.forward, from 02_tiny_llm/model/attention.py, to
   optionally RETURN attn_weights instead of only the output -- a small,
   non-invasive addition), and renders it as a heatmap (token vs token).
   This directly extends the multi-head attention numeric walkthrough
   from earlier in this conversation, where you already observed
   different heads producing different attention patterns on the same
   input -- this tool makes that inspectable on a TRAINED model, where
   the patterns should be meaningfully structured rather than random.
2. Implement `find_max_activating_examples(model, layer_idx,
   neuron_idx, dataset, top_k)` -- runs the model over many examples
   from your training/eval data, records which INPUT tokens cause the
   highest activation for a specific neuron in a specific
   FeedForward layer, returns the top_k examples. This is the basic
   building block of "what does this neuron seem to detect" analysis.
3. Implement `logit_lens(model, input_ids, layer_idx)` -- applies the
   model's OWN final_norm + out_head to the hidden state at an
   INTERMEDIATE layer (not just the final layer), producing a
   "prediction" from that layer's partial computation. This reveals how
   the model's "best guess" for the next token evolves layer by layer
   through the network -- a well-known, simple, surprisingly informative
   interpretability technique.
"""

raise NotImplementedError("Implement attention visualization, neuron analysis, and logit lens -- see module docstring")
