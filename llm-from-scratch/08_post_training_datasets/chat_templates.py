"""
chat_templates.py
====================

Formats (instruction, response) conversation pairs into a single
tokenized sequence using a chat template, with a loss mask so training
only computes loss over the RESPONSE tokens, never the prompt/instruction.

WHY THIS MATTERS
--------------------
Feeding raw "instruction + response" text into your model without a
consistent template means the model never learns where a turn starts or
ends -- which is exactly why real chat models use structured markers
(ChatML's <|im_start|>/<|im_end|>, or similar). Equally important: if
you compute loss over the INSTRUCTION tokens too, you're training the
model to predict the user's own question, which wastes training signal
and can subtly hurt instruction-following (the model shouldn't be
rewarded for "predicting" what the user was about to type).

TODO
-------
1. Define a template format (recommend ChatML, since it's what most
   current open-source SFT datasets already use):
       <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
2. Implement `format_example(instruction, response) -> str` applying
   this template.
3. Implement `tokenize_with_loss_mask(instruction, response, tokenizer)`
   returning (input_ids, labels) where labels has -100 (PyTorch's
   ignore_index for cross_entropy) at every position corresponding to
   the instruction + template markers, and the REAL token ID only at
   response positions. This -100 convention is what lets you pass
   labels straight into GPTForCausalLM.forward() (from 02_tiny_llm)
   unchanged -- F.cross_entropy already ignores -100 targets by default
   via its ignore_index parameter.
4. Add special tokens (<|im_start|>, <|im_end|>) to your tokenizer's
   vocabulary BEFORE using this -- if using the BPE tokenizer from
   02_tiny_llm/tokenizer/, they need to be added as fixed vocabulary
   entries, not learned via merges.
"""

raise NotImplementedError("Implement format_example() and tokenize_with_loss_mask() -- see module docstring")
