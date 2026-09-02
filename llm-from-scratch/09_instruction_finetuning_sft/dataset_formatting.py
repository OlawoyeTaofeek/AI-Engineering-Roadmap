"""Format (instruction, response) pairs into a single token sequence, with a
loss mask so cross-entropy is only computed over response tokens, not the
prompt -- the model shouldn't be trained to "predict" the instruction itself.
"""
raise NotImplementedError("TODO: tokenize prompt+response together, "
                           "set target_ids = -100 (ignored by cross_entropy) for prompt positions")
