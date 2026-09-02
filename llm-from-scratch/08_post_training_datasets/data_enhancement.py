"""
data_enhancement.py
======================

Enhances existing (instruction, response) examples: paraphrasing for
diversity, generating harder/easier variants, and expanding short
responses -- distinct from synthetic_data_generation.py, which creates
NEW examples from scratch; this module takes EXISTING examples and
makes more/better versions of them.

TODO
-------
1. Implement `paraphrase_instruction(instruction, teacher_model)` --
   generates an alternate phrasing of the same instruction, so the same
   underlying task appears in the dataset multiple ways (improves
   robustness to how users actually phrase requests).
2. Implement `vary_difficulty(instruction, teacher_model, direction)`
   where direction is "harder" or "easier" -- generates a modified
   instruction that's a meaningfully different difficulty level,
   following the same idea as Evol-Instruct-style dataset augmentation.
3. Implement `expand_short_response(instruction, response, teacher_model)`
   for cases where an existing response is too terse to be a good
   training example -- ask the teacher model to elaborate.
"""

raise NotImplementedError("Implement instruction/response enhancement functions -- see module docstring")
