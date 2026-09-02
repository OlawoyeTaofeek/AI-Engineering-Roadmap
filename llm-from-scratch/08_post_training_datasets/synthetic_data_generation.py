"""
synthetic_data_generation.py
================================

Generates synthetic instruction/response training examples using a
larger "teacher" model, following the self-instruct / OSS-Instruct
pattern discussed in this repo's data-sourcing conversation (Gorilla,
Hermes function-calling dataset, ToolACE).

APPROACH
------------
1. Seed prompts: a small hand-written set of example instructions
   covering the task types you want (coding, Q&A, tool use, etc).
2. Use a larger model (via Ollama or an API) to generate NEW, similar
   instructions from the seed set -- this is what "self-instruct" means:
   the model bootstraps more training data for itself/another model.
3. For each generated instruction, generate a response with the same
   (or a different, stronger) teacher model.
4. Filter (see quality_filtering.py in this same folder) before adding
   to your training set.

TODO
-------
1. Implement `generate_instructions(seed_examples, teacher_model, n)` --
   prompts the teacher model to produce n new instructions in the style
   of seed_examples. Use the Ollama HTTP API (see the Ollama serving
   conversation earlier) or an API client.
2. Implement `generate_response(instruction, teacher_model)`.
3. Implement `build_synthetic_dataset(seed_examples, teacher_model, n)`
   orchestrating both, returning a list of (instruction, response)
   pairs ready for chat_templates.py.

CAUTION: generating data with a model whose own output you'll train a
NEW model on risks amplifying that teacher model's biases/errors into
your model. Always run quality_filtering.py and spot-check a sample
before using generated data at scale.
"""

raise NotImplementedError("Implement synthetic instruction/response generation -- see module docstring")
