"""
model_based_evaluation.py
=============================

"LLM-as-judge": use a strong external model (via Ollama, same setup
covered in this repo's serving conversation) to score or compare your
model's outputs against a rubric -- much faster and cheaper than human
evaluation, useful for rapid iteration between training runs, though
less authoritative than real human judgment for final decisions.

TODO
-------
1. Implement `score_response(prompt, response, judge_model, rubric)` --
   sends a structured prompt to the judge model asking it to rate the
   response (e.g. 1-10) against a rubric (helpfulness, correctness,
   relevance), parses the score from the judge's reply. Consider using
   the constrained/structured-output approach discussed earlier in this
   conversation (JSON schema output) so the judge's score is reliably
   parseable rather than free-text.
2. Implement `compare_responses(prompt, response_a, response_b,
   judge_model)` -- asks the judge model to pick which of two responses
   is better, for direct A/B comparison (mirrors human_evaluation.py's
   structure, but automated).
3. IMPORTANT CAVEAT to document in your usage: judge models have known
   biases (e.g. favoring longer responses, favoring responses similar in
   style to their own). Don't treat model-based eval scores as ground
   truth -- use them for fast iteration signal, and validate important
   conclusions against human_evaluation.py before making final decisions.
"""

raise NotImplementedError("Implement LLM-as-judge evaluation functions -- see module docstring")
