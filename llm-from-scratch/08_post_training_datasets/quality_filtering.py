"""
quality_filtering.py
========================

Quality filters for instruction/response PAIRS specifically (as opposed
to data_collection/books/cleaner.py, which filters raw pretraining
text). Applied after synthetic_data_generation.py / data_enhancement.py,
before chat_templates.py tokenizes the final dataset.

TODO
-------
1. Implement `is_response_too_short(response, min_words)` -- filters out
   near-empty/lazy responses.
2. Implement `is_response_too_long(response, max_words)` -- filters out
   runaway generations (a common failure mode of synthetic generation --
   the teacher model rambles instead of giving a focused answer).
3. Implement `is_instruction_response_relevant(instruction, response,
   embedding_model)` -- uses a sentence-embedding model (e.g.
   sentence-transformers, same tool suggested for RAG earlier in this
   conversation) to check semantic similarity between instruction and
   response, filtering out pairs where the response doesn't actually
   address the instruction.
4. Implement `deduplicate_pairs(pairs)` -- reuse the fingerprinting
   approach from data_collection/books/cleaner.py's
   compute_fingerprint()/deduplicate_texts(), applied to the combined
   instruction+response text.
"""

raise NotImplementedError("Implement instruction/response quality filters -- see module docstring")
