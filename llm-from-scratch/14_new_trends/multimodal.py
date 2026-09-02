"""
multimodal.py
==================

Extends the text-only GPTForCausalLM (02_tiny_llm) to accept image
inputs alongside text -- the architectural pattern most current
multimodal LLMs (LLaVA-style) actually use.

THE CORE PATTERN (simpler than it might sound)
-----------------------------------------------------
1. Run an image through a pretrained VISION encoder (e.g. a CLIP image
   encoder) to get a sequence of image feature vectors -- conceptually
   similar to how your tokenizer turns text into a sequence of token
   embeddings, except the "tokens" here are patches of the image.
2. Project those image features into the SAME embedding dimension as
   your text token embeddings, using a small learned linear layer (a
   "projector" / "adapter").
3. Concatenate the projected image features with the text token
   embeddings into ONE sequence, and feed that combined sequence through
   your EXISTING GPTModel/GPTForCausalLM unchanged -- the transformer
   itself doesn't need to know some of its input "tokens" originally
   came from an image; causal self-attention just treats them as more
   positions in the sequence.

TODO
-------
1. Implement `ImageProjector` (nn.Module) -- a small MLP mapping vision
   encoder output dimension -> your model's emb_dim.
2. Implement `encode_image(image, vision_encoder, projector)` -- returns
   image features already projected into emb_dim, ready to concatenate
   with text embeddings.
3. Implement `build_multimodal_input(image_features, text_token_ids,
   model)` -- runs text_token_ids through model.tok_emb (reusing YOUR
   existing GPTModel's embedding layer unchanged), concatenates with
   image_features, and returns the combined embedding sequence -- NOTE
   this means you'll need a variant of GPTModel.forward() that accepts
   pre-computed EMBEDDINGS directly, bypassing its own tok_emb lookup,
   since the image portion never goes through tok_emb at all.
4. Training-wise: typically only the ImageProjector (and optionally the
   language model, unfrozen) are trained on (image, text) paired data --
   the vision encoder itself is usually kept FROZEN (pretrained CLIP
   weights), since training it from scratch would need vastly more
   image data than this repo's pretraining stages assume.
"""

raise NotImplementedError("Implement image encoding and multimodal input construction -- see module docstring")
