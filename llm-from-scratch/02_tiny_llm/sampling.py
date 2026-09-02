"""
sampling.py
=============

Decoding strategies for turning a model's logits into an actual next
token: greedy, temperature, top-k, top-p (nucleus) sampling, frequency
penalty, and beam search.

WHY THIS EXISTS
-------------------
Every generate() you've built in this conversation so far uses GREEDY
decoding (torch.argmax) -- always picks the single highest-probability
token. This is deterministic and easy to debug, but trained models using
greedy decoding tend to produce repetitive, bland text ("the the the...").
Real LLM products never use pure greedy decoding for open-ended
generation -- they sample from a controlled, narrowed distribution
instead (or search over multiple candidate sequences, in beam search's
case). This module implements that.

THE STRATEGIES
------------------
1. Greedy (baseline, already built elsewhere):
       next_token = argmax(probabilities)
   Deterministic, same input always gives same output.

2. Temperature scaling:
       scaled_logits = logits / temperature
       probabilities = softmax(scaled_logits)
       next_token = sample from probabilities
   temperature < 1.0 sharpens the distribution (more confident/greedy-like).
   temperature > 1.0 flattens it (more random/diverse).
   temperature == 1.0 is the model's raw, untouched distribution.

3. Top-k sampling:
       Keep only the k highest-probability tokens, zero out everything
       else, renormalize, then sample. Prevents sampling from the very
       long tail of implausible tokens, while still allowing some
       randomness among the plausible ones.

4. Top-p (nucleus) sampling:
       Sort tokens by probability descending. Keep adding tokens to a
       candidate set until their CUMULATIVE probability exceeds p (e.g.
       0.9), then zero out everything outside that set, renormalize, and
       sample. Unlike top-k's FIXED count, top-p adapts: when the model
       is very confident (one token dominates), the candidate set is
       small; when it's uncertain (many tokens similarly likely), the
       set is larger.

5. Frequency penalty:
       Subtract penalty * (times this token has already appeared) from
       each token's logit before sampling. Discourages the model from
       repeating itself -- unlike top-k/top-p, this looks BACKWARD at
       what's already been generated, not just at the current step's
       distribution shape.

6. Beam search:
       Instead of committing to one token at a time, track the
       num_beams most promising PARTIAL SEQUENCES simultaneously, each
       with a cumulative log-probability score. At each step, expand
       every beam by its top candidates, then keep only the best
       num_beams sequences overall. This can find a higher total-
       probability sequence than greedy decoding, which is short-
       sighted -- it only ever looks one token ahead and can't undo an
       early choice that turns out to lead somewhere worse.

These (1-5) can be combined (e.g. apply temperature, THEN top-k, THEN
top-p, THEN frequency penalty, then sample) -- this module builds each
as an independent, composable function so you can chain them explicitly
in generate(). Beam search (6) is a different mode of decoding entirely
-- deterministic search over multiple sequences rather than per-step
sampling from one -- so it lives in its own function with its own loop,
rather than composing into sample_next_token.
"""

from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Scale logits by temperature before softmax.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (..., vocab_size). Raw model output, NOT yet passed
        through softmax.
    temperature : float
        Must be > 0. Values < 1.0 sharpen the distribution (closer to
        greedy); values > 1.0 flatten it (more random). A value of
        exactly 1.0 leaves logits unchanged (dividing by 1.0 is a no-op).

    Returns
    -------
    torch.Tensor
        Same shape as input. Still raw logits, not yet softmaxed -- this
        function does ONE thing (scaling); softmax is applied separately
        by the caller, so this composes cleanly with top_k/top_p below.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature


def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Zero out (set to -inf) all logits except the top-k highest values,
    so that a subsequent softmax+sample only considers those k tokens.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (..., vocab_size).
    k : int
        Number of top tokens to keep. Must be >= 1 and <= vocab_size.

    Returns
    -------
    torch.Tensor
        Same shape as input. Positions outside the top-k are set to
        float('-inf') (so softmax will assign them exactly zero
        probability); positions inside top-k are UNCHANGED (not
        renormalized here -- softmax handles normalization when it's
        eventually applied, so we don't do it twice).
    """
    vocab_size = logits.size(-1)
    if not (1 <= k <= vocab_size):
        raise ValueError(f"k must be in [1, {vocab_size}], got {k}")

    top_values, top_indices = torch.topk(logits, k, dim=-1)

    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, top_indices, top_values)
    return filtered


def filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: keep the smallest set of highest-probability
    tokens whose CUMULATIVE probability exceeds p, zero out the rest.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (..., vocab_size). Raw logits (this function applies its
        own internal softmax to compute cumulative probability -- it
        does NOT modify the returned logits' scale, only which positions
        get zeroed).
    p : float
        Must be in (0.0, 1.0]. E.g. p=0.9 keeps tokens covering the top
        90% of probability mass.

    Returns
    -------
    torch.Tensor
        Same shape as input, with positions outside the nucleus set to
        float('-inf'), matching filter_top_k's convention.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0.0, 1.0], got {p}")

    probs = torch.softmax(logits, dim=-1)

    # Sort descending so we can walk the distribution from most to least likely.
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # A sorted position is "past the nucleus" once the cumulative probability
    # UP TO AND INCLUDING it already exceeds p. We then shift this mask right
    # by one position so the token that actually CROSSES the threshold is
    # kept (it's what pushed us over p, so it belongs in the nucleus) --
    # only tokens strictly AFTER the crossing point get removed.
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False  # always keep the single most likely token

    # Map the sorted-order removal mask back to original vocabulary positions.
    indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
    indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

    filtered = logits.masked_fill(indices_to_remove, float("-inf"))
    return filtered


def apply_frequency_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Penalize tokens proportional to how many times they've already
    appeared in the generated sequence, discouraging repetition.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (batch, vocab_size) -- logits for the current step.
    generated_ids : torch.Tensor
        Shape (batch, seq_len_so_far) -- token IDs generated in this
        sequence so far (NOT including the token about to be sampled).
        Same convention as `idx` in generate(): the running sequence.
    penalty : float
        Must be >= 0. 0.0 disables the penalty entirely (returns logits
        unchanged). Larger values push repeated tokens' logits down more
        aggressively. Unlike top-k/top-p, there's no natural upper bound
        -- typical values in practice are small, roughly 0.0-2.0.

    Returns
    -------
    torch.Tensor
        Same shape as `logits`. For each vocabulary position, the
        penalty * (number of times that token ID appears in
        generated_ids) is SUBTRACTED from its logit. Positions never
        generated are unaffected.
    """
    if penalty < 0:
        raise ValueError(f"penalty must be >= 0, got {penalty}")
    if penalty == 0.0:
        return logits

    vocab_size = logits.size(-1)

    # Count occurrences of each vocabulary ID within generated_ids, per
    # batch row. scatter_add_ accumulates a 1.0 into `counts` at each
    # position named by generated_ids, once per occurrence -- exactly a
    # per-row histogram over the vocabulary.
    counts = torch.zeros(logits.size(0), vocab_size, dtype=logits.dtype, device=logits.device)
    ones = torch.ones_like(generated_ids, dtype=logits.dtype)
    counts.scatter_add_(-1, generated_ids, ones)

    return logits - penalty * counts


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    frequency_penalty: float = 0.0,
    generated_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Full sampling pipeline: frequency penalty -> temperature -> top-k ->
    top-p -> sample.

    This is the function generate() (in causal_lm.py) should call once
    per generation step, replacing a greedy torch.argmax call.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (batch, vocab_size) -- logits for a SINGLE position (the
        last position in the sequence, same as generate()'s
        `logits[:, -1, :]` slice).
    temperature : float
        Passed to apply_temperature. Default 1.0 (no scaling).
    top_k : int or None
        If given, passed to filter_top_k. None disables top-k filtering
        (all tokens remain candidates going into top-p/sampling).
    top_p : float or None
        If given, passed to filter_top_p. None disables nucleus
        filtering.
    frequency_penalty : float
        If > 0, passed to apply_frequency_penalty along with
        generated_ids. Default 0.0 (disabled).
    generated_ids : torch.Tensor or None
        Shape (batch, seq_len_so_far). Required if frequency_penalty > 0
        (there's nothing to penalize repetition against otherwise).
        Ignored if frequency_penalty == 0.0.

    Returns
    -------
    torch.Tensor
        Shape (batch, 1) -- sampled token IDs, same shape convention as
        generate()'s torch.argmax(..., keepdim=True) output, so it's a
        drop-in replacement.

    Note
    ----
    With temperature=1.0, top_k=None, top_p=None, frequency_penalty=0.0,
    this does NOT reduce to greedy decoding -- it still samples randomly
    from the model's raw, unmodified distribution. Greedy behavior
    remains a separate code path (argmax), not something this function
    produces as a "no-op" case.
    """
    if frequency_penalty > 0.0:
        if generated_ids is None:
            raise ValueError("generated_ids is required when frequency_penalty > 0")
        logits = apply_frequency_penalty(logits, generated_ids, frequency_penalty)

    logits = apply_temperature(logits, temperature)

    if top_k is not None:
        logits = filter_top_k(logits, top_k)

    if top_p is not None:
        logits = filter_top_p(logits, top_p)

    probabilities = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token


def _model_logits(model, idx: torch.Tensor, context_size: int) -> torch.Tensor:
    """
    Shared helper: run `model` on the last `context_size` tokens of
    `idx` and return logits for the NEXT position only, shape
    (batch, vocab_size). Unwraps HuggingFace-style CausalLMOutput
    objects (see the `.logits` bug from earlier in this conversation)
    as well as plain-tensor outputs like MiniGPT's.
    """
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
        output = model(idx_cond)
    logits = output.logits if hasattr(output, "logits") else output
    return logits[:, -1, :]


@torch.no_grad()
def beam_search(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    num_beams: int = 4,
    eos_id: int | None = None,
    length_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Beam search decoding: track the `num_beams` most promising partial
    sequences at every step, instead of committing to one token at a
    time like greedy or sampling decoding do.

    WHY THIS CAN BEAT GREEDY
    ----------------------------
    Greedy decoding picks the single best NEXT token at every step, with
    no ability to reconsider. This is short-sighted: the locally-best
    token at step t can lead to a much worse sequence overall than a
    token that looked slightly worse at step t but opened up a much
    better step t+1. Beam search hedges against this by keeping several
    candidate continuations alive at once, and only commits at the end.

    HOW IT WORKS
    ----------------
    1. Start with one beam: the input sequence, cumulative log-prob 0.
    2. At each step, for every live beam, get the model's next-token
       log-probabilities and take the top `num_beams` candidate tokens.
       This produces num_beams * num_beams candidate sequences.
    3. Score each candidate by cumulative log-probability (sum of the
       log-prob of every token in the sequence so far), length-
       normalized by dividing by (length ** length_penalty) -- without
       this normalization, beam search would systematically prefer
       SHORTER sequences, since every additional token multiplies in
       another probability <= 1.0 (i.e. cumulative log-prob is
       monotonically non-increasing with length).
    4. Keep only the top `num_beams` candidates overall, discard the rest.
    5. Repeat until max_new_tokens is reached, or (if eos_id is given)
       every surviving beam has generated the end-of-sequence token.
    6. Return the single highest-scoring completed sequence.

    Parameters
    ----------
    model : nn.Module
        Must accept a (batch, seq_len) LongTensor of token IDs and
        return either a raw logits tensor of shape
        (batch, seq_len, vocab_size), OR an object with a `.logits`
        attribute of that shape (e.g. HuggingFace's CausalLMOutput).
    idx : torch.Tensor
        Shape (1, seq_len) -- the prompt to continue. Beam search here
        supports a single input sequence at a time (batch size 1 IN,
        though internally it tracks num_beams candidates).
    max_new_tokens : int
        Maximum number of tokens to generate before stopping, even if
        no beam has produced eos_id yet.
    context_size : int
        Max sequence length the model can attend to -- each step feeds
        the model only the last `context_size` tokens of each beam.
    num_beams : int
        Number of candidate sequences tracked simultaneously. num_beams=1
        reduces to (a slightly more expensive way of computing) greedy
        decoding. Higher values explore more of the search space at
        proportionally higher compute cost.
    eos_id : int or None
        If given, a beam stops extending once it generates this token.
        Once EVERY surviving beam has hit eos_id, search stops early.
    length_penalty : float
        Exponent used to length-normalize beam scores (see step 3
        above). 1.0 is a neutral full-length normalization; values < 1.0
        favor longer sequences relatively more, values > 1.0 favor
        shorter sequences more.

    Returns
    -------
    torch.Tensor
        Shape (1, original_seq_len + generated_len) -- the single best
        completed sequence found, including the original prompt.
    """
    if idx.size(0) != 1:
        raise ValueError(
            f"beam_search expects a single input sequence (batch size 1), "
            f"got batch size {idx.size(0)}"
        )
    if num_beams < 1:
        raise ValueError(f"num_beams must be >= 1, got {num_beams}")

    # Each beam: (sequence tensor of shape (1, cur_len), cumulative log-prob)
    beams: list[tuple[torch.Tensor, float]] = [(idx, 0.0)]
    finished: list[tuple[torch.Tensor, float]] = []

    def length_normalized_score(item: tuple[torch.Tensor, float]) -> float:
        seq, score = item
        return score / (seq.size(1) ** length_penalty)

    for _ in range(max_new_tokens):
        candidates: list[tuple[torch.Tensor, float]] = []

        for seq, score in beams:
            logits = _model_logits(model, seq, context_size)
            log_probs = torch.log_softmax(logits, dim=-1)  # shape (1, vocab_size)

            top_log_probs, top_indices = torch.topk(log_probs, num_beams, dim=-1)

            for i in range(num_beams):
                next_id = top_indices[:, i : i + 1]  # shape (1, 1)
                new_seq = torch.cat([seq, next_id], dim=1)
                new_score = score + top_log_probs[0, i].item()
                candidates.append((new_seq, new_score))

        # Keep only the best num_beams candidates overall (not per-beam --
        # this is what makes it a BEAM search rather than num_beams
        # independent greedy searches).
        candidates.sort(key=length_normalized_score, reverse=True)
        beams = candidates[:num_beams]

        if eos_id is not None:
            still_running = []
            for seq, score in beams:
                if seq[0, -1].item() == eos_id:
                    finished.append((seq, score))
                else:
                    still_running.append((seq, score))
            beams = still_running

            if not beams:  # every surviving beam has hit eos_id
                break

    finished.extend(beams)  # include any beams still running at max_new_tokens
    finished.sort(key=length_normalized_score, reverse=True)

    best_seq, _ = finished[0]
    return best_seq