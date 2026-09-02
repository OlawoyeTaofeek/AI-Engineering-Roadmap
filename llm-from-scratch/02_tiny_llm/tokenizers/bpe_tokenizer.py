"""
bpe_tokenizer.py
==================

A Byte-Pair Encoding (BPE) tokenizer, built from scratch.

WHY THIS EXISTS
-------------------
Every model in this repo so far has used either a character-level
tokenizer (toy demos) or tiktoken (GPT-2's BPE, already trained). Neither
teaches you how subword tokenization actually WORKS. BPE is what nearly
every modern LLM uses (GPT-2/3/4, LLaMA, Mistral all use BPE or a close
variant), and the algorithm itself is simple enough to implement in an
afternoon -- understanding it end to end closes a real gap in "do I
understand my own model's input pipeline."

THE ALGORITHM (read this before implementing)
---------------------------------------------------
BPE builds a vocabulary by starting from individual characters/bytes and
iteratively merging the MOST FREQUENT adjacent pair into a new token,
repeating until you hit a target vocabulary size.

    1. Start with a vocabulary of individual bytes/characters.
    2. Represent every word in your training corpus as a sequence of
       these base tokens (e.g. "low" -> ['l', 'o', 'w']).
    3. Count every ADJACENT PAIR of tokens across the whole corpus
       (e.g. ('l','o') appears how many times, ('o','w') how many times).
    4. Find the single most frequent pair, and merge it into one new
       token everywhere it appears (e.g. if ('l','o') is most frequent,
       every ['l','o',...] becomes ['lo',...]).
    5. Record this merge rule (('l','o') -> 'lo') -- this ordered list
       of merge rules IS the trained tokenizer.
    6. Repeat steps 3-5 until vocab_size is reached.

To ENCODE new text later: apply the SAME merge rules, IN THE SAME ORDER
they were learned, to the new text's character sequence.

To DECODE: just concatenate token strings back together (BPE merges are
lossless -- no information is thrown away, unlike some other subword
schemes).

YOUR TASK
------------
Implement `BPETokenizer.train()`, `.encode()`, and `.decode()`. Run:
    pytest 02_tiny_llm/tests/test_bpe_tokenizer.py -v
"""

from __future__ import annotations

from collections import Counter


class BPETokenizer:
    """
    A from-scratch Byte-Pair Encoding tokenizer.

    Attributes (populate these during train())
    --------------------------------------------
    merges : list[tuple[str, str]]
        Ordered list of merge rules, in the order they were learned.
        Order matters for encoding -- see encode()'s docstring.
    vocab : dict[str, int]
        Maps each token string to its integer ID.
    inverse_vocab : dict[int, str]
        Maps each integer ID back to its token string (for decode()).

    Examples
    --------
    >>> tokenizer = BPETokenizer()
    >>> tokenizer.train(corpus="low lower lowest slow slowest", vocab_size=270)
    >>> ids = tokenizer.encode("lowest")
    >>> tokenizer.decode(ids)
    'lowest'
    """

    def __init__(self) -> None:
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}

    def train(self, corpus: str, vocab_size: int) -> None:
        """
        Learn merge rules and build the vocabulary from a training corpus.

        Parameters
        ----------
        corpus : str
            Raw training text. For a first implementation, split on
            whitespace into words and treat each word independently
            (BPE traditionally doesn't merge ACROSS word boundaries --
            this keeps tokens from spanning "cat dog" into one token).
        vocab_size : int
            Target vocabulary size. Training stops once this many
            distinct tokens exist (base characters + learned merges).
            Must be >= the number of distinct base characters in the
            corpus, or training can never reach the target.

        TODO:
            1. Split corpus into words (str.split() is fine for a first
               version -- real BPE tokenizers use a more careful
               pre-tokenization regex, out of scope here).
            2. Represent each word as a list of individual characters,
               e.g. "low" -> ['l','o','w']. Store word frequencies too
               (Counter(words)) -- a word appearing 50 times should count
               its pairs 50 times, not once.
            3. Initialize self.vocab with one entry per distinct
               character seen (IDs 0, 1, 2, ...).
            4. Loop until len(self.vocab) >= vocab_size:
                a. Count all adjacent pairs across all words (weighted
                   by word frequency).
                b. If no pairs remain (all words fully merged into single
                   tokens), stop early.
                c. Find the most frequent pair. Break ties consistently
                   (e.g. alphabetically) so training is deterministic --
                   important for tests and reproducibility.
                d. Merge that pair everywhere it appears across all
                   words (e.g. ('l','o') found -> replace ['l','o',...]
                   with ['lo',...] in every word's token list).
                e. Record the merge in self.merges (append, preserving
                   order).
                f. Add the new merged token string to self.vocab with
                   the next available ID.
            5. Build self.inverse_vocab as the reverse of self.vocab.
        """
        raise NotImplementedError("Implement BPE training -- see algorithm description above")

    def encode(self, text: str) -> list[int]:
        """
        Convert text into a list of token IDs, using the learned merges.

        TODO:
            1. Split text into words (same splitting strategy as train()).
            2. For each word, start as a list of individual characters.
            3. Apply self.merges IN ORDER -- for each merge rule
               (a, b) -> ab, scan the word's token list and merge every
               occurrence of adjacent (a, b) into 'ab', before moving to
               the NEXT merge rule. Order matters: merges learned earlier
               in training take priority, exactly mirroring the order
               they were discovered.
            4. Look up each final token string in self.vocab to get its
               ID. If a token isn't in the vocabulary (shouldn't happen
               if every base character was seen during training, but
               guard against it), decide on a fallback (e.g. raise a
               clear error, or map to an <unk> token if you add one).
            5. Concatenate token IDs across all words in order, and
               return the flat list.
        """
        raise NotImplementedError("Implement encoding using self.merges, applied in order")

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back into the original text.

        TODO: look up each ID in self.inverse_vocab, concatenate the
        resulting token strings. Since encode() processes word-by-word
        and this reference implementation doesn't preserve explicit
        spaces as tokens, you'll need to decide how word boundaries are
        preserved -- the simplest approach: encode() inserts a space
        token between words, or you track word boundaries separately.
        Document whichever approach you take.
        """
        raise NotImplementedError("Implement decoding via self.inverse_vocab")
