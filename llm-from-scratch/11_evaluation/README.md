# 11 - Evaluation

Without this stage, you can't objectively say whether SFT (09),
preference alignment (10), or quantization (13) actually helped -- this
is the missing measurement layer the whole rest of the pipeline depends
on to be more than guesswork.

```
automated_benchmarks.py   -- standard benchmark suites (perplexity on held-out
                                text, plus task benchmarks like a small MMLU-style
                                multiple-choice eval)
human_evaluation.py         -- tooling for collecting human preference judgments
                                  between two model outputs (A/B comparison)
model_based_evaluation.py     -- "LLM-as-judge": use a strong model to score/compare
                                    your model's outputs, cheaper and faster than
                                    human eval, useful for rapid iteration
feedback_signal.py              -- aggregates eval results across runs into a
                                      single trackable signal (logged to MLflow,
                                      same experiment-tracking pattern as every
                                      training script in this repo)
```

## Suggested build order

1. `automated_benchmarks.py` first -- perplexity is something you
   already compute in every training loop (`calc_loss_loader`); this
   just formalizes it as a standing, comparable eval run against a FIXED
   held-out set, decoupled from any one training run.
2. `model_based_evaluation.py` -- fast to build (just prompts + a
   scoring rubric), gives you a way to compare model versions without
   needing human raters for every iteration.
3. `human_evaluation.py` -- more infrastructure (needs a simple UI or at
   minimum a structured CLI/spreadsheet workflow), do this once you have
   specific model comparisons worth spending human time on.
4. `feedback_signal.py` last -- ties the above together once they exist.
