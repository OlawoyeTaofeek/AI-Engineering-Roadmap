# 14 - New Trends

Four areas that don't fit neatly into the standard pretrain -> SFT ->
align -> deploy pipeline covered by stages 02-13, but represent active,
important directions worth understanding even at small scale.

```
model_merging.py       -- combine multiple fine-tuned models' weights directly
                             (no additional training) into one model
multimodal.py             -- extending a text-only model to accept image (or
                                other modality) inputs
interpretability.py          -- understanding WHAT a trained model has learned,
                                   beyond just its loss/benchmark numbers
test_time_compute.py           -- improving output quality by spending MORE
                                     compute at INFERENCE time, rather than only
                                     at training time (e.g. self-consistency,
                                     tree search over generations)
```

This stage is intentionally the most exploratory/optional in the repo --
treat it as "once stages 02-13 work end to end, here's where the field
is currently pushing further," not a strict prerequisite for having a
working model.
