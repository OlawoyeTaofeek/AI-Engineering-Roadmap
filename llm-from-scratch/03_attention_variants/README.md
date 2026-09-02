# 03 - Attention Variants

Every attention variant implemented separately, benchmarked against the same
tiny model/dataset, and explorable interactively.

```
implementations/    -- self, causal, multi-head, grouped-query (GQA), multi-query (MQA)
benchmarks/          -- run_comparison.py logs perplexity/speed/memory per variant to results.csv
charts/                -- generated comparison charts (not hand-made -- regenerate, don't hand-edit)
playground/              -- gradio app: type a prompt, tune temperature/top-k/top-p,
                              see the next-token probability distribution as a live bar chart
```

## Running the comparison

```bash
python benchmarks/run_comparison.py --variants self causal multihead gqa mqa
```

## Playground

```bash
python playground/app.py
```
