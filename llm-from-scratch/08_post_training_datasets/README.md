# 08 - Post-Training Datasets

Everything needed to turn raw text/preference data into the exact
formats SFT (09), reward modeling, DPO, and PPO (10) actually consume.
This stage sits between pretraining (07) and SFT (09) in the pipeline --
build it before instruction fine-tuning, since 09 depends on
`chat_templates.py`.

```
chat_templates.py           -- format (instruction, response) pairs into a single
                                  token sequence with a chat template + loss mask
synthetic_data_generation.py  -- generate training examples using a larger model
                                    (self-instruct / OSS-Instruct style, per the
                                    Gorilla/Hermes/ToolACE approaches discussed
                                    in this repo's data-sourcing conversation)
data_enhancement.py             -- rewrite/expand/diversify existing examples
                                      (paraphrasing, difficulty variation)
quality_filtering.py              -- general post-training quality filters
                                        (distinct from data_collection/books/
                                        cleaner.py, which is pretraining-corpus-
                                        specific -- this operates on
                                        instruction/response PAIRS, checking
                                        things like response relevance and
                                        length outliers, not raw book text)
```
