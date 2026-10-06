# Changes

## Publication-readiness fixes

- Replaced the old local lookup benchmark path with HotpotQA-based evaluation data saved as `data/evaluation/hotpotqa_eval.json`.
- Prepared one shared retrieval corpus in `data/corpus/hotpot_corpus.json`; Wikipedia article fetching is attempted by title, with a shared HotpotQA distractor-corpus fallback when article fetching is unavailable.
- Added `data/evaluation/supporting_fact_lookup.json` generation so retrieval metrics can compare retrieved chunks against gold supporting sentences.
- Added retrieval recall and hit-rate reporting alongside EM, F1, latency, tokens, and Groq-estimated cost.
- Updated adaptive retrieval buckets to Top-5, Top-8, and Top-12, with a minimum Top-5 retrieval floor.
- Moved feature min-max normalization into `src/normalizer.py`.
- Added `src/compute_stats.py` to compute feature min/max statistics from validation queries.
- Added validation-driven budget tuning in `src/tune_budget.py`, writing `data/best_budget_config.json`.
- Replaced noisy boolean keywords with stricter keywords and spaCy yes/no-question detection.
- Improved multihop detection using entity count and conjunction structure.
- Added multihop ablation, top-k truncation warnings, Groq API retry/backoff, and clearer pipeline result naming.
- Updated `run.py` with `--sample_size`, `--val_size`, `--budget_config`, `--feature_stats`, `compute-stats`, and `tune`.
