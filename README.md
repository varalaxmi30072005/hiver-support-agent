# Hiver SDE Intern Take-Home — AI Support Agent (AmazonHelp)

An AI support agent that classifies intent, drafts a grounded reply, and
decides auto-handle vs escalate for customer tweets to a chosen brand's
support account, built on the Kaggle "Customer Support on Twitter" dataset.

## Setup (5 min)

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here   # free key from console.groq.com, no card needed
```

Download `twcs.csv` from Kaggle (thoughtvector/customer-support-on-twitter)
and place it at `data/twcs.csv`.

## Reproduce headline results (<15 min)

```bash
# 1. Build brand-specific (customer -> reply) thread pairs
python src/build_dataset.py --brand AmazonHelp --sample 5000

# 2. (Optional, one-time) discover intents from the data instead of guessing
python src/discover_intents.py --threads data/AmazonHelp_threads.csv

# 3. Try the pipeline on a single message
python src/agent.py --threads data/AmazonHelp_threads.csv \
    --message "my package never arrived and its been 2 weeks"

# 4. Run the full eval harness against your hand-labeled golden set
python src/eval_harness.py --threads data/AmazonHelp_threads.csv \
    --golden data/golden_eval.csv --human_agreement data/human_judge_agreement.csv
```

Results land in `eval/results.csv` and `eval/summary.json`.

## Repo structure

```
src/
  build_dataset.py     # raw twcs.csv -> brand-specific thread pairs
  discover_intents.py  # data-driven intent taxonomy discovery
  agent.py              # classify -> retrieve -> draft -> escalate pipeline
  eval_harness.py       # metrics + LLM-as-judge + judge/human agreement
data/
  golden_eval_TEMPLATE.csv   # replace with your real 150-250 hand-labeled rows
decision_log.md
report.md              # or REPORT section below
```

## Status / what's actually done vs scaffold

This repo was scaffolded to run end-to-end, but the following steps still
need REAL work before submission — do not submit as-is:

- [ ] Run `discover_intents.py` on real AmazonHelp data, manually review
      clusters, finalize `INTENT_SET` in `src/agent.py`.
- [ ] Hand-label 150-250 real rows into `data/golden_eval.csv` (see sampling
      note below — do not just take the first N rows).
- [ ] Hand-score a subset (~30 rows) yourself into
      `data/human_judge_agreement.csv` to compute judge-human agreement.
- [ ] Run the eval harness, fill in real numbers in `report.md`.
- [ ] Do the failure analysis with real examples once you have real results.
- [ ] Fill in `decision_log.md` with your actual decisions (10-15 total).

## Golden set sampling note (fill in for real)

Describe here how you sampled your 150-250 examples — e.g. stratified across
your discovered intents, weighted toward longer/ambiguous threads, including
some deliberately hard/edge cases, excluding non-English tweets, etc. State
it explicitly; "how you sampled" is graded.
