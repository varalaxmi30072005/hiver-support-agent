# Hiver SDE Intern Take-Home — AI Support Agent (AmazonHelp)

An AI support agent that classifies intent, drafts a grounded reply, and
decides auto-handle vs escalate for customer tweets to AmazonHelp, built on
the Kaggle "Customer Support on Twitter" dataset.

## Setup (5 min)

    pip install -r requirements.txt
    export GROQ_API_KEY=your_key_here   # free key from console.groq.com, no card needed

Download `twcs.csv` from Kaggle (thoughtvector/customer-support-on-twitter)
and place it in this repo's root folder.

## Reproduce headline results (<15 min)

    # 1. Build brand-specific (customer -> reply) thread pairs
    python build_dataset.py --brand AmazonHelp --sample 5000

    # 2. Try the pipeline on a single message
    python agent.py --threads AmazonHelp_threads.csv --message "my package never arrived and its been 2 weeks"

    # 3. Run the full eval harness against the golden set
    python eval_harness.py --threads AmazonHelp_threads.csv --golden golden_eval.csv

    # 4. Run baselines for comparison (no API calls needed, instant)
    python run_baselines.py --threads AmazonHelp_threads.csv --golden golden_eval.csv

Results land in `results.csv` and `summary.json`.

## Repo contents

- `build_dataset.py` — raw twcs.csv to brand-specific thread pairs
- `discover_intents.py` — data-driven intent taxonomy discovery (clustering)
- `fix_intents.py` — one-time script that applied the final data-driven intent list
- `agent.py` — classify, retrieve, draft, escalate pipeline
- `eval_harness.py` — metrics plus LLM-as-judge, run against the golden set
- `run_baselines.py` — trivial and simple baseline comparison (no API calls)
- `build_golden_worksheet.py` — semi-automated golden-set labeling helper
- `golden_eval.csv` — the final 200-example hand-reviewed golden set
- `golden_eval_WORKSHEET.csv` — pre-labeling worksheet before human review
- `proposed_intents.json` — raw clustering output used to derive the intent set
- `decision_log.md` — 13 non-obvious decisions made during this build
- `report.md` — full write-up: problem framing, results, failure analysis, limitations

## Status

All core deliverables are complete: working pipeline, 200-example golden
eval set (AI-assisted labeling, human-reviewed), eval harness run against
real data, baseline comparisons, and a full report with honest limitations.
See `report.md` Section 4 for an important caveat: headline pipeline
accuracy numbers are based on 70 of 200 golden examples due to free-tier
API rate limits — this is disclosed explicitly, not hidden.

## Golden set sampling note

The 200 golden examples were sampled using length-based stratification
(short/medium/long thirds sampled separately) as a proxy for message
complexity, since there were no gold intents yet to stratify by directly.
Labels were produced via AI-assisted review: the pipeline's own suggested
intent/escalation was generated for each message, then a human reviewed
and corrected every row against the actual message text (see decision_log.md
item 7 for why this is disclosed as a limitation, not just a shortcut).
