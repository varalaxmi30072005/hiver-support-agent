# Report — AI Support Agent for AmazonHelp

## 1. Problem framing

What does "good" mean for this brand? (e.g. for AmazonHelp: fast triage of
delivery/refund status questions without over-promising refunds; escalate
anything money- or account-security-related.)

What did I choose NOT to build, and why?
- e.g. no multi-turn conversation memory (single-message triage only) —
  scoped down to fit the time budget; documented as future work.
- e.g. no fine-tuning, only prompting + retrieval — faster to iterate on and
  easier to audit/explain live.

## 2. Results vs. baselines

| Metric | Trivial baseline | Simple baseline (keyword+NN) | This pipeline |
|---|---|---|---|
| Intent accuracy | — | — | — |
| Escalation accuracy | — | — | — |
| Avg LLM-judge overall (1-5) | — | — | — |

(Fill in from `eval/summary.json` after running the harness on real data
with all three approaches — trivial_baseline() and simple_baseline() are
implemented in src/eval_harness.py.)

## 3. Failure analysis — top 5 failure modes

For each: 1-2 real examples + your hypothesis for why it happens.

1. **[failure mode name]** — example: "..." — hypothesis: ...
2. ...
3. ...
4. ...
5. ...

## 4. What is misleading about my headline number?

(Mandatory section — be honest here. E.g.: the golden set may be skewed
toward common intents so accuracy looks better than real-world long-tail
performance; LLM-judge may be biased toward fluent-but-ungrounded replies;
subsample size limits statistical confidence; etc.)

## 5. What I'd do next with one more week

- ...
- ...
- ...
