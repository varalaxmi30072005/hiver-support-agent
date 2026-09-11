# Decision Log

1. **Brand: AmazonHelp.** Chosen for high message volume and a broad but
   tractable spread of issue types (delivery, refunds, billing, account
   access), which made both intent discovery and grounding-by-example
   feasible within the time available.

2. **Intents were discovered from the data via clustering, not guessed
   upfront.** 300 real customer messages were TF-IDF clustered into 10
   groups; each cluster was manually reviewed and merged/relabeled into
   a final 8-intent taxonomy (see `data/proposed_intents.json` for the raw
   clusters). Two clusters were duplicates and got merged; several
   non-English clusters informed the decision to scope this v1 to English.

3. **Used Groq's free API (`openai/gpt-oss-120b`) instead of a paid provider.**
   Zero cost, but subject to daily token-quota rate limits that materially
   affected how much of the golden set could be evaluated in one run (see
   report.md Section 4) — a real trade-off, not hidden.

4. **Retrieval for grounding uses TF-IDF cosine similarity, not embeddings.**
   Zero extra cost/latency, adequate for lexically-similar tweet-length text,
   but misses semantically-similar-but-differently-worded messages — a
   documented failure mode, not an oversight.

5. **Escalation is a separate LLM call from reply drafting**, not inferred
   from the drafted reply's tone. Keeps the escalation reasoning auditable
   and independently testable, and matches how the assignment frames it as
   a distinct decision requiring a stated reason.

6. **Two baselines with genuinely different mechanisms**: a trivial
   canned-reply baseline (always auto-handle) and a simple keyword-rule +
   nearest-neighbor baseline with zero LLM reasoning — both computed
   without any API calls, so they're cheap to re-run and not subject to
   rate limits, unlike the full pipeline eval.

7. **Golden eval set labels were AI-assisted, not built from scratch by
   hand.** All 200 sampled messages were first run through the pipeline to
   get a suggested intent/escalation, then reviewed and corrected against
   the actual message text. This is disclosed explicitly in report.md
   Section 4 because it's a genuine methodological limitation, not just a
   time-saving trick — shared blind spots between labeler and model are a
   real risk that a from-scratch human labeling process wouldn't have.

8. **Golden set sampling used length-based stratification** (short/medium/long
   message thirds, sampled separately) as a proxy for message complexity,
   since there were no gold intents yet to stratify by directly. This is
   an imperfect proxy — a message being long doesn't guarantee it's
   intent-ambiguous — but was chosen over pure random sampling to avoid
   under-representing rarer, more complex message types.

9. **Non-English messages were labeled `other_or_unclear` in the golden
   set, not skipped or auto-translated.** This keeps them in the eval
   denominator (so the pipeline is fairly penalized for not handling
   them) rather than quietly excluding a real slice of traffic from
   evaluation.

10. **Failed pipeline calls are skipped, not fatal, in the eval harness.**
    An early version crashed the entire eval run on a single malformed
    JSON response from the model. This was changed to catch and skip
    failed rows so partial results are always recoverable, which mattered
    given the rate-limit issues encountered during actual runs.

11. **JSON parsing from the LLM response is regex-extracted, not raw
    `json.loads`.** The model occasionally wraps JSON in markdown fences
    or adds stray text, which broke naive parsing; a small regex extracts
    the first `{...}` block before parsing, which fixed most failures.

12. **Escalation rules explicitly include "money, security, anger, or
    uncertainty in the drafted reply"** rather than a longer, vaguer list.
    Kept intentionally short so the reasoning stays auditable and
    explainable, at the cost of missing some real escalation signals
    (see failure analysis item 2 — "prior failed contact" isn't currently
    an explicit trigger).

13. **Max token budgets were increased from an initial low default (50-200)
    to 150-800 across prompts** after discovering the model spends part of
    its token budget on internal reasoning before writing the final answer,
    which silently produced empty outputs at the lower limits. This was a
    real bug caught during testing, not a design choice from the start —
    worth being upfront about if asked.
