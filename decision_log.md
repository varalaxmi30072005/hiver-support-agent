# Decision Log

A plain list of non-obvious decisions made while building this, and why.
Fill in / edit these as you actually make choices — the ones below are
starting points from the initial scaffold.

1. **Brand: AmazonHelp** — chosen for high message volume and a small number
   of clearly repeatable issue types (delivery, refunds, account), which
   makes both intent definition and grounding-by-example tractable.
2. **Retrieval for grounding uses TF-IDF cosine similarity, not embeddings** —
   zero extra API cost/latency, good enough for lexically-similar tweet-length
   text. Trade-off: misses semantically-similar-but-differently-worded
   messages. Documented as a "what I chose not to build."
3. **Intents are discovered via clustering + LLM labeling, not hand-guessed
   upfront** — reduces the risk of an intent taxonomy that doesn't match
   what's actually in the data. Requires manual review of clusters before
   finalizing (see src/discover_intents.py).
4. **Escalation is a separate LLM call, not baked into the reply-drafting
   prompt** — keeps the "why escalate" reasoning auditable and separately
   testable/gradable, rather than inferred from reply tone.
5. **Two baselines**: a trivial canned-reply baseline, and a simple
   keyword-rule + nearest-neighbor baseline with no LLM reasoning at all —
   to make the LLM pipeline's marginal value measurable, not just "it works."
6. **LLM-as-judge uses 3 sub-dimensions (grounded/helpful/tone) plus an
   overall score**, not a single number — makes failure analysis possible
   (e.g. "helpful but not grounded" is a distinct, diagnosable failure mode).
7. **Judge-human agreement is measured with quadratic-weighted Cohen's kappa**,
   not raw accuracy — appropriate for ordinal 1-5 ratings where "off by one"
   should be penalized less than "off by four."
8. [Add more as you go — e.g. how you sampled the golden set, why you capped
   grounding examples at k=3, how you handled multi-turn threads vs single
   tweets, any brand-specific PII redaction, etc.]

<!-- Continue numbering to 10-15 total; the assignment requires this. -->
