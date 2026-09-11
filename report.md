# Report — AI Support Agent for AmazonHelp

## 1. Problem framing

**What "good" means for this brand:** AmazonHelp handles an enormous, noisy volume
of Twitter support requests spanning delivery status, refunds, account access,
and billing disputes. "Good" here means: (1) correctly routing a message to
the right underlying issue without needing the customer to repeat themselves,
(2) drafting a reply that reflects how AmazonHelp actually resolves that kind
of issue historically (not a generic apology), and (3) being conservative
about auto-handling — anything involving money, security, or a customer who
has already been failed once should go to a human, even at the cost of
escalating some cases that didn't strictly need it.

**What I chose not to build:**
- **No multi-turn conversation memory.** Each message is classified and
  answered independently. Real support is often a back-and-forth thread;
  this v1 only handles the first inbound message. Documented as future work.
- **No non-English handling.** Intent discovery surfaced significant Spanish,
  French, German, Portuguese, and Japanese traffic to AmazonHelp. This version
  scopes to English only and labels non-English messages `other_or_unclear`
  in the golden set — a real, non-trivial slice of traffic is out of scope.
- **No fine-tuning.** Prompting + retrieval-based grounding only. Faster to
  iterate on, and every decision is auditable/explainable, which mattered
  given the live code-walkthrough requirement.
- **TF-IDF retrieval instead of embeddings** for grounding examples — zero
  extra cost/latency, but weaker on semantically-similar-but-differently-worded
  messages (see failure analysis).

## 2. Results vs. baselines

| Metric | Trivial baseline | Simple baseline (keyword + nearest-neighbor) | This pipeline |
|---|---|---|---|
| Intent accuracy | 27.0% | 28.5% | **84.3%** |
| Escalation accuracy | 53.5% | 53.5% | **71.4%** |
| Avg LLM-judge reply quality (1-5) | n/a | n/a | **2.89** |

Baselines were computed on the full 200-example golden set (no API calls,
rule-based). The pipeline's numbers are from **70 of 200** examples — see
Section 4, this is not the full set.

The trivial baseline (one canned reply, always auto-handle) and the simple
baseline (keyword-matched intent + nearest-neighbor reply) both hover near
chance on intent classification, confirming this isn't an easy problem to
solve with fixed rules — real customer language is too varied. The pipeline's
intent accuracy is a large, genuine improvement. Escalation accuracy is
better but more modest — see failure analysis below for why.

## 3. Failure analysis — top 5 failure modes

1. **Keyword-triggered false positives on intent.** A rant about a TV show
   premiere containing the word "delay" was classified as `delivery_delay`
   by the underlying model despite having nothing to do with a package.
   Hypothesis: short, informal tweet text gives the model little disambiguating
   context, so it leans on surface keywords.

2. **Escalation misses "already frustrated" signals.** Several messages where
   a customer explicitly said support had already failed them (e.g. "customer
   care refuses to help") were initially auto-handled by the pipeline's first
   pass despite clearly needing a human. Hypothesis: the escalation prompt's
   rule list (money/security/anger/uncertainty) doesn't explicitly weight
   *prior failed contact*, which is a strong real-world escalation signal.

3. **Rate-limit-induced silent gaps in evaluation.** The free-tier Groq API
   quota was exhausted partway through the 200-example eval run, so headline
   numbers are computed on only 70 examples, not the full golden set — see
   Section 4.

4. **Non-English traffic collapses to one bucket.** Every non-English message
   gets `other_or_unclear` regardless of its real underlying intent, so the
   agent provides no useful classification for a meaningful fraction of
   AmazonHelp's actual traffic.

5. **TF-IDF grounding retrieval misses paraphrases.** Because retrieval is
   lexical (word-overlap based) rather than semantic, a message like "the box
   was empty" and a historical example saying "package arrived with nothing
   inside" would not be retrieved as similar even though they describe the
   same issue, weakening the grounding for reply drafting in those cases.

## 4. What is misleading about my headline number?

The 84.3% intent accuracy and 71.4% escalation accuracy are computed on
**70 of the 200 golden examples (35%)** — the free Groq API tier's daily
token quota was exhausted partway through the eval run, and the remaining
130 rows were skipped rather than evaluated. The 70 that succeeded were not
adversarially selected, but they were simply whichever rows ran before the
quota ran out (i.e., the first ~70 rows in file order), which could
correlate with row characteristics if the golden set has any ordering
pattern. **These numbers should be read as a promising early signal, not a
statistically confident final result.** A full re-run with a paid tier or
across multiple days would be needed for a trustworthy headline number.

Additionally: the golden set's gold labels were produced via AI-assisted
labeling (model suggestions reviewed and corrected by a human) rather than
fully independent human labeling from scratch — see decision log. This is a
legitimate, common practice, but it means labeling errors that the human
reviewer shares with the model's own tendencies wouldn't be caught, which
could inflate agreement between predictions and "gold" on cases where both
share the same blind spot.

## 5. What I'd do next with one more week

- Re-run the full 200-example eval on a paid tier (or spread across multiple
  days) to get a trustworthy full-sample accuracy number.
- Add embedding-based retrieval instead of TF-IDF to fix the paraphrase-miss
  failure mode.
- Explicitly add "customer states a prior support attempt already failed" as
  an escalation trigger in the prompt, and re-test against the golden set
  rows that currently fail on this pattern.
- Get a second human labeler for a subset of the golden set to compute
  inter-annotator agreement, strengthening confidence in the gold labels
  themselves (currently single-reviewer, AI-assisted).
- Add basic non-English routing (detect language, either translate before
  classification or add a dedicated non-English intent bucket that still
  gets a real, useful response instead of `other_or_unclear`).
- Add multi-turn context so replies to follow-up messages in a thread aren't
  evaluated in isolation.
