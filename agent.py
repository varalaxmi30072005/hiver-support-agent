"""
Core support-agent pipeline.

For each incoming customer message:
  1. Classify intent (from a small closed set you define after looking at the data).
  2. Retrieve the most similar historical (customer_text -> brand_reply_text) pairs
     from the same brand's thread data, to ground the drafted reply.
  3. Draft a reply grounded in those historical resolutions.
  4. Decide auto-handle vs escalate, with a stated reason.

Uses the Anthropic API (claude-sonnet-4-6). Requires ANTHROPIC_API_KEY env var.

Usage:
    python src/agent.py --threads data/AmazonHelp_threads.csv --message "my package never arrived"
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"

# ---- 1. INTENT SET -----------------------------------------------------
# NOTE: Do not hardcode this blindly. Run src/discover_intents.py first on a
# sample of your brand's customer_text column, inspect the clusters, and
# replace this list with what the DATA actually shows. This is a placeholder
# starting point for a retail/e-commerce brand like AmazonHelp.
INTENT_SET = [
    "order_not_delivered",
    "wrong_or_damaged_item",
    "refund_or_return_request",
    "account_or_login_issue",
    "billing_or_charge_dispute",
    "delivery_delay_status_check",
    "general_complaint_or_feedback",
    "other_or_unclear",
]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Cheap lexical embedding fallback (TF-IDF) so this runs with zero extra
    API cost for retrieval. Swap for a real embedding model if you want
    better grounding quality."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    return vec.fit_transform(texts), vec


def retrieve_similar(query: str, threads: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    from sklearn.metrics.pairwise import cosine_similarity
    corpus = threads["customer_text"].fillna("").tolist() + [query]
    matrix, _ = embed_texts(corpus)
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    top_idx = sims.argsort()[::-1][:k]
    return threads.iloc[top_idx].assign(similarity=sims[top_idx])


def classify_intent(message: str) -> dict:
    prompt = f"""You are classifying a customer support tweet into ONE of these intents:
{json.dumps(INTENT_SET, indent=2)}

Customer message: "{message}"

Respond ONLY with JSON: {{"intent": "<one of the above>", "confidence": <0-1 float>, "reasoning": "<one short sentence>"}}"""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip().strip("```json").strip("```").strip()
    return json.loads(text)


def draft_reply(message: str, intent: str, grounding_examples: pd.DataFrame) -> str:
    examples_str = "\n\n".join(
        f"Similar past customer message: \"{row.customer_text}\"\nBrand's actual reply: \"{row.brand_reply_text}\""
        for row in grounding_examples.itertuples()
    )
    prompt = f"""You are drafting a customer support reply for a brand's Twitter support account.
Classified intent: {intent}

Here are real historical examples of how this brand resolved similar issues:
{examples_str}

New customer message: "{message}"

Draft a reply in the brand's voice/style, consistent with how they've actually resolved similar issues above.
Keep it concise (tweet-length, under 280 chars), empathetic, and actionable.
Respond ONLY with the reply text, nothing else."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def decide_escalation(message: str, intent: str, draft: str) -> dict:
    prompt = f"""You are a triage layer deciding whether a drafted customer support reply is safe
to auto-send, or should be escalated to a human agent.

Customer message: "{message}"
Classified intent: {intent}
Drafted reply: "{draft}"

Escalate to human if: the issue involves money/refund amounts, legal/safety complaints, account
security, strong negative sentiment/anger, or the drafted reply expresses uncertainty.
Otherwise, auto-handle is fine for simple status/info questions.

Respond ONLY with JSON: {{"decision": "auto_handle" | "escalate", "reason": "<one short sentence>"}}"""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip().strip("```json").strip("```").strip()
    return json.loads(text)


def run_pipeline(message: str, threads: pd.DataFrame) -> dict:
    intent_result = classify_intent(message)
    grounding = retrieve_similar(message, threads, k=3)
    draft = draft_reply(message, intent_result["intent"], grounding)
    escalation = decide_escalation(message, intent_result["intent"], draft)
    return {
        "message": message,
        "intent": intent_result["intent"],
        "intent_confidence": intent_result.get("confidence"),
        "intent_reasoning": intent_result.get("reasoning"),
        "grounding_examples_used": grounding["customer_text"].tolist(),
        "draft_reply": draft,
        "decision": escalation["decision"],
        "decision_reason": escalation["reason"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", required=True)
    ap.add_argument("--message", required=True)
    args = ap.parse_args()

    threads_df = pd.read_csv(args.threads)
    result = run_pipeline(args.message, threads_df)
    print(json.dumps(result, indent=2))
