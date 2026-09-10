"""
Evaluation harness.

Takes a golden eval CSV (data/golden_eval.csv) with columns:
    customer_text, gold_intent, gold_should_escalate, gold_escalate_reason (optional)

Runs the pipeline on each row, then scores:
  - Intent classification accuracy vs gold_intent
  - Escalation decision accuracy vs gold_should_escalate
  - Reply quality via LLM-as-judge rubric (1-5 scale, multiple dimensions)

Also supports comparing the LLM judge's scores against a small set of
human-scored rows (data/human_judge_agreement.csv) to report judge<->human
agreement (Cohen's kappa / correlation) — required by the assignment.

Usage:
    python src/eval_harness.py --threads data/AmazonHelp_threads.csv --golden data/golden_eval.csv
"""
import os
import json
import argparse
import pandas as pd
from anthropic import Anthropic
from sklearn.metrics import accuracy_score, cohen_kappa_score

from agent import run_pipeline

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"

JUDGE_RUBRIC = """Score this customer support reply on 3 dimensions, 1 (poor) to 5 (excellent):
- grounded: does it match how the brand has actually resolved similar issues historically?
- helpful: does it actually address the customer's problem / give a clear next step?
- tone: is it appropriately empathetic and on-brand for a support account?

Customer message: "{message}"
Drafted reply: "{reply}"

Respond ONLY with JSON: {{"grounded": <1-5>, "helpful": <1-5>, "tone": <1-5>, "overall": <1-5>, "notes": "<one sentence>"}}"""


def llm_judge(message: str, reply: str) -> dict:
    prompt = JUDGE_RUBRIC.format(message=message, reply=reply)
    resp = client.messages.create(model=MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt}])
    text = resp.content[0].text.strip().strip("```json").strip("```").strip()
    return json.loads(text)


def trivial_baseline(message: str) -> dict:
    """Baseline 1: always the same canned reply, always auto-handle."""
    return {
        "intent": "other_or_unclear",
        "draft_reply": "Thanks for reaching out! Please DM us your order details and we'll look into it.",
        "decision": "auto_handle",
    }


def simple_baseline(message: str, threads: pd.DataFrame) -> dict:
    """Baseline 2: keyword-rule intent + nearest-neighbor reply, no LLM reasoning."""
    from agent import retrieve_similar
    kw_map = {
        "refund": "refund_or_return_request",
        "damaged": "wrong_or_damaged_item",
        "broken": "wrong_or_damaged_item",
        "late": "delivery_delay_status_check",
        "delayed": "delivery_delay_status_check",
        "never arrived": "order_not_delivered",
        "login": "account_or_login_issue",
        "charged": "billing_or_charge_dispute",
    }
    intent = "other_or_unclear"
    for kw, lab in kw_map.items():
        if kw in message.lower():
            intent = lab
            break
    nearest = retrieve_similar(message, threads, k=1)
    reply = nearest.iloc[0]["brand_reply_text"] if len(nearest) else "Please DM us for help."
    decision = "escalate" if intent in ("billing_or_charge_dispute", "account_or_login_issue") else "auto_handle"
    return {"intent": intent, "draft_reply": reply, "decision": decision}


def run_eval(threads: pd.DataFrame, golden: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in golden.itertuples():
        result = run_pipeline(row.customer_text, threads)
        judge = llm_judge(row.customer_text, result["draft_reply"])
        rows.append({
            "customer_text": row.customer_text,
            "gold_intent": row.gold_intent,
            "pred_intent": result["intent"],
            "gold_should_escalate": row.gold_should_escalate,
            "pred_decision": result["decision"],
            "pred_reason": result["decision_reason"],
            "draft_reply": result["draft_reply"],
            "judge_grounded": judge["grounded"],
            "judge_helpful": judge["helpful"],
            "judge_tone": judge["tone"],
            "judge_overall": judge["overall"],
        })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame):
    intent_acc = accuracy_score(results["gold_intent"], results["pred_intent"])
    pred_escalate_bool = results["pred_decision"] == "escalate"
    gold_escalate_bool = results["gold_should_escalate"].astype(bool)
    escalation_acc = accuracy_score(gold_escalate_bool, pred_escalate_bool)
    avg_judge_overall = results["judge_overall"].mean()

    print(f"Intent classification accuracy: {intent_acc:.3f}")
    print(f"Escalation decision accuracy:    {escalation_acc:.3f}")
    print(f"Avg LLM-judge overall score:     {avg_judge_overall:.2f} / 5")
    return {
        "intent_accuracy": intent_acc,
        "escalation_accuracy": escalation_acc,
        "avg_judge_overall": avg_judge_overall,
    }


def judge_human_agreement(human_scored_path: str):
    """Compares LLM judge overall scores against a human's scores on the same
    subset. REQUIRED by the assignment as evidence the judge is trustworthy.
    Expects columns: customer_text, draft_reply, human_overall (1-5)."""
    df = pd.read_csv(human_scored_path)
    llm_scores, human_scores = [], []
    for row in df.itertuples():
        j = llm_judge(row.customer_text, row.draft_reply)
        llm_scores.append(j["overall"])
        human_scores.append(row.human_overall)
    kappa = cohen_kappa_score(llm_scores, human_scores, weights="quadratic")
    corr = pd.Series(llm_scores).corr(pd.Series(human_scores))
    print(f"LLM-judge vs human: quadratic-weighted kappa={kappa:.3f}, pearson r={corr:.3f}")
    return {"kappa": kappa, "pearson_r": corr}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--human_agreement", default=None, help="Optional path to human-scored subset CSV")
    ap.add_argument("--out", default="eval/results.csv")
    args = ap.parse_args()

    threads_df = pd.read_csv(args.threads)
    golden_df = pd.read_csv(args.golden)

    results = run_eval(threads_df, golden_df)
    results.to_csv(args.out, index=False)
    summary = summarize(results)

    with open("eval/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.human_agreement:
        agreement = judge_human_agreement(args.human_agreement)
        with open("eval/judge_human_agreement.json", "w") as f:
            json.dump(agreement, f, indent=2)
