"""
Builds a labeling worksheet for the golden eval set.

Samples N diverse real customer messages (stratified so we don't just get
the most common intent over and over), runs the pipeline on each to
pre-fill a SUGGESTED intent/escalation, and writes a CSV for you to review
and correct by hand. Reviewing 200 pre-filled rows is much faster than
labeling 200 from scratch, and is a legitimate, common practice --
just be honest about it in your decision log / sampling note.

Usage:
    python src/build_golden_worksheet.py --threads data/AmazonHelp_threads.csv --n 200

Output: data/golden_eval_WORKSHEET.csv with columns:
    customer_text, suggested_intent, gold_intent (YOU FILL THIS),
    suggested_escalate, gold_should_escalate (YOU FILL THIS),
    gold_escalate_reason (YOU FILL THIS), notes
"""
import argparse
import pandas as pd
from agent import classify_intent, decide_escalation, draft_reply, retrieve_similar


def stratified_sample(threads: pd.DataFrame, n: int) -> pd.DataFrame:
    """Simple length-based stratification as a proxy for message complexity/
    diversity, since we don't have gold intents yet to stratify by. Mixes
    short (likely simple) and long (likely complex/ambiguous) messages."""
    threads = threads.dropna(subset=["customer_text"]).copy()
    threads["len"] = threads["customer_text"].str.len()
    threads = threads.sort_values("len")

    short = threads.iloc[: len(threads) // 3].sample(min(n // 3, len(threads) // 3), random_state=42)
    mid = threads.iloc[len(threads) // 3: 2 * len(threads) // 3].sample(
        min(n // 3, len(threads) // 3), random_state=42
    )
    long = threads.iloc[2 * len(threads) // 3:].sample(
        min(n - len(short) - len(mid), len(threads) // 3), random_state=42
    )
    return pd.concat([short, mid, long]).sample(frac=1, random_state=1).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data/golden_eval_WORKSHEET.csv")
    args = ap.parse_args()

    threads_df = pd.read_csv(args.threads)
    sample = stratified_sample(threads_df, args.n)

    rows = []
    for i, row in enumerate(sample.itertuples()):
        msg = row.customer_text
        try:
            intent_result = classify_intent(msg)
            grounding = retrieve_similar(msg, threads_df, k=3)
            draft = draft_reply(msg, intent_result["intent"], grounding)
            escalation = decide_escalation(msg, intent_result["intent"], draft)
            rows.append({
                "customer_text": msg,
                "suggested_intent": intent_result["intent"],
                "gold_intent": "",  # YOU FILL THIS IN (or copy suggestion if correct)
                "suggested_escalate": escalation["decision"] == "escalate",
                "gold_should_escalate": "",  # YOU FILL THIS IN
                "gold_escalate_reason": "",  # YOU FILL THIS IN
                "notes": "",
            })
        except Exception as e:
            rows.append({
                "customer_text": msg, "suggested_intent": f"ERROR: {e}",
                "gold_intent": "", "suggested_escalate": "", "gold_should_escalate": "",
                "gold_escalate_reason": "", "notes": "pipeline errored, label manually",
            })
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(sample)}...")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df)} rows to {args.out}")
    print("Open this in Excel/Sheets and fill gold_intent + gold_should_escalate for each row.")
    print("Where you agree with the suggestion, just copy it into the gold column -- that's fine,")
    print("but actually READ each message, don't rubber-stamp blindly (some suggestions will be wrong,")
    print("that's the point of building a real eval set).")
