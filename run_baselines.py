"""
Computes trivial and simple baseline accuracy against the golden eval set.
Neither baseline calls the LLM API, so this runs fast and doesn't touch
your Groq quota -- useful when the full pipeline eval has used up rate limits.

Usage:
    python src/run_baselines.py --threads data/AmazonHelp_threads.csv --golden data/golden_eval.csv
"""
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from eval_harness import trivial_baseline, simple_baseline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", required=True)
    ap.add_argument("--golden", required=True)
    args = ap.parse_args()

    threads_df = pd.read_csv(args.threads)
    golden_df = pd.read_csv(args.golden)

    trivial_intents, simple_intents = [], []
    trivial_escalate, simple_escalate = [], []

    for row in golden_df.itertuples():
        t = trivial_baseline(row.customer_text)
        s = simple_baseline(row.customer_text, threads_df)
        trivial_intents.append(t["intent"])
        simple_intents.append(s["intent"])
        trivial_escalate.append(t["decision"] == "escalate")
        simple_escalate.append(s["decision"] == "escalate")

    gold_intent = golden_df["gold_intent"].tolist()
    gold_escalate = golden_df["gold_should_escalate"].astype(bool).tolist()

    print(f"On all {len(golden_df)} golden examples (no API calls needed):\n")
    print(f"Trivial baseline  -- intent accuracy: {accuracy_score(gold_intent, trivial_intents):.3f}  "
          f"escalation accuracy: {accuracy_score(gold_escalate, trivial_escalate):.3f}")
    print(f"Simple baseline   -- intent accuracy: {accuracy_score(gold_intent, simple_intents):.3f}  "
          f"escalation accuracy: {accuracy_score(gold_escalate, simple_escalate):.3f}")
