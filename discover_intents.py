"""
Run this BEFORE finalizing INTENT_SET in agent.py.

Clusters a sample of customer messages (unsupervised) and asks the LLM to
propose short intent labels for each cluster, so your intent taxonomy is
grounded in what's actually in the data rather than guessed upfront.

Usage:
    python src/discover_intents.py --threads data/AmazonHelp_threads.csv --n_clusters 10 --sample 500
"""
import os
import json
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "openai/gpt-oss-120b"


def cluster_messages(texts: list[str], n_clusters: int):
    vec = TfidfVectorizer(max_features=3000, stop_words="english")
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels


def label_cluster(sample_texts: list[str]) -> str:
    prompt = f"""Here are {len(sample_texts)} real customer support messages that were grouped
together by an unsupervised clustering algorithm because they are textually similar.

Messages:
{chr(10).join(f'- {t}' for t in sample_texts[:15])}

Propose a single short snake_case intent label (2-4 words) that captures what this cluster
is about, e.g. "order_not_delivered" or "refund_request". Respond ONLY with the label."""
    resp = client.chat.completions.create(model=MODEL, max_tokens=200, messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", required=True)
    ap.add_argument("--n_clusters", type=int, default=10)
    ap.add_argument("--sample", type=int, default=500)
    args = ap.parse_args()

    df = pd.read_csv(args.threads)
    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)
    texts = df["customer_text"].fillna("").tolist()

    labels = cluster_messages(texts, args.n_clusters)
    df["cluster"] = labels

    print("Proposed intents per cluster (REVIEW these manually before using them):\n")
    proposed = {}
    for c in sorted(set(labels)):
        cluster_texts = df[df["cluster"] == c]["customer_text"].tolist()
        label = label_cluster(cluster_texts)
        proposed[int(c)] = {"label": label, "size": len(cluster_texts), "examples": cluster_texts[:3]}
        print(f"Cluster {c} (n={len(cluster_texts)}): {label}")
        for ex in cluster_texts[:3]:
            print(f"    - {ex[:100]}")
        print()

    with open("data/proposed_intents.json", "w") as f:
        json.dump(proposed, f, indent=2)
    print("Saved to data/proposed_intents.json — manually review, merge/split clusters as needed,")
    print("then update INTENT_SET in src/agent.py.")
