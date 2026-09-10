"""
Loads the raw Kaggle 'Customer Support on Twitter' CSV (twcs.csv), filters to
one brand, and reconstructs (customer_message -> brand_reply) pairs using the
response_tweet_id / in_response_to_tweet_id thread links.

Usage:
    python src/build_dataset.py --brand AmazonHelp --sample 5000

Output:
    data/{brand}_threads.csv  with columns:
        customer_tweet_id, customer_text, brand_tweet_id, brand_reply_text,
        created_at
"""
import argparse
import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    return df


def build_brand_threads(df: pd.DataFrame, brand_handle: str, sample: int | None = None) -> pd.DataFrame:
    # Brand tweets: outbound (inbound == False) and authored by the brand handle
    brand_tweets = df[(df["inbound"] == "False") & (df["author_id"] == brand_handle)].copy()

    # Customer tweets: inbound == True
    cust_tweets = df[df["inbound"] == "True"].copy()
    cust_tweets = cust_tweets.set_index("tweet_id")

    rows = []
    for _, brow in brand_tweets.iterrows():
        parent_id = brow.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id == "":
            continue
        if parent_id not in cust_tweets.index:
            continue
        crow = cust_tweets.loc[parent_id]
        rows.append({
            "customer_tweet_id": parent_id,
            "customer_text": crow["text"],
            "brand_tweet_id": brow["tweet_id"],
            "brand_reply_text": brow["text"],
            "created_at": crow["created_at"],
        })

    out = pd.DataFrame(rows)
    if sample and len(out) > sample:
        out = out.sample(sample, random_state=42).reset_index(drop=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", default="data/twcs.csv")
    ap.add_argument("--brand", default="AmazonHelp", help="Author handle of the brand, e.g. AmazonHelp")
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load_raw(args.raw_path)
    threads = build_brand_threads(df, args.brand, args.sample)
    out_path = args.out or f"data/{args.brand}_threads.csv"
    threads.to_csv(out_path, index=False)
    print(f"Wrote {len(threads)} customer->brand reply pairs to {out_path}")
