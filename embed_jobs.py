"""
embed_jobs.py - embed every posting in `jobs_unified` and test dense search.

Run from the repo root with MONGODB_URI set:
    python embed_jobs.py

Outputs (local, keep out of git):
    data/embeddings.npy   float32, shape (N, 384), L2-normalized
    data/keys.json        dedupe_key for each row, same order as embeddings
"""
import os
import json
import time

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
DESC_CHARS = 1000


def posting_text(d):
    parts = [d.get("title", ""), d.get("company", ""), d.get("location", "")]
    desc = (d.get("description") or "")[:DESC_CHARS]
    if desc:
        parts.append(desc)
    return ". ".join(p.strip() for p in parts if p and p.strip())


def dense_search(query_vec, matrix, k=5):
    """Vectors are L2-normalized, so dot product == cosine similarity."""
    scores = matrix @ query_vec
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]


def main():
    from pymongo import MongoClient
    from sentence_transformers import SentenceTransformer

    db = MongoClient(os.environ["MONGODB_URI"])["jobfinder"]
    docs = list(db["jobs_unified"].find({}, {"_id": 0}))
    print("postings to embed:", len(docs))

    texts = [posting_text(d) for d in docs]
    model = SentenceTransformer(MODEL_NAME)

    t0 = time.time()
    emb = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    secs = time.time() - t0
    print(f"embedded {emb.shape} in {secs:.1f}s ({len(docs)/secs:.0f} postings/s)")

    os.makedirs("data", exist_ok=True)
    np.save("data/embeddings.npy", emb)
    with open("data/keys.json", "w") as f:
        json.dump([d["dedupe_key"] for d in docs], f)
    print(f"saved data/embeddings.npy ({emb.nbytes/1e6:.1f} MB) and data/keys.json")

    queries = [
        "data analyst internship in Boston",
        "machine learning co-op with Python and PyTorch",
        "remote data engineering intern SQL Spark",
    ]
    for q in queries:
        qv = model.encode([q], normalize_embeddings=True)[0].astype(np.float32)
        print(f"\nQUERY: {q}")
        for i, s in dense_search(qv, emb):
            d = docs[i]
            print(f"  {s:.3f} | {d['title'][:55]} | {d['company'][:25]} | {d['location'][:30]}")


if __name__ == "__main__":
    main()
