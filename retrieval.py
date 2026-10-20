"""
retrieval.py - BM25 (from scratch), dense search, and Reciprocal Rank Fusion.

Run the side-by-side demo (needs MONGODB_URI, data/embeddings.npy, data/keys.json):
    python retrieval.py
"""
import os
import re
import json
import math
from collections import Counter, defaultdict

import numpy as np

from embed_jobs import posting_text, dense_search, MODEL_NAME

TOKEN_RE = re.compile(r"[a-z0-9+#]+")


def tokenize(text):
    return TOKEN_RE.findall((text or "").lower())


class BM25:
    """Okapi BM25. Same formula and defaults as rank_bm25.BM25Okapi
    (k1=1.5, b=0.75, epsilon=0.25) so the two can be compared directly."""

    def __init__(self, corpus_tokens, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.n = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float64)
        self.avgdl = self.doc_len.sum() / self.n

        df = Counter()
        self.postings = defaultdict(list)  # term -> [(doc_idx, tf), ...]
        for i, doc in enumerate(corpus_tokens):
            for term, tf in Counter(doc).items():
                df[term] += 1
                self.postings[term].append((i, tf))

        raw = {t: math.log(self.n - f + 0.5) - math.log(f + 0.5) for t, f in df.items()}
        avg_idf = sum(raw.values()) / len(raw)
        floor = self.epsilon * avg_idf
        self.idf = {t: (floor if v < 0 else v) for t, v in raw.items()}

    def get_scores(self, query_tokens):
        scores = np.zeros(self.n, dtype=np.float64)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (tf * (self.k1 + 1) / (tf + norm))
        return scores

    def top_k(self, query_tokens, k=10):
        s = self.get_scores(query_tokens)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]


def rrf(rankings, k=60, top=10):
    """Reciprocal Rank Fusion. rankings: list of lists of doc indices, best first.
    score(d) = sum over lists of 1 / (k + rank), rank starting at 1."""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] += 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda x: -x[1])
    return ordered[:top]


def load_docs_in_embedding_order():
    from pymongo import MongoClient

    db = MongoClient(os.environ["MONGODB_URI"])["jobfinder"]
    with open("data/keys.json") as f:
        keys = json.load(f)
    by_key = {d["dedupe_key"]: d for d in db["jobs_unified"].find({}, {"_id": 0})}
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise SystemExit(f"{len(missing)} embedded rows are missing from jobs_unified; rerun embed_jobs.py")
    return [by_key[k] for k in keys]


def load_all():
    """Load docs, embeddings, BM25 index and the embedding model."""
    from sentence_transformers import SentenceTransformer

    docs = load_docs_in_embedding_order()
    emb = np.load("data/embeddings.npy")
    assert len(docs) == emb.shape[0], "docs and embeddings are out of sync"
    bm25 = BM25([tokenize(posting_text(d)) for d in docs])
    model = SentenceTransformer(MODEL_NAME)
    return docs, emb, bm25, model


def rank_all(query, bm25, emb, model, pool=50):
    """Ranked doc indices (best first) for BM25, dense, and hybrid RRF."""
    b = [i for i, _ in bm25.top_k(tokenize(query), k=pool)]
    qv = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    d = [i for i, _ in dense_search(qv, emb, k=pool)]
    h = [i for i, _ in rrf([b, d], top=pool)]
    return {"bm25": b, "dense": d, "hybrid": h}


def show(label, hits, docs, scores=True):
    print(f"  [{label}]")
    for i, s in hits:
        d = docs[i]
        sc = f"{s:.3f}" if scores else "     "
        print(f"    {sc} | {d['title'][:50]} | {d['company'][:22]} | {d['location'][:26]}")


def main():
    from sentence_transformers import SentenceTransformer

    docs = load_docs_in_embedding_order()
    emb = np.load("data/embeddings.npy")
    assert len(docs) == emb.shape[0], "docs and embeddings are out of sync"
    print(f"loaded {len(docs)} postings, embeddings {emb.shape}")

    bm25 = BM25([tokenize(posting_text(d)) for d in docs])
    model = SentenceTransformer(MODEL_NAME)

    queries = [
        "data analyst internship in Boston",
        "machine learning co-op with Python and PyTorch",
        "remote data engineering intern SQL Spark",
    ]
    for q in queries:
        print(f"\nQUERY: {q}")
        b_hits = bm25.top_k(tokenize(q), k=50)
        qv = model.encode([q], normalize_embeddings=True)[0].astype(np.float32)
        d_hits = dense_search(qv, emb, k=50)
        h_hits = rrf([[i for i, _ in b_hits], [i for i, _ in d_hits]], top=5)

        show("BM25", b_hits[:5], docs)
        show("DENSE", d_hits[:5], docs)
        show("HYBRID (RRF)", h_hits, docs)


if __name__ == "__main__":
    main()
