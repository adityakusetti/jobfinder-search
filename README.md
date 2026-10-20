# Job & Co-op Finder: search over internship and co-op postings

**Live demo:** YOUR_STREAMLIT_URL_HERE
Hosted on the Streamlit Community Cloud free tier. The app sleeps when idle, so the first load and first search can take 30 seconds or more.

## What this is

A search system over 3,410 deduplicated internship and co-op postings (6,957 raw postings from 4 job APIs, stored in MongoDB Atlas). It compares three ways to search:

- **BM25**, implemented from scratch (keyword matching)
- **Dense retrieval** with the pretrained `all-MiniLM-L6-v2` sentence-transformer (384-dim vectors, searched by cosine similarity)
- **Hybrid**, which merges the BM25 and dense rankings with Reciprocal Rank Fusion (RRF)

The point of the project is the measurement, not the app: each method was scored on a hand-labeled query set.

## Results (15 queries)

| method | Recall@10 | nDCG@10 | MRR@10 |
|---|---|---|---|
| BM25 | 0.417 | 0.518 | 0.629 |
| Dense | 0.595 | 0.719 | 0.806 |
| Hybrid (RRF) | 0.489 | 0.588 | 0.619 |

Dense retrieval scored highest on all three metrics. Equal-weight RRF hybrid did **not** beat dense. My guess, which I have not tested, is that the weaker BM25 list pulls the fused ranking down.

Recall@10 has a ceiling below 1.0 because most queries have more than 10 relevant postings, so nDCG@10 and MRR are the fairer numbers to compare.

## How it was measured

- 15 queries in `queries.txt`, written by me.
- For each query, I pooled the top 10 results of each method, removed repeats, shuffled them, and labeled each posting 1 (relevant) or 0 by hand (`label_sheet.csv`). Postings outside the pool count as not relevant.
- **Labeling rule:** a posting is relevant if it is a data-related internship or co-op in the USA. City, remote and seniority wording in the query were *not* enforced.
- Metrics are computed by `evaluate.py`.

## Limitations

- Small evaluation: 15 queries, one labeler, pooled labels, no confidence intervals yet. Treat the table as a first look, not a final verdict.
- The labeling rule ignores city and remote, so keyword matching on location earns nothing in this table.
- 2,344 of the 3,410 postings (69%) have no description in the data. They are searched on title, company and location only.
- Deduplication is an exact-match key on company, title and location. Near-duplicates remain, for example the same job with a differently written location.
- The corpus includes non-US postings. The app's "US only" checkbox is a string heuristic and has known misses.
- The data is a snapshot. The scheduled collector is currently paused.

## Pipeline

1. **Collection** (separate repo, GitHub Actions): 4 job APIs into MongoDB Atlas raw collections.
2. `normalize.py` merges the 4 raw collections into one `jobs_unified` collection and removes duplicates (6,957 raw to 3,410 unified).
3. `embed_jobs.py` embeds each posting (title, company, location, first 1,000 characters of any description). 3,410 postings took 4.2 seconds on a laptop. Output: `data/embeddings.npy` and `data/keys.json`.
4. `retrieval.py` holds BM25, dense search and RRF.
5. `evaluate.py` scores the methods against `label_sheet.csv`.
6. `app.py` is the Streamlit search page.

## Testing

BM25 matches both a plain-loop version of the textbook formula and `rank_bm25.BM25Okapi` on 6 test queries. Run:

```bash
python test_bm25.py
python test_eval.py
```

## Run it yourself

```bash
pip install -r requirements.txt
export MONGODB_URI="YOUR_URI_HERE"
streamlit run app.py
```

The app reads postings from MongoDB and the vectors from `data/`. If you rebuild `jobs_unified`, re-run `embed_jobs.py` so the keys still match.

## Next steps

- Confidence intervals and a per-query breakdown of the results
- 15 new held-out queries to test weighted fusion and a better US filter
- Fix quota counting in the collector and add location filters at the API level
