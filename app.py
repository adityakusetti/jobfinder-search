"""
app.py - small search page over jobs_unified.

    pip install streamlit
    export MONGODB_URI="YOUR_URI_HERE"
    streamlit run app.py

Needs data/embeddings.npy and data/keys.json (made by embed_jobs.py).
"""
import os

import streamlit as st

from filters import is_us_location
from retrieval import load_all, rank_all

st.set_page_config(page_title="Job & Co-op Finder", layout="wide")

# On Streamlit Community Cloud the connection string lives in st.secrets,
# not in a shell variable. Locally the env var is used as before.
if "MONGODB_URI" not in os.environ:
    try:
        os.environ["MONGODB_URI"] = st.secrets["MONGODB_URI"]
    except Exception:
        pass


@st.cache_resource(show_spinner="Loading postings, index and model...")
def get_engine():
    return load_all()


METHOD_LABELS = {
    "dense": "Dense (best in my 15-query eval)",
    "hybrid": "Hybrid (BM25 + dense, RRF)",
    "bm25": "BM25 (keyword)",
}

st.title("Job & Co-op Finder")
st.caption("Search over deduplicated postings from 4 job APIs. "
           "Search quality was measured on a small hand-labeled query set; see the sidebar.")

with st.sidebar:
    st.header("Options")
    method = st.radio("Search method", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
    us_only = st.checkbox("US locations only (string heuristic)", value=False)
    k = st.slider("Results", 5, 30, 10)
    st.divider()
    st.subheader("Evaluation")
    if os.path.exists("eval_results.md"):
        st.markdown(open("eval_results.md", encoding="utf-8").read())
        st.caption("15 queries, one labeler, pooled labels. Starter-size result.")
    else:
        st.write("Run `python evaluate.py` to create eval_results.md.")

query = st.text_input("What are you looking for?", placeholder="e.g. data analyst co-op Boston")

if query:
    docs, emb, bm25, model = get_engine()
    pool = 400 if us_only else 50
    ranking = rank_all(query, bm25, emb, model, pool=pool)[method]

    hits = []
    for i in ranking:
        d = docs[i]
        if us_only and not is_us_location(d.get("location", "")):
            continue
        hits.append(d)
        if len(hits) == k:
            break

    if not hits:
        st.warning("No results. Try different words or turn off the US filter.")
    for rank, d in enumerate(hits, start=1):
        title = d["title"]
        if d.get("url"):
            title = f"[{title}]({d['url']})"
        st.markdown(f"**{rank}. {title}**")
        meta = f"{d['company']} | {d.get('location') or 'location not listed'} | {d['source']}"
        if not d.get("description"):
            meta += " | no description in the data"
        st.caption(meta)
        if d.get("description"):
            st.write(d["description"][:240].strip() + "...")
