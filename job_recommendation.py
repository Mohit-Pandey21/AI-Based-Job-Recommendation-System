# =========================
# Imports
# =========================
import os
import re
import pickle
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Paths & Constants
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
SBERT_DIR = os.path.join(BASE_DIR, "sbert_model")

RAW_DATA_PATH = os.path.join(BASE_DIR, "Combined_Jobs_Final.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_jobs.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# =========================
# Utility Functions
# =========================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (np.ptp(x) + 1e-6)

# =========================
# TRAINING (RUN ONCE)
# =========================
def train_and_save_artifacts():
    df = pd.read_csv(RAW_DATA_PATH, engine="python", on_bad_lines="skip")

    df = df[
        [
            "Title",
            "Company",
            "City",
            "Position",
            "Industry",
            "Employment.Type",
            "Job.Description",
            "Requirements",
        ]
    ]

    df.columns = df.columns.str.lower()

    df["combined_text"] = (
        df["title"].fillna("") + " " +
        df["position"].fillna("") + " " +
        df["job.description"].fillna("") + " " +
        df["requirements"].fillna("")
    ).apply(clean_text)

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    # ----- TF-IDF -----
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_text"])
    with open(os.path.join(ARTIFACTS_DIR, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)

    # ----- SBERT -----
    # Prefer local folder if present; otherwise download once and save locally.
    if os.path.isdir(SBERT_DIR) and os.listdir(SBERT_DIR):
        sbert = SentenceTransformer(SBERT_DIR)
    else:
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        sbert.save(SBERT_DIR)

    embeddings = sbert.encode(
        df["combined_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    np.save(os.path.join(ARTIFACTS_DIR, "job_embeddings.npy"), embeddings)

    # ----- FAISS -----
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(ARTIFACTS_DIR, "faiss_jobs.index"))

# =========================
# LOADING
# =========================
def load_model():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df.columns = df.columns.str.lower()

    embeddings = np.load(os.path.join(ARTIFACTS_DIR, "job_embeddings.npy"))

    with open(os.path.join(ARTIFACTS_DIR, "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)

    job_index = faiss.read_index(os.path.join(ARTIFACTS_DIR, "faiss_jobs.index"))

    # Must be offline-safe
    sbert_model = SentenceTransformer(SBERT_DIR)

    return df, embeddings, tfidf_matrix, job_index, sbert_model

# =========================
# TF-IDF ONLY (baseline)
# =========================
def recommend_tfidf(query_title, df, tfidf_matrix, top_k=10):
    qidx = df[df["title"].astype(str).str.lower() == str(query_title).lower()].index
    if len(qidx) == 0:
        return df.sample(top_k)

    qidx = qidx[0]
    scores = cosine_similarity(tfidf_matrix[qidx], tfidf_matrix).flatten()
    scores[qidx] = -1  # remove self
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return df.iloc[top_idx][["title", "company", "city", "industry"]]

# =========================
# SBERT ONLY (baseline)
# =========================
def recommend_sbert(query_title, df, embeddings, job_index, top_k=10):
    qidx = df[df["title"].astype(str).str.lower() == str(query_title).lower()].index
    if len(qidx) == 0:
        return df.sample(top_k)

    qidx = qidx[0]
    query_vec = embeddings[qidx].reshape(1, -1).astype("float32")
    _, idx = job_index.search(query_vec, top_k + 1)
    idx = idx[0][1:]  # remove self
    return df.iloc[idx][["title", "company", "city", "industry"]]

# =========================
# HYBRID (SBERT + TF-IDF)
# =========================
def recommend_hybrid(
    title_query,
    df,
    embeddings,
    tfidf_matrix,
    job_index,
    sbert_model,
    top_k=10,
    ann_k=50
):
    # Use SBERT to get candidate pool (ANN)
    query_emb = sbert_model.encode([title_query], normalize_embeddings=True).astype("float32")
    _, candidate_idx = job_index.search(query_emb, ann_k)
    candidate_idx = candidate_idx[0]

    # Scores within candidates
    sbert_scores = np.dot(embeddings[candidate_idx], query_emb.T).squeeze()

    qidx = df[df["title"].astype(str).str.lower() == str(title_query).lower()].index
    if len(qidx) > 0:
        qidx = qidx[0]
        tfidf_scores = cosine_similarity(
            tfidf_matrix[qidx], tfidf_matrix[candidate_idx]
        ).flatten()
    else:
        tfidf_scores = np.zeros(len(candidate_idx))

    final_scores = 0.6 * normalize(sbert_scores) + 0.4 * normalize(tfidf_scores)
    top_idx = candidate_idx[np.argsort(final_scores)[-top_k:][::-1]]

    return df.iloc[top_idx][["title", "company", "city", "industry"]]

# Backward-compatible name used by evaluation.py/run.py
def recommend_jobs(*args, **kwargs):
    return recommend_hybrid(*args, **kwargs)

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    train_and_save_artifacts()
