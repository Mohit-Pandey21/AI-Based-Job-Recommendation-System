import os
import re
import json
import pickle
import threading
from typing import Dict, Any, List, Optional
from scipy.sparse import vstack

import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
SBERT_DIR = os.path.join(BASE_DIR, "sbert_model")

RAW_DATA_PATH = os.path.join(BASE_DIR, "Combined_Jobs_Final.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_jobs.csv")

VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.pkl")
EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "job_embeddings.npy")
FAISS_PATH = os.path.join(ARTIFACTS_DIR, "faiss_jobs.index")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (np.ptp(x) + 1e-6)


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

    # Add stable id + active flag
    df = df.reset_index(drop=True)
    df["id"] = df.index.astype(int)
    df["active"] = 1

    df["combined_text"] = (
        df["title"].fillna("") + " " +
        df["position"].fillna("") + " " +
        df["job.description"].fillna("") + " " +
        df["requirements"].fillna("")
    ).apply(clean_text)

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    # TF-IDF (save BOTH vectorizer + matrix)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_text"])

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(tfidf, f)

    with open(TFIDF_MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)

    # SBERT
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

    np.save(EMBEDDINGS_PATH, embeddings)

    # FAISS
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_PATH)


class JobRecommenderEngine:
    """
    Dynamic engine:
    - recommend() works on free-text query
    - add_job() updates embeddings + faiss index live
    - delete_job() is soft delete (active=0)
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.alpha = 0.6

        self.df = pd.read_csv(PROCESSED_DATA_PATH)
        self.df.columns = self.df.columns.str.lower()

        # Backward compatibility if older processed file doesn't have id/active
        if "id" not in self.df.columns:
            self.df = self.df.reset_index(drop=True)
            self.df["id"] = self.df.index.astype(int)
        if "active" not in self.df.columns:
            self.df["active"] = 1

        self.embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
        self.job_index = faiss.read_index(FAISS_PATH)
        self.sbert_model = SentenceTransformer(SBERT_DIR)

        with open(VECTORIZER_PATH, "rb") as f:
            self.tfidf = pickle.load(f)
        with open(TFIDF_MATRIX_PATH, "rb") as f:
            self.tfidf_matrix = pickle.load(f)

    def _save_state(self):
        # Persist df + embeddings + faiss + tfidf_matrix
        self.df.to_csv(PROCESSED_DATA_PATH, index=False)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        faiss.write_index(self.job_index, FAISS_PATH)
        with open(TFIDF_MATRIX_PATH, "wb") as f:
            pickle.dump(self.tfidf_matrix, f)

    def recommend(
        self,
        query_text: str,
        top_k: int = 10,
        ann_k: int = 60,
        city: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_text = (query_text or "").strip()
        if not query_text:
            # default: show some active jobs
            tmp = self.df[self.df["active"] == 1].head(top_k)
            return tmp[["id", "title", "company", "city", "industry","position"]].fillna("").to_dict("records")

        # SBERT query embedding (dynamic)
        q_emb = self.sbert_model.encode([query_text], normalize_embeddings=True).astype("float32")
        _, cand_idx = self.job_index.search(q_emb, ann_k)
        cand_idx = cand_idx[0].tolist()

        # Soft-delete filter
        cand_df = self.df.iloc[cand_idx]
        active_mask = (cand_df["active"].astype(int) == 1).values
        cand_idx = [i for i, ok in zip(cand_idx, active_mask) if ok]
        if not cand_idx:
            return []

        # Optional city filter (simple contains match)
        if city:
            city_l = city.strip().lower()
            cand_df = self.df.iloc[cand_idx]
            mask = cand_df["city"].astype(str).str.lower().str.contains(city_l, na=False).values
            cand_idx = [i for i, ok in zip(cand_idx, mask) if ok]
            if not cand_idx:
                return []

        # Hybrid scoring: SBERT + TF-IDF(query->jobs)
        sbert_scores = np.dot(self.embeddings[cand_idx], q_emb.T).squeeze()

        q_tfidf = self.tfidf.transform([clean_text(query_text)])
        # cosine similarity for sparse matrices: (A·B) / (||A|| ||B||)
        # We'll do dot product because TF-IDF is L2-normalized by default in sklearn
        tfidf_scores = (self.tfidf_matrix[cand_idx] @ q_tfidf.T).toarray().squeeze()

        alpha = getattr(self, "alpha", 0.6)
        final = alpha * normalize(sbert_scores) + (1 - alpha) * normalize(tfidf_scores)
        order = np.argsort(final)[-top_k:][::-1]
        top_rows = self.df.iloc[np.array(cand_idx)[order]]

        return top_rows[["id", "title", "company", "city", "industry","position"]].fillna("").to_dict("records")

    def add_job(self, title: str, company: str, city: str, desc: str) -> int:
        title = (title or "").strip()
        desc = (desc or "").strip()
        if not title or not desc:
            raise ValueError("title and desc are required")

        with self._lock:
            new_id = int(self.df["id"].max()) + 1 if len(self.df) else 0

            combined_text = clean_text(f"{title} {desc}")

            # Build new row (keep schema consistent)
            row = {
                "title": title,
                "company": company or "",
                "city": city or "",
                "position": "",
                "industry": "Unknown",
                "employment.type": "",
                "job.description": desc,
                "requirements": "",
                "combined_text": combined_text,
                "id": new_id,
                "active": 1,
            }

            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

            # Update embeddings + faiss
            emb = self.sbert_model.encode([combined_text], normalize_embeddings=True).astype("float32")
            self.job_index.add(emb)
            self.embeddings = np.vstack([self.embeddings, emb])

            # Update TF-IDF matrix WITHOUT refitting (freeze vocabulary for true dynamic ingestion)
            new_vec = self.tfidf.transform([combined_text]) # uses existing vocab
            if not hasattr(self, "tfidf_matrix") or self.tfidf_matrix is None:
                self.tfidf_matrix = new_vec
            else:
                self.tfidf_matrix = vstack([self.tfidf_matrix, new_vec]).tocsr()

            self._save_state()
            return new_id

    def delete_job(self, job_id: int) -> bool:
        with self._lock:
            mask = self.df["id"].astype(int) == int(job_id)
            if not mask.any():
                return False
            self.df.loc[mask, "active"] = 0
            self._save_state()
            return True


def load_model():
    # Keep old API for compatibility with your existing code
    engine = JobRecommenderEngine()
    return engine.df, engine.embeddings, engine.tfidf_matrix, engine.job_index, engine.sbert_model


if __name__ == "__main__":
    train_and_save_artifacts()
