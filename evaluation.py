import numpy as np
import random

# ---------- metrics ----------
def precision_at_k(recommended, relevant_set, k):
    rec = recommended[:k]
    return sum(1 for i in rec if i in relevant_set) / k

def recall_at_k(recommended, relevant_set, k):
    rec = recommended[:k]
    if not relevant_set:
        return 0.0
    return sum(1 for i in rec if i in relevant_set) / len(relevant_set)

def ndcg_at_k(recommended, relevant_set, k):
    rec = recommended[:k]
    dcg = 0.0
    for rank, idx in enumerate(rec, start=1):
        if idx in relevant_set:
            dcg += 1.0 / np.log2(rank + 1)
    # ideal DCG (all relevant at top)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0

# ---------- relevance (matches your earlier logic: Industry OR Position) ----------
def relevant_indices_for_query(df, qidx):
    ind = df.iloc[qidx]["industry"]
    pos = df.iloc[qidx]["position"]
    rel = df[(df["industry"] == ind) | (df["position"] == pos)].index.tolist()
    if qidx in rel:
        rel.remove(qidx)
    return set(rel)

# ---------- recommenders (INDEX-BASED, no title-not-found nonsense) ----------
def recommend_tfidf_idx(qidx, tfidf_matrix, top_k=10):
    # cosine against all rows (sparse -> ok)
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(tfidf_matrix[qidx], tfidf_matrix).flatten()
    scores[qidx] = -1
    return np.argsort(scores)[-top_k:][::-1].tolist()

def recommend_sbert_idx(qidx, embeddings, job_index, top_k=10):
    qvec = embeddings[qidx].reshape(1, -1).astype("float32")
    _, idx = job_index.search(qvec, top_k + 1)
    idx = idx[0].tolist()
    # remove self if present
    idx = [i for i in idx if i != qidx]
    return idx[:top_k]

def recommend_hybrid_idx(qidx, embeddings, tfidf_matrix, job_index, top_k=10, ann_k=50, w_sbert=0.6, w_tfidf=0.4):
    from sklearn.metrics.pairwise import cosine_similarity

    # candidate pool from SBERT ANN
    qvec = embeddings[qidx].reshape(1, -1).astype("float32")
    _, cand = job_index.search(qvec, ann_k + 1)
    cand = cand[0].tolist()
    cand = [i for i in cand if i != qidx]
    if not cand:
        return []

    # SBERT scores within candidates (dot because embeddings should be normalized in training)
    sbert_scores = np.dot(embeddings[cand], embeddings[qidx])

    # TF-IDF scores within candidates
    tfidf_scores = cosine_similarity(tfidf_matrix[qidx], tfidf_matrix[cand]).flatten()

    # normalize both
    def norm(x):
        return (x - x.min()) / (np.ptp(x) + 1e-6)

    final = w_sbert * norm(sbert_scores) + w_tfidf * norm(tfidf_scores)

    order = np.argsort(final)[::-1]
    top = [cand[i] for i in order[:top_k]]
    return top

# ---------- unified evaluation ----------
def evaluate_all_models(df, embeddings, tfidf_matrix, job_index, sample_size=500, k=10, seed=42):
    random.seed(seed)
    indices = random.sample(range(len(df)), min(sample_size, len(df)))

    out = {
        "TF-IDF": {"p": [], "r": [], "n": []},
        "SBERT": {"p": [], "r": [], "n": []},
        "HYBRID": {"p": [], "r": [], "n": []},
    }

    for qidx in indices:
        rel = relevant_indices_for_query(df, qidx)
        if not rel:
            continue

        rec_t = recommend_tfidf_idx(qidx, tfidf_matrix, top_k=k)
        rec_s = recommend_sbert_idx(qidx, embeddings, job_index, top_k=k)
        rec_h = recommend_hybrid_idx(qidx, embeddings, tfidf_matrix, job_index, top_k=k)

        for name, rec in [("TF-IDF", rec_t), ("SBERT", rec_s), ("HYBRID", rec_h)]:
            out[name]["p"].append(precision_at_k(rec, rel, k))
            out[name]["r"].append(recall_at_k(rec, rel, k))
            out[name]["n"].append(ndcg_at_k(rec, rel, k))

    def pack(d):
        return {
            "Precision@K": float(np.mean(d["p"])) if d["p"] else 0.0,
            "Recall@K": float(np.mean(d["r"])) if d["r"] else 0.0,
            "NDCG@K": float(np.mean(d["n"])) if d["n"] else 0.0,
        }

    return {
        "TF-IDF": pack(out["TF-IDF"]),
        "SBERT": pack(out["SBERT"]),
        "HYBRID": pack(out["HYBRID"]),
    }
