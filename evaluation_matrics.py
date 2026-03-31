import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import jensenshannon

from job_recommendation import JobRecommenderEngine

LABELS_CSV = "manual_labels.csv"
K = 10


# -----------------------------
# Metrics
# -----------------------------

def compute_hhi(distribution):
    probs = np.array(list(distribution.values()))
    probs = probs / probs.sum()
    return np.sum(probs ** 2)


def compute_gini(distribution):
    values = np.array(list(distribution.values()))
    values = np.sort(values)

    n = len(values)
    cumulative = np.cumsum(values)

    if cumulative[-1] == 0:
        return 0.0

    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini


def compute_js_divergence(p_dist, q_dist):
    keys = set(p_dist.keys()).union(set(q_dist.keys()))

    p = np.array([p_dist.get(k, 0) for k in keys], dtype=float)
    q = np.array([q_dist.get(k, 0) for k in keys], dtype=float)

    if p.sum() == 0 or q.sum() == 0:
        return 0.0

    p = p / p.sum()
    q = q / q.sum()

    return jensenshannon(p, q)


# -----------------------------
# Core evaluation
# -----------------------------

def get_recommendation_distribution(alpha, attribute):
    engine = JobRecommenderEngine()
    engine.alpha = alpha

    df_queries = pd.read_csv(LABELS_CSV)
    df_queries = df_queries[df_queries["query"].notna()]

    counter = Counter()

    for _, row in df_queries.iterrows():
        query = str(row["query"]).strip()
        if not query:
            continue

        recs = engine.recommend(query_text=query, top_k=K)

        for r in recs:
            val = str(r.get(attribute, "")).strip()
            if val and val != "nan":
                counter[val] += 1

    return counter


def get_dataset_distribution(attribute):
    engine = JobRecommenderEngine()
    df = engine.df.copy()

    counter = Counter()

    for _, row in df.iterrows():
        val = str(row.get(attribute, "")).strip()
        if val and val != "nan":
            counter[val] += 1

    return counter


# -----------------------------
# Main computation
# -----------------------------

def evaluate_bias():
    attributes = ["industry", "city", "position"]
    models = [
        ("TF-IDF", 0.0),
        ("Hybrid", 0.1),
        ("SBERT", 1.0)
    ]

    results = []

    for attr in attributes:
        print(f"\nProcessing attribute: {attr}")

        dataset_dist = get_dataset_distribution(attr)

        for model_name, alpha in models:
            print(f"  Model: {model_name} (alpha={alpha})")

            rec_dist = get_recommendation_distribution(alpha, attr)

            hhi = compute_hhi(rec_dist)
            gini = compute_gini(rec_dist)
            js = compute_js_divergence(rec_dist, dataset_dist)

            results.append({
                "Attribute": attr.capitalize(),
                "Alpha": alpha,
                "Model": model_name,
                "HHI": round(hhi, 4),
                "Gini": round(gini, 4),
                "JS Divergence": round(js, 4)
            })

    return results


def print_table(results):
    print("\nFinal Bias Table:\n")
    print("{:<10} {:<8} {:<10} {:<10} {:<10} {:<15}".format(
        "Attribute", "Alpha", "Model", "HHI", "Gini", "JS Divergence"
    ))
    print("-" * 75)

    for r in results:
        print("{:<10} {:<8} {:<10} {:<10} {:<10} {:<15}".format(
            r["Attribute"],
            r["Alpha"],
            r["Model"],
            r["HHI"],
            r["Gini"],
            r["JS Divergence"]
        ))


def main():
    results = evaluate_bias()
    print_table(results)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("results/bias_results.csv", index=False)
    print("\nSaved to results/bias_results.csv")


if __name__ == "__main__":
    main()