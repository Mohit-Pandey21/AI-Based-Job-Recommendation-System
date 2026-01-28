from job_recommendation import load_model, recommend_hybrid
from evaluation import evaluate_all_models


def run_inference():
    df, embeddings, tfidf_matrix, job_index, sbert_model = load_model()

    query = "Dancer"

    results = recommend_hybrid(
        title_query=query,
        df=df,
        embeddings=embeddings,
        tfidf_matrix=tfidf_matrix,
        job_index=job_index,
        sbert_model=sbert_model,
        top_k=5
    )

    print("\nRecommended Jobs (Hybrid Model):\n")
    print(results[["title", "company", "industry"]])


def run_evaluation():
    df, embeddings, tfidf_matrix, job_index, sbert_model = load_model()

    print("\nRunning evaluation on all models...\n")

    metrics = evaluate_all_models(
        df=df,
        embeddings=embeddings,
        tfidf_matrix=tfidf_matrix,
        job_index=job_index,
        sample_size=500,
        k=10,
        seed=42
    )

    print("TF-IDF :", metrics["TF-IDF"])
    print("SBERT  :", metrics["SBERT"])
    print("HYBRID :", metrics["HYBRID"])


if __name__ == "__main__":
    # Choose ONE
    # run_inference()
    run_evaluation()
