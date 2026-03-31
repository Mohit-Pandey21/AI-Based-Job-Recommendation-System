"""
run.py

One entry-point for:
1) Inference (text query like your website)
2) Deployed-system evaluation (Precision@K / Recall@K / NDCG@K) using self-query
3) Optional legacy/proxy evaluation comparing TF-IDF vs SBERT vs HYBRID

Usage examples:
  python run.py --mode infer --query "python sql data analyst" --top_k 10 --city "delhi"
  python run.py --mode eval_deployed --sample_size 300 --k 10
  python run.py --mode eval_proxy --sample_size 300 --k 10
"""

import argparse
import json
from typing import Optional


def run_infer(query: str, top_k: int, city: Optional[str] = None):
    from job_recommendation import JobRecommenderEngine

    engine = JobRecommenderEngine()
    results = engine.recommend(query_text=query, top_k=top_k, city=city)

    print("\n=== Recommendations ===\n")
    if not results:
        print("No results.")
        return

    for i, r in enumerate(results, start=1):
        print(f"{i:02d}. {r.get('title','')} | {r.get('company','')} | {r.get('city','')} | {r.get('industry','')}")


def run_eval_deployed(sample_size: int, k: int, seed: int):
    from evaluation import evaluate_deployed_system_self_query

    metrics = evaluate_deployed_system_self_query(sample_size=sample_size, k=k, seed=seed)
    print("\n=== Deployed-System Evaluation (Self-Query) ===")
    print(json.dumps(metrics, indent=2))


def run_eval_proxy(sample_size: int, k: int, seed: int, alpha: float):
    from job_recommendation import load_model
    from evaluation import evaluate_all_models

    df, embeddings, tfidf_matrix, job_index, sbert_model = load_model()

    metrics = evaluate_all_models(
        df=df,
        embeddings=embeddings,
        tfidf_matrix=tfidf_matrix,
        job_index=job_index,
        sample_size=sample_size,
        k=k,
        seed=seed,
        alpha=alpha,
    )

    print("\n=== Proxy Evaluation (Model Comparison) ===")
    print(json.dumps(metrics, indent=2))


def run_add_job(title: str, company: str, city: str, desc: str):
    from job_recommendation import JobRecommenderEngine

    engine = JobRecommenderEngine()
    new_id = engine.add_job(title=title, company=company, city=city, desc=desc)

    print("\n=== Job Added Dynamically ===")
    print(f"New Job ID: {new_id}")


def run_delete_job(job_id: int):
    from job_recommendation import JobRecommenderEngine

    engine = JobRecommenderEngine()
    ok = engine.delete_job(job_id)

    print("\n=== Delete Job (Soft Delete) ===")
    print("Status:", "success" if ok else "not_found")


def main():
    parser = argparse.ArgumentParser(description="JobNexus Runner (Inference + Evaluation)")
    parser.add_argument(
        "--mode",
        choices=["infer", "eval_deployed", "eval_proxy", "add_job", "delete_job"],
        default="infer",
        help="What to run"
    )

    # Common
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10, help="K for metrics")
    parser.add_argument("--sample_size", type=int, default=300, help="Evaluation sample size")

    # Inference
    parser.add_argument("--query", type=str, default="python sql data analyst", help="Free-text query")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K recommendations")
    parser.add_argument("--city", type=str, default=None, help="Optional city filter")

    # Proxy eval
    parser.add_argument("--alpha", type=float, default=0.6, help="Hybrid weight for SBERT in proxy eval")

    # Add/delete job
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--company", type=str, default="")
    parser.add_argument("--job_city", type=str, default="")
    parser.add_argument("--desc", type=str, default="")
    parser.add_argument("--job_id", type=int, default=-1)

    args = parser.parse_args()

    if args.mode == "infer":
        run_infer(query=args.query, top_k=args.top_k, city=args.city)

    elif args.mode == "eval_deployed":
        run_eval_deployed(sample_size=args.sample_size, k=args.k, seed=args.seed)

    elif args.mode == "eval_proxy":
        run_eval_proxy(sample_size=args.sample_size, k=args.k, seed=args.seed, alpha=args.alpha)

    elif args.mode == "add_job":
        if not args.title or not args.desc:
            raise SystemExit("For add_job, --title and --desc are required.")
        run_add_job(title=args.title, company=args.company, city=args.job_city, desc=args.desc)

    elif args.mode == "delete_job":
        if args.job_id < 0:
            raise SystemExit("For delete_job, --job_id is required.")
        run_delete_job(job_id=args.job_id)


if __name__ == "__main__":
    main()
