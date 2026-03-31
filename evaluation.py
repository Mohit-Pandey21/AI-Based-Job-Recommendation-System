"""
evaluation.py

What this file provides (in order of usefulness for IEEE):
1) Manual-labeled query evaluation (REAL evaluation): evaluate_manual_labeled_queries()
2) Alpha sweep on manual labels: alpha_sweep_manual_labels()
3) Statistical significance testing on manual labels:
   - collect_manual_query_metrics()
   - paired_randomization_test()
   - significance_compare_manual()
4) 5-fold cross-validation on manual labels:
   - cross_validate_alpha()
5) Proxy evaluation (heuristic): evaluate_proxy_relevance()
6) Self-query sanity check (NOT real recommendation evaluation): evaluate_self_query_hit()

Outputs:
- CSV results into --out_dir
- PNG plots (alpha curves) into --out_dir
- CSV significance test results into --out_dir
- CSV cross-validation results into --out_dir
"""

from __future__ import annotations

import os
import csv
import math
import argparse
import tempfile
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold

from job_recommendation import JobRecommenderEngine


# -----------------------------
# Metrics
# -----------------------------

def precision_at_k(predicted: List[int], relevant: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for x in top_k if x in relevant)
    return hits / float(k)


def recall_at_k(predicted: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for x in top_k if x in relevant)
    return hits / float(len(relevant))


def ndcg_at_k(predicted: List[int], relevant: Set[int], k: int) -> float:
    """
    Binary relevance NDCG@k.
    """
    if k <= 0:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(predicted[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)  # rank = i+1

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return (dcg / idcg) if idcg > 0 else 0.0


def _bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Returns (mean, lower, upper) for bootstrap CI.
    """
    if not values:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    boot_means = []
    n = len(arr)

    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(sample.mean()))

    boot_means.sort()
    lo = boot_means[int((alpha / 2) * n_boot)]
    hi = boot_means[int((1 - alpha / 2) * n_boot) - 1]
    return mean, float(lo), float(hi)


# -----------------------------
# Helpers
# -----------------------------

def _ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def _save_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _save_alpha_plot(xs: List[float], ys: List[float], out_path: str, title: str, ylabel: str) -> None:
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("alpha (SBERT weight)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def safe_str(x) -> str:
    """
    Converts values (including NaN floats) into safe stripped strings.
    """
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def _parse_relevant_ids(raw: str) -> Set[int]:
    """
    Supports:
      - pipe separated: "12|55|98"
      - comma separated: "12,55,98"
      - space separated: "12 55 98"
    """
    if not raw:
        return set()
    raw = raw.strip()
    sep = "|" if "|" in raw else ("," if "," in raw else None)
    parts = raw.split(sep) if sep else raw.split()
    out = set()
    for p in parts:
        p = p.strip()
        if p.isdigit():
            out.add(int(p))
    return out


def _extract_ids(recs: List[Dict]) -> List[int]:
    out = []
    for r in recs:
        if "id" in r:
            try:
                out.append(int(r["id"]))
            except Exception:
                pass
    return out


# -----------------------------
# Manual query-level collection
# -----------------------------

def collect_manual_query_metrics(
    labels_csv: str,
    k: int = 10,
    alpha: float = 0.6,
) -> List[Dict]:
    """
    Collect per-query metrics for manual-labeled evaluation.
    Useful for paired significance testing.

    Returns one row per evaluated query with:
      query, relevant_count, precision_at_k, recall_at_k, ndcg_at_k
    """
    engine = JobRecommenderEngine()
    engine.alpha = float(alpha)

    rows_out: List[Dict] = []

    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            query = safe_str(row.get("query"))
            rel_raw = safe_str(row.get("relevant_ids"))

            if not query or not rel_raw:
                continue

            relevant = _parse_relevant_ids(rel_raw)
            if not relevant:
                continue

            recs = engine.recommend(query_text=query, top_k=k)
            pred_ids = _extract_ids(recs)

            p = precision_at_k(pred_ids, relevant, k)
            r = recall_at_k(pred_ids, relevant, k)
            n = ndcg_at_k(pred_ids, relevant, k)

            rows_out.append({
                "query": query,
                "relevant_count": len(relevant),
                "precision_at_k": p,
                "recall_at_k": r,
                "ndcg_at_k": n,
            })

    return rows_out


# -----------------------------
# 1) IEEE-grade evaluation: manual labeled queries
# -----------------------------

def evaluate_manual_labeled_queries(
    labels_csv: str,
    k: int = 10,
    alpha: float = 0.6,
    n_boot: int = 0,
    seed: int = 42,
) -> Dict[str, float]:
    """
    CSV format (required):
      query,relevant_ids

    Example row:
      data analyst python sql,120|455|981

    Returns mean metrics (+ optional bootstrap CI if n_boot > 0).

    Also returns:
      TotalRows, SkippedEmpty, SkippedNoRelevant, EvaluatedRows
    """
    engine = JobRecommenderEngine()
    engine.alpha = float(alpha)

    p_list, r_list, n_list = [], [], []

    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)
    skipped_empty = 0
    skipped_no_relevant = 0

    for row in rows:
        query = safe_str(row.get("query"))
        rel_raw = safe_str(row.get("relevant_ids"))

        if not query or not rel_raw:
            skipped_empty += 1
            continue

        relevant = _parse_relevant_ids(rel_raw)
        if not relevant:
            skipped_no_relevant += 1
            continue

        recs = engine.recommend(query_text=query, top_k=k)
        pred_ids = _extract_ids(recs)

        p_list.append(precision_at_k(pred_ids, relevant, k))
        r_list.append(recall_at_k(pred_ids, relevant, k))
        n_list.append(ndcg_at_k(pred_ids, relevant, k))

    result: Dict[str, float] = {
        "Alpha": float(alpha),
        "K": float(k),
        "Queries": float(len(p_list)),
        "EvaluatedRows": float(len(p_list)),
        "TotalRows": float(total_rows),
        "SkippedEmpty": float(skipped_empty),
        "SkippedNoRelevant": float(skipped_no_relevant),
        "Precision@K": float(np.mean(p_list)) if p_list else 0.0,
        "Recall@K": float(np.mean(r_list)) if r_list else 0.0,
        "NDCG@K": float(np.mean(n_list)) if n_list else 0.0,
    }

    if n_boot and p_list:
        p_mean, p_lo, p_hi = _bootstrap_ci(p_list, n_boot=n_boot, seed=seed)
        r_mean, r_lo, r_hi = _bootstrap_ci(r_list, n_boot=n_boot, seed=seed)
        n_mean, n_lo, n_hi = _bootstrap_ci(n_list, n_boot=n_boot, seed=seed)

        result["Precision@K"] = p_mean
        result["Recall@K"] = r_mean
        result["NDCG@K"] = n_mean
        result["Precision_CI_L"] = p_lo
        result["Precision_CI_U"] = p_hi
        result["Recall_CI_L"] = r_lo
        result["Recall_CI_U"] = r_hi
        result["NDCG_CI_L"] = n_lo
        result["NDCG_CI_U"] = n_hi

    return result


def alpha_sweep_manual_labels(
    labels_csv: str,
    alphas: Optional[List[float]] = None,
    k: int = 10,
    out_dir: str = "results",
    n_boot: int = 0,
    seed: int = 42,
) -> str:
    """
    Alpha sweep using REAL manual-labeled queries.
    """
    _ensure_out_dir(out_dir)

    if alphas is None:
        alphas = [round(x, 2) for x in np.linspace(0.0, 1.0, 11).tolist()]

    rows = []
    printed_counts = False

    for a in alphas:
        m = evaluate_manual_labeled_queries(
            labels_csv=labels_csv,
            k=k,
            alpha=a,
            n_boot=n_boot,
            seed=seed
        )
        rows.append(m)

        if not printed_counts:
            print(
                f"Manual label file rows: total={int(m['TotalRows'])}, "
                f"skipped_empty={int(m['SkippedEmpty'])}, "
                f"skipped_no_relevant={int(m['SkippedNoRelevant'])}, "
                f"evaluated={int(m['EvaluatedRows'])}"
            )
            printed_counts = True

        print(
            f"alpha={a:.2f} -> P@{k}={m['Precision@K']:.4f}, "
            f"R@{k}={m['Recall@K']:.4f}, "
            f"NDCG@{k}={m['NDCG@K']:.4f}"
        )

    csv_path = os.path.join(out_dir, "alpha_sweep_manual_labels.csv")
    _save_csv(csv_path, rows)

    xs = [float(r["Alpha"]) for r in rows]
    p = [float(r["Precision@K"]) for r in rows]
    r_ = [float(r["Recall@K"]) for r in rows]
    n = [float(r["NDCG@K"]) for r in rows]

    _save_alpha_plot(xs, n, os.path.join(out_dir, "alpha_vs_ndcg_manual.png"), f"Alpha vs NDCG@{k} (Manual Labels)", f"NDCG@{k}")
    _save_alpha_plot(xs, p, os.path.join(out_dir, "alpha_vs_precision_manual.png"), f"Alpha vs Precision@{k} (Manual Labels)", f"Precision@{k}")
    _save_alpha_plot(xs, r_, os.path.join(out_dir, "alpha_vs_recall_manual.png"), f"Alpha vs Recall@{k} (Manual Labels)", f"Recall@{k}")

    best = max(rows, key=lambda t: float(t["NDCG@K"]))
    print(f"\nBEST (Manual) by NDCG@{k}: alpha={best['Alpha']:.2f}, NDCG@{k}={best['NDCG@K']:.4f}")
    return csv_path


# -----------------------------
# 1B) Statistical significance
# -----------------------------

def paired_randomization_test(
    scores_a: List[float],
    scores_b: List[float],
    n_perm: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Paired randomization test (approximate randomization).

    Returns:
      observed_mean_diff, p_value
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")

    if len(scores_a) == 0:
        return 0.0, 1.0

    rng = np.random.default_rng(seed)

    arr_a = np.array(scores_a, dtype=float)
    arr_b = np.array(scores_b, dtype=float)

    observed = float(np.mean(arr_a - arr_b))
    extreme_count = 0

    for _ in range(n_perm):
        swap_mask = rng.random(len(arr_a)) < 0.5
        perm_a = np.where(swap_mask, arr_b, arr_a)
        perm_b = np.where(swap_mask, arr_a, arr_b)
        diff = float(np.mean(perm_a - perm_b))

        if abs(diff) >= abs(observed):
            extreme_count += 1

    p_value = (extreme_count + 1) / float(n_perm + 1)
    return observed, p_value


def significance_compare_manual(
    labels_csv: str,
    alpha_a: float = 0.1,
    alpha_b: float = 0.0,
    k: int = 10,
    metric: str = "ndcg",
    n_perm: int = 10000,
    seed: int = 42,
    out_dir: str = "results",
) -> str:
    """
    Compare two alpha settings on manual-labeled queries using paired randomization.

    metric options: precision, recall, ndcg
    """
    _ensure_out_dir(out_dir)

    rows_a = collect_manual_query_metrics(labels_csv=labels_csv, k=k, alpha=alpha_a)
    rows_b = collect_manual_query_metrics(labels_csv=labels_csv, k=k, alpha=alpha_b)

    if len(rows_a) != len(rows_b):
        raise ValueError("Mismatch in evaluated query counts between systems")

    metric_key_map = {
        "precision": "precision_at_k",
        "recall": "recall_at_k",
        "ndcg": "ndcg_at_k",
    }

    if metric not in metric_key_map:
        raise ValueError("metric must be one of: precision, recall, ndcg")

    key = metric_key_map[metric]

    scores_a = [float(r[key]) for r in rows_a]
    scores_b = [float(r[key]) for r in rows_b]

    observed_diff, p_value = paired_randomization_test(
        scores_a=scores_a,
        scores_b=scores_b,
        n_perm=n_perm,
        seed=seed
    )

    mean_a = float(np.mean(scores_a)) if scores_a else 0.0
    mean_b = float(np.mean(scores_b)) if scores_b else 0.0

    result_rows = [{
        "metric": metric,
        "k": k,
        "alpha_a": alpha_a,
        "alpha_b": alpha_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "observed_mean_diff": observed_diff,
        "p_value": p_value,
        "n_queries": len(scores_a),
        "n_permutations": n_perm,
        "seed": seed,
    }]

    out_path = os.path.join(out_dir, f"significance_{metric}_a{alpha_a}_vs_b{alpha_b}.csv")
    _save_csv(out_path, result_rows)

    print("\nStatistical Significance Test")
    print("--------------------------------")
    print(f"Metric           : {metric}@{k}")
    print(f"System A alpha   : {alpha_a}")
    print(f"System B alpha   : {alpha_b}")
    print(f"Queries compared : {len(scores_a)}")
    print(f"Mean A           : {mean_a:.6f}")
    print(f"Mean B           : {mean_b:.6f}")
    print(f"Mean Diff (A-B)  : {observed_diff:.6f}")
    print(f"p-value          : {p_value:.6f}")
    print(f"Saved            : {out_path}")

    return out_path


# -----------------------------
# 1C) Cross-validation
# -----------------------------

def cross_validate_alpha(
    labels_csv: str,
    k: int = 10,
    folds: int = 5,
    alphas: Optional[List[float]] = None,
    seed: int = 42,
    n_boot: int = 0,
    out_dir: str = "results",
) -> str:
    """
    5-fold cross-validation on queries:
    - tune alpha on train folds
    - evaluate best alpha on held-out fold
    - average across folds
    """
    _ensure_out_dir(out_dir)

    if alphas is None:
        alphas = [round(x, 2) for x in np.linspace(0.0, 1.0, 11).tolist()]

    df = pd.read_csv(labels_csv)

    if "query" not in df.columns or "relevant_ids" not in df.columns:
        raise ValueError("labels_csv must contain columns: query,relevant_ids")

    # Keep only valid rows for CV
    valid_mask = df["query"].notna() & df["relevant_ids"].notna()
    df = df[valid_mask].copy()

    if len(df) < folds:
        raise ValueError(f"Not enough valid queries ({len(df)}) for {folds}-fold CV")

    df["query"] = df["query"].astype(str).str.strip()
    df["relevant_ids"] = df["relevant_ids"].astype(str).str.strip()
    df = df[(df["query"] != "") & (df["relevant_ids"] != "")].copy()

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    fold_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        best_alpha = None
        best_train_ndcg = -1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, f"train_fold_{fold_id}.csv")
            test_path = os.path.join(tmpdir, f"test_fold_{fold_id}.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            # Tune alpha on training fold
            for a in alphas:
                train_res = evaluate_manual_labeled_queries(
                    labels_csv=train_path,
                    k=k,
                    alpha=a,
                    n_boot=0,
                    seed=seed
                )

                train_ndcg = float(train_res["NDCG@K"])
                if train_ndcg > best_train_ndcg:
                    best_train_ndcg = train_ndcg
                    best_alpha = a

            # Evaluate best alpha on test fold
            test_res = evaluate_manual_labeled_queries(
                labels_csv=test_path,
                k=k,
                alpha=float(best_alpha),
                n_boot=n_boot,
                seed=seed
            )

        row = {
            "Fold": fold_id,
            "TrainQueries": len(train_df),
            "TestQueries": len(test_df),
            "BestAlpha": float(best_alpha),
            "TrainNDCG@K": float(best_train_ndcg),
            "TestPrecision@K": float(test_res["Precision@K"]),
            "TestRecall@K": float(test_res["Recall@K"]),
            "TestNDCG@K": float(test_res["NDCG@K"]),
        }

        if "Precision_CI_L" in test_res:
            row["TestPrecision_CI_L"] = float(test_res["Precision_CI_L"])
            row["TestPrecision_CI_U"] = float(test_res["Precision_CI_U"])
            row["TestRecall_CI_L"] = float(test_res["Recall_CI_L"])
            row["TestRecall_CI_U"] = float(test_res["Recall_CI_U"])
            row["TestNDCG_CI_L"] = float(test_res["NDCG_CI_L"])
            row["TestNDCG_CI_U"] = float(test_res["NDCG_CI_U"])

        fold_rows.append(row)

        print(
            f"Fold {fold_id}: best_alpha={best_alpha:.2f}, "
            f"train_ndcg={best_train_ndcg:.4f}, "
            f"test_ndcg={test_res['NDCG@K']:.4f}"
        )

    summary = {
        "Fold": "AVG",
        "TrainQueries": float(np.mean([r["TrainQueries"] for r in fold_rows])),
        "TestQueries": float(np.mean([r["TestQueries"] for r in fold_rows])),
        "BestAlpha": float(np.mean([r["BestAlpha"] for r in fold_rows])),
        "TrainNDCG@K": float(np.mean([r["TrainNDCG@K"] for r in fold_rows])),
        "TestPrecision@K": float(np.mean([r["TestPrecision@K"] for r in fold_rows])),
        "TestRecall@K": float(np.mean([r["TestRecall@K"] for r in fold_rows])),
        "TestNDCG@K": float(np.mean([r["TestNDCG@K"] for r in fold_rows])),
    }
    fold_rows.append(summary)

    out_path = os.path.join(out_dir, "cross_validation_results.csv")
    _save_csv(out_path, fold_rows)

    print("\nCross-validation summary")
    print("------------------------")
    print(f"Average Test Precision@{k}: {summary['TestPrecision@K']:.4f}")
    print(f"Average Test Recall@{k}   : {summary['TestRecall@K']:.4f}")
    print(f"Average Test NDCG@{k}     : {summary['TestNDCG@K']:.4f}")
    print(f"Average Best Alpha        : {summary['BestAlpha']:.4f}")
    print(f"Saved                     : {out_path}")

    return out_path


# -----------------------------
# 2) Proxy evaluation
# -----------------------------

def evaluate_proxy_relevance(
    k: int = 10,
    alpha: float = 0.6,
    sample_n: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Proxy relevance:
      relevant(q, d) = same industry OR same position

    This is NOT ground truth. Use as secondary analysis only.
    """
    engine = JobRecommenderEngine()
    engine.alpha = float(alpha)

    df = engine.df.copy()
    df = df[df["active"] == 1].reset_index(drop=True)

    if 0 < sample_n < len(df):
        df_q = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    else:
        df_q = df

    industry_counts = df["industry"].fillna("").value_counts().to_dict() if "industry" in df.columns else {}
    position_counts = df["position"].fillna("").value_counts().to_dict() if "position" in df.columns else {}

    p_list, r_list, n_list = [], [], []

    for _, q in df_q.iterrows():
        q_text = safe_str(q.get("combined_text"))
        q_ind = safe_str(q.get("industry"))
        q_pos = safe_str(q.get("position"))

        recs = engine.recommend(query_text=q_text, top_k=k)

        rel_flags: List[int] = []
        for rec in recs:
            r_ind = safe_str(rec.get("industry"))
            r_pos = safe_str(rec.get("position"))

            same_industry = (q_ind != "" and r_ind != "" and q_ind == r_ind)
            same_position = (q_pos != "" and r_pos != "" and q_pos == r_pos)
            rel = 1 if (same_industry or same_position) else 0
            rel_flags.append(rel)

        total_rel = 0
        if q_ind and q_ind in industry_counts:
            total_rel = max(total_rel, int(industry_counts[q_ind]) - 1)
        if q_pos and q_pos in position_counts:
            total_rel = max(total_rel, int(position_counts[q_pos]) - 1)

        p_list.append(sum(rel_flags[:k]) / float(k))
        r_list.append((sum(rel_flags[:k]) / float(total_rel)) if total_rel > 0 else 0.0)

        dcg = 0.0
        for i, rflag in enumerate(rel_flags[:k]):
            if rflag:
                dcg += 1.0 / math.log2(i + 2)
        ideal_hits = min(sum(rel_flags[:k]), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        n_list.append((dcg / idcg) if idcg > 0 else 0.0)

    return {
        "Alpha": float(alpha),
        "K": float(k),
        "Queries": float(len(df_q)),
        "Precision@K": float(np.mean(p_list)) if p_list else 0.0,
        "Recall@K": float(np.mean(r_list)) if r_list else 0.0,
        "NDCG@K": float(np.mean(n_list)) if n_list else 0.0,
    }


def alpha_sweep_proxy(
    k: int = 10,
    alphas: Optional[List[float]] = None,
    sample_n: int = 1000,
    seed: int = 42,
    out_dir: str = "results",
) -> str:
    _ensure_out_dir(out_dir)

    if alphas is None:
        alphas = [round(x, 2) for x in np.linspace(0.0, 1.0, 11).tolist()]

    rows = []
    for a in alphas:
        m = evaluate_proxy_relevance(k=k, alpha=a, sample_n=sample_n, seed=seed)
        rows.append(m)
        print(f"alpha={a:.2f} -> P@{k}={m['Precision@K']:.4f}, R@{k}={m['Recall@K']:.4f}, NDCG@{k}={m['NDCG@K']:.4f}")

    csv_path = os.path.join(out_dir, "alpha_sweep_proxy.csv")
    _save_csv(csv_path, rows)

    xs = [float(r["Alpha"]) for r in rows]
    p = [float(r["Precision@K"]) for r in rows]
    r_ = [float(r["Recall@K"]) for r in rows]
    n = [float(r["NDCG@K"]) for r in rows]

    _save_alpha_plot(xs, n, os.path.join(out_dir, "alpha_vs_ndcg_proxy.png"), f"Alpha vs NDCG@{k} (Proxy)", f"NDCG@{k}")
    _save_alpha_plot(xs, p, os.path.join(out_dir, "alpha_vs_precision_proxy.png"), f"Alpha vs Precision@{k} (Proxy)", f"Precision@{k}")
    _save_alpha_plot(xs, r_, os.path.join(out_dir, "alpha_vs_recall_proxy.png"), f"Alpha vs Recall@{k} (Proxy)", f"Recall@{k}")

    best = max(rows, key=lambda t: float(t["NDCG@K"]))
    print(f"\nBEST (Proxy) by NDCG@{k}: alpha={best['Alpha']:.2f}, NDCG@{k}={best['NDCG@K']:.4f}")
    return csv_path


# -----------------------------
# 3) Self-query sanity check
# -----------------------------

def evaluate_self_query_hit(
    sample_size: int = 300,
    k: int = 10,
    seed: int = 42,
    alpha: float = 0.6,
) -> Dict[str, float]:
    """
    Self-query sanity check:
      - query = job's own combined_text
      - relevant set = {same job id}
    """
    engine = JobRecommenderEngine()
    engine.alpha = float(alpha)

    df = engine.df.copy()
    df = df[df["active"] == 1].reset_index(drop=True)

    if sample_size > 0 and sample_size < len(df):
        df_q = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        df_q = df

    p_list, r_list, n_list = [], [], []

    for _, q in df_q.iterrows():
        q_id = int(q["id"])
        q_text = safe_str(q.get("combined_text"))
        relevant = {q_id}

        recs = engine.recommend(query_text=q_text, top_k=k)
        pred_ids = _extract_ids(recs)

        p_list.append(precision_at_k(pred_ids, relevant, k))
        r_list.append(recall_at_k(pred_ids, relevant, k))
        n_list.append(ndcg_at_k(pred_ids, relevant, k))

    return {
        "Alpha": float(alpha),
        "K": float(k),
        "Queries": float(len(df_q)),
        "Precision@K": float(np.mean(p_list)) if p_list else 0.0,
        "Recall@K": float(np.mean(r_list)) if r_list else 0.0,
        "NDCG@K": float(np.mean(n_list)) if n_list else 0.0,
    }


def alpha_sweep_self_query(
    sample_size: int = 500,
    k: int = 10,
    seed: int = 42,
    out_dir: str = "results",
    alphas: Optional[List[float]] = None,
) -> str:
    _ensure_out_dir(out_dir)

    if alphas is None:
        alphas = [round(x, 2) for x in np.linspace(0.0, 1.0, 11).tolist()]

    rows = []
    for a in alphas:
        m = evaluate_self_query_hit(sample_size=sample_size, k=k, seed=seed, alpha=a)
        rows.append(m)
        print(f"alpha={a:.2f} -> P@{k}={m['Precision@K']:.4f}, R@{k}={m['Recall@K']:.4f}, NDCG@{k}={m['NDCG@K']:.4f}")

    csv_path = os.path.join(out_dir, "alpha_sweep_self_query.csv")
    _save_csv(csv_path, rows)

    xs = [float(r["Alpha"]) for r in rows]
    p = [float(r["Precision@K"]) for r in rows]
    r_ = [float(r["Recall@K"]) for r in rows]
    n = [float(r["NDCG@K"]) for r in rows]

    _save_alpha_plot(xs, n, os.path.join(out_dir, "alpha_vs_ndcg_self_query.png"), f"Alpha vs NDCG@{k} (Self-query)", f"NDCG@{k}")
    _save_alpha_plot(xs, p, os.path.join(out_dir, "alpha_vs_precision_self_query.png"), f"Alpha vs Precision@{k} (Self-query)", f"Precision@{k}")
    _save_alpha_plot(xs, r_, os.path.join(out_dir, "alpha_vs_recall_self_query.png"), f"Alpha vs Recall@{k} (Self-query)", f"Recall@{k}")

    best = max(rows, key=lambda t: float(t["NDCG@K"]))
    print(f"\nBEST (Self-query) by NDCG@{k}: alpha={best['Alpha']:.2f}, NDCG@{k}={best['NDCG@K']:.4f}")
    return csv_path


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation utilities for the job recommender.")
    parser.add_argument(
        "--mode",
        type=str,
        default="manual_sweep",
        choices=[
            "manual",
            "manual_sweep",
            "significance",
            "crossval",
            "proxy",
            "proxy_sweep",
            "self_query",
            "self_query_sweep"
        ]
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="manual_labels.csv",
        help="CSV with columns: query,relevant_ids (required for manual/manual_sweep/significance/crossval)."
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--alpha_a", type=float, default=0.1, help="System A alpha for significance testing")
    parser.add_argument("--alpha_b", type=float, default=0.0, help="System B alpha for significance testing")
    parser.add_argument("--metric", type=str, default="ndcg", choices=["precision", "recall", "ndcg"])
    parser.add_argument("--n_perm", type=int, default=10000, help="Permutations for randomization test")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--sample_n", type=int, default=1000, help="For proxy evaluation")
    parser.add_argument("--sample_size", type=int, default=500, help="For self-query evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--n_boot", type=int, default=0, help="Bootstrap samples for manual evaluation / test fold eval")
    args = parser.parse_args()

    _ensure_out_dir(args.out_dir)

    if args.mode == "manual":
        res = evaluate_manual_labeled_queries(
            labels_csv=args.labels_csv,
            k=args.k,
            alpha=args.alpha,
            n_boot=args.n_boot,
            seed=args.seed
        )
        out_path = os.path.join(args.out_dir, "manual_eval.csv")
        _save_csv(out_path, [res])

        print(
            f"Manual label file rows: total={int(res['TotalRows'])}, "
            f"skipped_empty={int(res['SkippedEmpty'])}, "
            f"skipped_no_relevant={int(res['SkippedNoRelevant'])}, "
            f"evaluated={int(res['EvaluatedRows'])}"
        )
        print(res)
        print(f"\nSaved: {out_path}")

    elif args.mode == "manual_sweep":
        if not os.path.exists(args.labels_csv):
            raise FileNotFoundError(f"labels_csv not found: {args.labels_csv}")
        path = alpha_sweep_manual_labels(
            labels_csv=args.labels_csv,
            k=args.k,
            out_dir=args.out_dir,
            n_boot=args.n_boot,
            seed=args.seed
        )
        print(f"\nSaved: {path}")

    elif args.mode == "significance":
        if not os.path.exists(args.labels_csv):
            raise FileNotFoundError(f"labels_csv not found: {args.labels_csv}")
        path = significance_compare_manual(
            labels_csv=args.labels_csv,
            alpha_a=args.alpha_a,
            alpha_b=args.alpha_b,
            k=args.k,
            metric=args.metric,
            n_perm=args.n_perm,
            seed=args.seed,
            out_dir=args.out_dir
        )
        print(f"\nSaved: {path}")

    elif args.mode == "crossval":
        if not os.path.exists(args.labels_csv):
            raise FileNotFoundError(f"labels_csv not found: {args.labels_csv}")
        path = cross_validate_alpha(
            labels_csv=args.labels_csv,
            k=args.k,
            folds=args.folds,
            seed=args.seed,
            n_boot=args.n_boot,
            out_dir=args.out_dir
        )
        print(f"\nSaved: {path}")

    elif args.mode == "proxy":
        res = evaluate_proxy_relevance(k=args.k, alpha=args.alpha, sample_n=args.sample_n, seed=args.seed)
        out_path = os.path.join(args.out_dir, "proxy_eval.csv")
        _save_csv(out_path, [res])
        print(res)
        print(f"\nSaved: {out_path}")

    elif args.mode == "proxy_sweep":
        path = alpha_sweep_proxy(k=args.k, sample_n=args.sample_n, seed=args.seed, out_dir=args.out_dir)
        print(f"\nSaved: {path}")

    elif args.mode == "self_query":
        res = evaluate_self_query_hit(sample_size=args.sample_size, k=args.k, seed=args.seed, alpha=args.alpha)
        out_path = os.path.join(args.out_dir, "self_query_eval.csv")
        _save_csv(out_path, [res])
        print(res)
        print(f"\nSaved: {out_path}")

    elif args.mode == "self_query_sweep":
        path = alpha_sweep_self_query(sample_size=args.sample_size, k=args.k, seed=args.seed, out_dir=args.out_dir)
        print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()