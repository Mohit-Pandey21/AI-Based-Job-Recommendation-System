import os
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from job_recommendation import JobRecommenderEngine


# ----------------------------
# Config
# ----------------------------
OUT_ROOT = Path("bias_compare_outputs")
TOP_K = 10
QUERY_SAMPLE_N = 300     # keep same as your earlier bias notebook (or set 500)
SEED = 42
ALPHAS = [0.0, 0.1, 1.0]


# ----------------------------
# Utilities
# ----------------------------
def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def normalize_dist(s: pd.Series) -> pd.Series:
    """Normalized value counts (drop empty)."""
    s = s.fillna("").astype(str).map(lambda x: x.strip())
    s = s[s != ""]
    if len(s) == 0:
        return pd.Series(dtype=float)
    return s.value_counts(normalize=True)


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def hhi(p: np.ndarray) -> float:
    return float((p ** 2).sum()) if len(p) else 0.0


def gini_from_probs(p: np.ndarray) -> float:
    """
    Gini on probability distribution (0=perfectly even, higher=more concentrated).
    """
    if len(p) == 0:
        return 0.0
    p = np.sort(p)
    n = len(p)
    cum = np.cumsum(p)
    # Gini for discrete distribution
    g = (n + 1 - 2 * (cum.sum() / cum[-1])) / n
    return float(g)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen–Shannon divergence with log2.
    """
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def align_dists(dataset_dist: pd.Series, rec_dist: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Align dataset and recommendation distributions over the union of categories.
    Returns p, q arrays and aligned dataframe.
    """
    all_idx = dataset_dist.index.union(rec_dist.index)
    aligned = pd.DataFrame({
        "dataset_share": dataset_dist.reindex(all_idx).fillna(0.0),
        "recommend_share": rec_dist.reindex(all_idx).fillna(0.0)
    })
    p = aligned["dataset_share"].to_numpy(dtype=float)
    q = aligned["recommend_share"].to_numpy(dtype=float)
    return p, q, aligned


def exposure_table(dataset_dist: pd.Series, rec_dist: pd.Series, top_n: int = 15) -> pd.DataFrame:
    """
    Exposure ratio table for top categories by dataset share.
    """
    top_idx = dataset_dist.head(top_n).index
    tbl = pd.DataFrame({
        "dataset_share": dataset_dist.reindex(top_idx).fillna(0.0),
        "recommend_share": rec_dist.reindex(top_idx).fillna(0.0)
    })
    tbl["exposure_ratio"] = np.where(tbl["dataset_share"] > 0, tbl["recommend_share"] / tbl["dataset_share"], np.nan)
    tbl = tbl.sort_values("exposure_ratio", ascending=False)
    return tbl


def compare_plot(dataset_dist: pd.Series, rec_dist: pd.Series, title: str, out_path: Path, top_n: int = 10) -> None:
    """
    Bar plot comparing top_n dataset categories vs their exposure in recommendations.
    """
    top_idx = dataset_dist.head(top_n).index
    comp = pd.DataFrame({
        "dataset": dataset_dist.reindex(top_idx).fillna(0.0),
        "recommend": rec_dist.reindex(top_idx).fillna(0.0)
    })

    plt.figure()
    x = np.arange(len(comp))
    plt.bar(x - 0.2, comp["dataset"].values, width=0.4, label="dataset")
    plt.bar(x + 0.2, comp["recommend"].values, width=0.4, label="recommendations")
    plt.xticks(x, comp.index.tolist(), rotation=45, ha="right")
    plt.ylabel("Share")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------------
# Core bias run
# ----------------------------
def run_bias_for_alpha(engine: JobRecommenderEngine, alpha: float) -> Dict[str, pd.DataFrame]:
    engine.alpha = float(alpha)

    df = engine.df.copy()
    df = df[df["active"].astype(int) == 1].reset_index(drop=True)

    # fixed query sample so all alphas are comparable
    query_sample = df.sample(n=min(QUERY_SAMPLE_N, len(df)), random_state=SEED).reset_index(drop=True)

    rec_industry: List[str] = []
    rec_city: List[str] = []
    rec_position: List[str] = []

    for _, row in query_sample.iterrows():
        q = safe_str(row.get("combined_text")) or safe_str(row.get("title"))
        recs = engine.recommend(query_text=q, top_k=TOP_K)

        for r in recs:
            rec_industry.append(safe_str(r.get("industry")))
            rec_city.append(safe_str(r.get("city")))
            rec_position.append(safe_str(r.get("position")))

    # Dataset series
    ds_industry = df["industry"].map(safe_str)
    ds_city = df["city"].map(safe_str)
    ds_position = df["position"].map(safe_str)

    # Recommendation series
    rs_industry = pd.Series(rec_industry)
    rs_city = pd.Series(rec_city)
    rs_position = pd.Series(rec_position)

    # Distributions
    d_ind = normalize_dist(ds_industry)
    d_city = normalize_dist(ds_city)
    d_pos = normalize_dist(ds_position)

    r_ind = normalize_dist(rs_industry)
    r_city = normalize_dist(rs_city)
    r_pos = normalize_dist(rs_position)

    outputs: Dict[str, pd.DataFrame] = {}

    # Metrics table
    rows = []
    for name, dd, rr in [
        ("industry", d_ind, r_ind),
        ("city", d_city, r_city),
        ("position", d_pos, r_pos),
    ]:
        p, q, aligned = align_dists(dd, rr)
        rows.append({
            "alpha": alpha,
            "attribute": name,
            "dataset_entropy": entropy(p),
            "rec_entropy": entropy(q),
            "dataset_hhi": hhi(p),
            "rec_hhi": hhi(q),
            "dataset_gini": gini_from_probs(p),
            "rec_gini": gini_from_probs(q),
            "js_divergence": js_divergence(p, q),
            "unique_dataset": int((p > 0).sum()),
            "unique_recommend": int((q > 0).sum()),
        })
        outputs[f"{name}_aligned_dist"] = aligned.sort_values("dataset_share", ascending=False)

        # exposure table (top categories)
        outputs[f"{name}_exposure_table"] = exposure_table(dd, rr, top_n=15)

    outputs["bias_concentration_metrics"] = pd.DataFrame(rows)

    return outputs, (d_ind, r_ind, d_city, r_city, d_pos, r_pos)


def main():
    OUT_ROOT.mkdir(exist_ok=True)
    engine = JobRecommenderEngine()

    summary_rows = []

    for a in ALPHAS:
        out_dir = OUT_ROOT / f"alpha_{a:.1f}"
        fig_dir = out_dir / "figures"
        tab_dir = out_dir / "tables"
        fig_dir.mkdir(parents=True, exist_ok=True)
        tab_dir.mkdir(parents=True, exist_ok=True)

        outputs, dists = run_bias_for_alpha(engine, a)
        d_ind, r_ind, d_city, r_city, d_pos, r_pos = dists

        # Save tables
        outputs["bias_concentration_metrics"].to_csv(tab_dir / "bias_concentration_metrics.csv", index=False)
        outputs["industry_exposure_table"].to_csv(tab_dir / "industry_exposure_table.csv")
        outputs["city_exposure_table"].to_csv(tab_dir / "city_exposure_table.csv")
        outputs["position_exposure_table"].to_csv(tab_dir / "position_exposure_table.csv")

        # Save compare plots
        compare_plot(d_ind, r_ind, f"Industry: Dataset vs Recs (alpha={a:.1f})", fig_dir / "compare_industry.png")
        compare_plot(d_city, r_city, f"City: Dataset vs Recs (alpha={a:.1f})", fig_dir / "compare_city.png")
        compare_plot(d_pos, r_pos, f"Position: Dataset vs Recs (alpha={a:.1f})", fig_dir / "compare_position.png")

        # Add to overall summary
        m = outputs["bias_concentration_metrics"].copy()
        summary_rows.append(m)

        print(f"[DONE] Bias outputs for alpha={a:.1f} saved to: {out_dir}")

    # Save cross-alpha summary
    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(OUT_ROOT / "bias_metrics_across_alphas.csv", index=False)

    print("\nSaved cross-alpha summary:", (OUT_ROOT / "bias_metrics_across_alphas.csv").resolve())


if __name__ == "__main__":
    main()