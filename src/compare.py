"""Compare two strategies and produce summary statistics."""

import numpy as np
import pandas as pd


def compare_pair(
    metrics_a: pd.DataFrame,
    metrics_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    risk_anchor: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merge metrics for two strategies on (profile_id, period) and compute deltas.
    If risk_anchor is provided, computes |vol - σ_mid| per strategy for risk-fit comparison.
    """
    merged = metrics_a.merge(
        metrics_b,
        on=["profile_id", "period"],
        suffixes=(f"_{label_a}", f"_{label_b}"),
    )

    ra, rb = f"annualized_return_{label_a}", f"annualized_return_{label_b}"
    va, vb = f"annualized_vol_{label_a}", f"annualized_vol_{label_b}"
    sa, sb = f"sharpe_ratio_{label_a}", f"sharpe_ratio_{label_b}"

    merged["delta_return"] = merged[ra] - merged[rb]
    merged["delta_vol"] = merged[va] - merged[vb]
    merged["delta_sharpe"] = merged[sa] - merged[sb]
    merged["delta_sigma"] = merged[va] - merged[vb]  # same as delta_vol


    # Risk-fit: |vol - σ_mid| per strategy (Δσ from 风险锚体系)
    if risk_anchor is not None:
        merged = merged.merge(
            risk_anchor[["profile_id", "sigma_mid"]],
            on="profile_id",
            how="left",
        )
        merged[f"abs_delta_sigma_{label_a}"] = (merged[va] - merged["sigma_mid"]).abs()
        merged[f"abs_delta_sigma_{label_b}"] = (merged[vb] - merged["sigma_mid"]).abs()

    return merged


def summarize(detail: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    """Produce summary stats grouped by period."""
    ra, rb = f"annualized_return_{label_a}", f"annualized_return_{label_b}"
    va, vb = f"annualized_vol_{label_a}", f"annualized_vol_{label_b}"
    sa, sb = f"sharpe_ratio_{label_a}", f"sharpe_ratio_{label_b}"

    has_anchor = f"abs_delta_sigma_{label_a}" in detail.columns

    rows = []
    for period, grp in detail.groupby("period"):
        n = len(grp)
        row = {
            "period": period,
            "n_clients": n,
            f"mean_return_{label_a}": grp[ra].mean(),
            f"mean_return_{label_b}": grp[rb].mean(),
            f"mean_vol_{label_a}": grp[va].mean(),
            f"mean_vol_{label_b}": grp[vb].mean(),
            f"mean_sharpe_{label_a}": grp[sa].mean(),
            f"mean_sharpe_{label_b}": grp[sb].mean(),
            "win_rate_return": (grp["delta_return"] > 0).mean(),
            "win_rate_sharpe": (grp["delta_sharpe"] > 0).mean(),
        }
        if has_anchor:
            ads_a = f"abs_delta_sigma_{label_a}"
            ads_b = f"abs_delta_sigma_{label_b}"
            # A wins when |vol_A - σ_mid| < |vol_B - σ_mid| (closer to risk target)
            row["win_rate_abs_delta_sigma"] = (grp[ads_a] < grp[ads_b]).mean()
            row[f"mean_abs_delta_sigma_{label_a}"] = grp[ads_a].mean()
            row[f"mean_abs_delta_sigma_{label_b}"] = grp[ads_b].mean()
        else:
            row["win_rate_abs_delta_sigma"] = (grp[va] < grp[vb]).mean()
        rows.append(row)

    return pd.DataFrame(rows)
