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
    # Also carry max_drawdown_tolerance for exceed_rate computation in summarize()
    if risk_anchor is not None:
        anchor_cols = ["profile_id", "sigma_mid"]
        if "max_drawdown_tolerance" in risk_anchor.columns:
            anchor_cols.append("max_drawdown_tolerance")
        merged = merged.merge(risk_anchor[anchor_cols], on="profile_id", how="left")
        merged[f"abs_delta_sigma_{label_a}"] = (merged[va] - merged["sigma_mid"]).abs()
        merged[f"abs_delta_sigma_{label_b}"] = (merged[vb] - merged["sigma_mid"]).abs()

    return merged


def summarize(
    detail: pd.DataFrame, label_a: str, label_b: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce two summary tables grouped by period:
      - main_table: absolute levels (return, sharpe, |Δσ|, MaxDD exceed rate)
      - win_rate_table: win rates (return, sharpe, risk-match)
    Returns (main_table, win_rate_table).
    """
    ra, rb = f"annualized_return_{label_a}", f"annualized_return_{label_b}"
    sa, sb = f"sharpe_ratio_{label_a}", f"sharpe_ratio_{label_b}"
    mdd_a = f"max_drawdown_{label_a}"
    mdd_b = f"max_drawdown_{label_b}"

    has_anchor = f"abs_delta_sigma_{label_a}" in detail.columns
    has_mdd_tol = (
        "max_drawdown_tolerance" in detail.columns
        and mdd_a in detail.columns
        and mdd_b in detail.columns
    )

    main_rows, win_rows = [], []
    for period, grp in detail.groupby("period"):
        # ── 正文主表 ──────────────────────────────────────────────────────
        main_row = {
            "window": period,
            f"mean_return_{label_a}": grp[ra].mean(),
            f"mean_return_{label_b}": grp[rb].mean(),
            f"mean_sharpe_{label_a}": grp[sa].mean(),
            f"mean_sharpe_{label_b}": grp[sb].mean(),
        }
        if has_anchor:
            ads_a, ads_b = f"abs_delta_sigma_{label_a}", f"abs_delta_sigma_{label_b}"
            main_row[f"mean_abs_delta_sigma_{label_a}"] = grp[ads_a].mean()
            main_row[f"mean_abs_delta_sigma_{label_b}"] = grp[ads_b].mean()
        if has_mdd_tol:
            # exceed = actual drawdown is worse (more negative) than tolerance
            tol = grp["max_drawdown_tolerance"]
            main_row[f"exceed_rate_maxdd_{label_a}"] = (grp[mdd_a] < tol).mean()
            main_row[f"exceed_rate_maxdd_{label_b}"] = (grp[mdd_b] < tol).mean()
        main_rows.append(main_row)

        # ── 胜率表 ────────────────────────────────────────────────────────
        win_row = {
            "window": period,
            "win_rate_return": (grp["delta_return"] > 0).mean(),
            "win_rate_sharpe": (grp["delta_sharpe"] > 0).mean(),
        }
        if has_anchor:
            ads_a, ads_b = f"abs_delta_sigma_{label_a}", f"abs_delta_sigma_{label_b}"
            win_row["win_rate_risk_match"] = (grp[ads_a] < grp[ads_b]).mean()
        win_rows.append(win_row)

    return pd.DataFrame(main_rows), pd.DataFrame(win_rows)
