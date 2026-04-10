"""Portfolio return calculation and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Portfolio monthly returns
# ---------------------------------------------------------------------------

def portfolio_monthly_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    portfolio_type: str,
    join_col: str,  # "asset_class" for index layer, "product_code" for product layer
) -> pd.DataFrame:
    """
    Compute monthly portfolio returns for all clients under a given portfolio_type.

    Returns DataFrame with columns: [profile_id, date, port_return]
    """
    w = weights[weights["portfolio_type"] == portfolio_type][["profile_id", join_col, "weight"]]

    merged = w.merge(returns, on=join_col, how="inner")
    merged["weighted_return"] = merged["weight"] * merged["return"]

    port = merged.groupby(["profile_id", "date"])["weighted_return"].sum().reset_index()
    port.rename(columns={"weighted_return": "port_return"}, inplace=True)
    return port.sort_values(["profile_id", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _annualized_return(monthly: np.ndarray) -> float:
    cumulative = np.prod(1 + monthly) ** (12 / len(monthly)) - 1
    return cumulative


def _annualized_vol(monthly: np.ndarray) -> float:
    return np.std(monthly, ddof=1) * np.sqrt(12)


def _sharpe_ratio(monthly: np.ndarray, rf_annual: float = 0.02) -> float:
    ann_ret = _annualized_return(monthly)
    ann_vol = _annualized_vol(monthly)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - rf_annual) / ann_vol


def _align_rf(dates: np.ndarray, rf_series: "pd.Series") -> np.ndarray:
    """Align annual CGB_1Y (%) to portfolio dates; return monthly rf rates (decimal).

    Uses forward-fill to handle sparse rf data. Raises if rf is unavailable at
    the start of the window (no prior value to fill from).
    """
    dates_idx = pd.DatetimeIndex(dates)
    # Union with rf index so ffill can propagate across any date gaps
    combined_idx = rf_series.index.union(dates_idx)
    rf_filled = rf_series.reindex(combined_idx).ffill()
    rf_aligned = rf_filled.reindex(dates_idx)

    if rf_aligned.isna().any():
        first_missing = dates_idx[rf_aligned.isna()][0].date()
        raise ValueError(
            f"CGB_1Y unavailable at or before {first_missing}; "
            "no prior value to forward-fill. Cannot compute dynamic Sharpe."
        )

    # Convert annual % → monthly decimal: (1 + r/100)^(1/12) - 1
    return ((1 + rf_aligned.values / 100) ** (1 / 12) - 1).astype(float)


def _sharpe_ratio_dynamic(monthly: np.ndarray, rf_monthly: np.ndarray) -> float:
    """Sharpe ratio from per-month excess returns (no look-ahead bias).

    monthly:   array of monthly portfolio returns (decimal)
    rf_monthly: array of monthly risk-free rates (decimal), same length
    Returns annualized Sharpe scalar.
    """
    if len(monthly) != len(rf_monthly):
        raise ValueError(
            f"Length mismatch: monthly={len(monthly)}, rf_monthly={len(rf_monthly)}"
        )
    excess = monthly - rf_monthly
    std_excess = np.std(excess, ddof=1)
    if std_excess == 0:
        return 0.0
    return float(np.mean(excess) / std_excess * np.sqrt(12))


def _max_drawdown(monthly: np.ndarray) -> float:
    cumulative = np.cumprod(1 + monthly)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    return float(np.min(drawdowns))


def compute_metrics(monthly_returns: np.ndarray) -> dict:
    return {
        "annualized_return": _annualized_return(monthly_returns),
        "annualized_vol": _annualized_vol(monthly_returns),
        "sharpe_ratio": _sharpe_ratio(monthly_returns),
        "max_drawdown": _max_drawdown(monthly_returns),
    }


def compute_all_metrics(
    port_returns: pd.DataFrame,
    periods_months: dict[str, int] | None = None,
    rf_series: "pd.Series | None" = None,
) -> pd.DataFrame:
    """
    Compute metrics for each client profile, optionally across multiple lookback periods.

    port_returns: [profile_id, date, port_return]
    periods_months: e.g. {"1y": 12, "3y": 36, ...}. If None, use full history.
    rf_series: date-indexed pd.Series of annual CGB_1Y rates (%). If provided,
               Sharpe is computed from per-month excess returns (dynamic rf).
               If None, falls back to fixed 2% annual rf.
    """
    if periods_months is None:
        periods_months = {"full": 0}

    rows = []
    for profile_id, grp in port_returns.groupby("profile_id"):
        grp = grp.sort_values("date")
        monthly = grp["port_return"].values
        dates = grp["date"].values
        total_months = len(monthly)

        for period_name, n_months in periods_months.items():
            if n_months == 0:
                subset = monthly
                subset_dates = dates
            else:
                if total_months < n_months:
                    continue
                subset = monthly[-n_months:]
                subset_dates = dates[-n_months:]

            m = compute_metrics(subset)
            if rf_series is not None:
                rf_monthly = _align_rf(subset_dates, rf_series)
                m["sharpe_ratio"] = _sharpe_ratio_dynamic(subset, rf_monthly)
            m["profile_id"] = profile_id
            m["period"] = period_name
            rows.append(m)

    return pd.DataFrame(rows)
