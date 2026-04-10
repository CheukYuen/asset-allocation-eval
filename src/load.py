"""Load and validate input CSVs."""

import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run generate_mock.py first.")
    return pd.read_csv(path)


def _melt_asset_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide-format asset_returns (month, CASH, BOND, EQUITY, ALT)
    to long-format (date, asset_class, return) expected by calc.py."""
    long = df.melt(id_vars="month", var_name="asset_class", value_name="return")
    long = long.dropna(subset=["return"])
    long["date"] = pd.to_datetime(long["month"]) + pd.offsets.MonthEnd(0)
    long = long[["date", "asset_class", "return"]].sort_values(["date", "asset_class"])
    return long.reset_index(drop=True)


def _melt_product_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide-format product_returns (date, CASH, BOND, EQUITY, ALT)
    to long-format (date, asset_class, return) expected by calc.py."""
    long = df.melt(id_vars="date", var_name="asset_class", value_name="return")
    long = long.dropna(subset=["return"])
    long["date"] = pd.to_datetime(long["date"]) + pd.offsets.MonthEnd(0)
    long = long[["date", "asset_class", "return"]].sort_values(["date", "asset_class"])
    return long.reset_index(drop=True)


def load_rf_series() -> pd.Series:
    """Load CGB_1Y from data/rf_series.csv as a month-end-date-indexed Series.

    Values are annual rates in % (e.g. 2.32 means 2.32%).
    Built by build_asset_returns.py. Forward-fills any within-series gaps.
    Caller must handle leading NaNs (raises via _align_rf).
    """
    df = load_csv("rf_series.csv")
    df["date"] = pd.to_datetime(df["month"]) + pd.offsets.MonthEnd(0)
    df = df.sort_values("date").drop(columns="month")
    df["CGB_1Y"] = df["CGB_1Y"].ffill()
    return df.set_index("date")["CGB_1Y"]


def load_all() -> dict[str, pd.DataFrame]:
    names = [
        "client_profiles.csv",
        "strategy_weights.csv",
        "asset_returns.csv",
        "product_returns.csv",
        "risk_anchor.csv",
        "eligibility_matrix.csv",
    ]
    data = {n.replace(".csv", ""): load_csv(n) for n in names}
    data["asset_returns"] = _melt_asset_returns(data["asset_returns"])
    data["product_returns"] = _melt_product_returns(data["product_returns"])
    return data


def validate_weights(weights: pd.DataFrame) -> None:
    """Check that weights sum to 1 per (portfolio_type, profile_id)."""
    sums = weights.groupby(["portfolio_type", "profile_id"])["weight"].sum()
    bad = sums[~sums.between(0.999, 1.001)]
    if len(bad) > 0:
        raise ValueError(f"Weight sums != 1:\n{bad}")


def validate_eligibility(
    weights: pd.DataFrame, eligibility: pd.DataFrame,
) -> None:
    """Check that no weight is assigned to ineligible asset classes."""
    # Build set of (risk_level, asset_class) where eligible=0
    ineligible = set(
        eligibility[eligibility["eligible"] == 0]
        .apply(lambda r: (r["risk_level"], r["asset_class"]), axis=1)
    )
    if not ineligible:
        return

    # Extract risk_level from profile_id (e.g. "C3_S2" -> "C3")
    w = weights[weights["asset_class"] != ""].copy()
    w["risk_level"] = w["profile_id"].str.split("_").str[0]
    w["pair"] = list(zip(w["risk_level"], w["asset_class"]))

    violations = w[w["pair"].isin(ineligible) & (w["weight"] > 0)]
    if len(violations) > 0:
        raise ValueError(
            f"Eligibility violations ({len(violations)} rows):\n"
            f"{violations[['portfolio_type', 'profile_id', 'asset_class', 'weight']].head(10)}"
        )
