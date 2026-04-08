"""Load and validate input CSVs."""

import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run generate_mock.py first.")
    return pd.read_csv(path)


def load_all() -> dict[str, pd.DataFrame]:
    names = [
        "client_profiles.csv",
        "strategy_weights.csv",
        "asset_returns.csv",
        "product_returns.csv",
        "risk_anchor.csv",
        "eligibility_matrix.csv",
    ]
    return {n.replace(".csv", ""): load_csv(n) for n in names}


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
