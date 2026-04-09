"""
Build strategy_weights.csv with:
  - 3.0       = AI-generated weights from AI-invest/outputs/extracted_weights_v3.csv
  - 420_static = original weights from 420/420_growth_clients_35_minimal.csv

Usage:
    python3 build_ai_strategy_weights.py [--ai-weights AI_CSV]
"""

import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Paths ---
AI_WEIGHTS_PATH = ROOT / "AI-invest" / "outputs" / "extracted_weights_v3.csv"
CSV_420_PATH = ROOT / "420" / "420_growth_clients_35_minimal.csv"
SW_PATH = ROOT / "data" / "strategy_weights.csv"

# Allow overriding AI weights path via CLI
if len(sys.argv) > 2 and sys.argv[1] == "--ai-weights":
    AI_WEIGHTS_PATH = Path(sys.argv[2])

LIFECYCLE_MAP = {
    "刚毕业": "S1", "单身青年": "S2", "二人世界": "S3", "小孩学前": "S4",
    "小孩成年前": "S5", "子女成年": "S6", "退休": "S7",
}

ELIGIBILITY = {
    "C1": {"CASH", "BOND"},
    "C2": {"CASH", "BOND", "ALT"},
    "C3": {"CASH", "BOND", "EQUITY", "ALT"},
    "C4": {"CASH", "BOND", "EQUITY", "ALT"},
    "C5": {"CASH", "BOND", "EQUITY", "ALT"},
}


def build_rows(df, portfolio_type, weight_cols):
    """Build long-format weight rows from a wide dataframe."""
    rows = []
    for _, r in df.iterrows():
        profile_id = "C" + str(int(r["risk_level"])) + "_" + LIFECYCLE_MAP[r["lifecycle"]]
        risk = "C" + str(int(r["risk_level"]))
        eligible = ELIGIBILITY[risk]
        for col, asset_class in weight_cols.items():
            w = r[col]
            if pd.isna(w) or asset_class not in eligible:
                w = 0.0
            rows.append({
                "portfolio_type": portfolio_type,
                "profile_id": profile_id,
                "asset_class": asset_class,
                "product_code": "",
                "weight": round(float(w), 6),
            })
    return pd.DataFrame(rows)


# --- Build 420_static from original CSV ---
print("Building 420_static from", CSV_420_PATH)
raw_420 = pd.read_csv(CSV_420_PATH, encoding="utf-8-sig")
cols_420 = {"cash_pct": "CASH", "bond_pct": "BOND", "equity_pct": "EQUITY", "commodity_pct": "ALT"}
rows_420 = build_rows(raw_420, "420_static", cols_420)
# Convert from percentage (0-100) to proportion (0-1)
rows_420["weight"] = (rows_420["weight"] / 100.0).round(6)

# --- Build 3.0 from AI-extracted weights ---
print("Building 3.0 from", AI_WEIGHTS_PATH)
ai = pd.read_csv(AI_WEIGHTS_PATH, encoding="utf-8-sig")

# Check parse status
total = len(ai)
success = (ai["parse_status"] == "success").sum()
failed = (ai["parse_status"] == "failed").sum()
print(f"  AI weights: {success}/{total} success, {failed} failed")

if failed > 0:
    print(f"  WARNING: {failed} clients have failed extraction, will use 0 weights")
    print(f"  Failed IDs: {ai[ai['parse_status'] == 'failed']['id'].tolist()}")

cols_ai = {"CASH": "CASH", "BOND": "BOND", "EQUITY": "EQUITY", "ALT": "ALT"}
rows_ai = build_rows(ai, "3.0", cols_ai)
# AI weights are already in percentage (0-100), convert to proportion (0-1)
rows_ai["weight"] = (rows_ai["weight"] / 100.0).round(6)

# Handle NaN from failed extractions: replace with 0
rows_ai["weight"] = rows_ai["weight"].fillna(0.0)

# Normalize: ensure weights sum to 1 per profile
for pid in rows_ai["profile_id"].unique():
    mask = rows_ai["profile_id"] == pid
    s = rows_ai.loc[mask, "weight"].sum()
    if s > 0 and abs(s - 1.0) > 0.001:
        rows_ai.loc[mask, "weight"] = (rows_ai.loc[mask, "weight"] / s).round(6)
    elif s == 0:
        # Failed extraction: distribute equally among eligible assets
        risk = pid.split("_")[0]
        eligible = ELIGIBILITY[risk]
        n = sum(1 for _, r in rows_ai[mask].iterrows() if r["asset_class"] in eligible)
        if n > 0:
            for idx in rows_ai[mask].index:
                if rows_ai.loc[idx, "asset_class"] in eligible:
                    rows_ai.loc[idx, "weight"] = round(1.0 / n, 6)

# --- Merge with existing strategy_weights (keep product-layer entries) ---
sw = pd.read_csv(SW_PATH)
sw_keep = sw[~sw["portfolio_type"].isin(["3.0", "420_static"])]
result = pd.concat([rows_ai, rows_420, sw_keep], ignore_index=True)
result.sort_values(["portfolio_type", "profile_id", "asset_class"], inplace=True)

result.to_csv(SW_PATH, index=False)

# --- Verify ---
for pt in ["3.0", "420_static"]:
    sub = result[result["portfolio_type"] == pt]
    sums = sub.groupby("profile_id")["weight"].sum()
    print(f"\n{pt}: {sub['profile_id'].nunique()} profiles, "
          f"weight sums: min={sums.min():.4f}, max={sums.max():.4f}")
    sample = sub[sub["profile_id"] == "C3_S2"][["asset_class", "weight"]]
    if len(sample) > 0:
        print(f"  Sample C3_S2:")
        print(sample.to_string(index=False))

print("\nDone. strategy_weights.csv updated.")
