"""
Build the 3.0 portion of strategy_weights.csv from real weights in
420/420_growth_clients_35_minimal.csv.

Mapping:
  risk_level: 1→C1 .. 5→C5
  lifecycle:  刚毕业→S1, 单身青年→S2, 二人世界→S3, 小孩学前→S4,
              小孩成年前→S5, 子女成年→S6, 退休→S7
  commodity_pct → ALT
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Read real 3.0 weights ---
raw = pd.read_csv(ROOT / "420" / "420_growth_clients_35_minimal.csv", encoding="utf-8-sig")

lifecycle_map = {
    "刚毕业": "S1", "单身青年": "S2", "二人世界": "S3", "小孩学前": "S4",
    "小孩成年前": "S5", "子女成年": "S6", "退休": "S7",
}

raw["profile_id"] = "C" + raw["risk_level"].astype(str) + "_" + raw["lifecycle"].map(lifecycle_map)

# Melt to long format
asset_map = {"cash_pct": "CASH", "bond_pct": "BOND", "equity_pct": "EQUITY", "commodity_pct": "ALT"}
rows = []
for _, r in raw.iterrows():
    for col, asset_class in asset_map.items():
        weight = r[col] / 100.0
        rows.append({
            "portfolio_type": "3.0",
            "profile_id": r["profile_id"],
            "asset_class": asset_class,
            "product_code": "",
            "weight": weight,
        })

new_30 = pd.DataFrame(rows)

# --- Read existing strategy_weights and replace 3.0 portion ---
sw_path = ROOT / "data" / "strategy_weights.csv"
sw = pd.read_csv(sw_path)

sw_other = sw[sw["portfolio_type"] != "3.0"]
result = pd.concat([new_30, sw_other], ignore_index=True)
result.sort_values(["portfolio_type", "profile_id", "asset_class"], inplace=True)

result.to_csv(sw_path, index=False)

# --- Verify ---
check = result[result["portfolio_type"] == "3.0"]
sums = check.groupby("profile_id")["weight"].sum()
print(f"Replaced 3.0 weights: {len(check)} rows, {check['profile_id'].nunique()} profiles")
print(f"Weight sums: min={sums.min()}, max={sums.max()}")
print(f"\nSample (C1_S1):")
print(check[check["profile_id"] == "C1_S1"][["asset_class", "weight"]].to_string(index=False))
print(f"\nSample (C5_S7):")
print(check[check["profile_id"] == "C5_S7"][["asset_class", "weight"]].to_string(index=False))
