"""
Build asset_returns.csv from 建模月频序列.csv using Total Return口径.

Asset legs:
  CASH:   CBA02201
  BOND:   CBOND_NEW_COMPOSITE_WEALTH
  EQUITY: CSI300_TR
  ALT:    70% × AU9999 + 30% × NHCI

All converted to monthly returns: P_t / P_{t-1} - 1

Output format: wide table (month, CASH, BOND, EQUITY, ALT) — same layout as 建模月频序列.csv.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

raw = pd.read_csv(DATA_DIR / "建模月频序列.csv")

# Rename for convenience (column has .CS suffix)
raw.rename(columns={"CBA02201.CS": "CBA02201"}, inplace=True)

# Sort by month
raw.sort_values("month", inplace=True)
raw.reset_index(drop=True, inplace=True)

# Compute monthly returns from price/index levels
result = pd.DataFrame({"month": raw["month"]})

for asset_class, col in [("CASH", "CBA02201"), ("BOND", "CBOND_NEW_COMPOSITE_WEALTH"),
                          ("EQUITY", "CSI300_TR")]:
    result[asset_class] = raw[col].pct_change()

# ALT = 70% gold + 30% commodity
result["ALT"] = 0.7 * raw["AU9999"].pct_change() + 0.3 * raw["NHCI"].pct_change()

# Drop the first row (no prior month to compute return from) and rows where all assets are NaN
result = result.dropna(subset=["CASH", "BOND", "EQUITY", "ALT"], how="all")

# Round to 8 decimal places
for col in ["CASH", "BOND", "EQUITY", "ALT"]:
    result[col] = result[col].round(8)

# Also extract latest risk-free rate (informational)
rf_row = raw[["month", "CGB_1Y"]].dropna(subset=["CGB_1Y"]).iloc[-1]
print(f"Latest rf (CGB_1Y): {rf_row['CGB_1Y']}% as of {rf_row['month']}")

# Save CGB_1Y series for dynamic Sharpe rf
rf_series = raw[["month", "CGB_1Y"]].copy()
rf_series["CGB_1Y"] = rf_series["CGB_1Y"].ffill()
rf_out = DATA_DIR / "rf_series.csv"
rf_series.to_csv(rf_out, index=False)
print(f"Saved rf_series.csv ({rf_series['CGB_1Y'].notna().sum()} rows) to {rf_out}")

# Save
out_path = DATA_DIR / "asset_returns.csv"
result.to_csv(out_path, index=False)

# Summary
print(f"\nSaved {len(result)} rows to {out_path}")
print(f"\nCoverage per asset class:")
for col in ["CASH", "BOND", "EQUITY", "ALT"]:
    valid = result[col].dropna()
    print(f"  {col}: {result.loc[valid.index[0], 'month']} ~ {result.loc[valid.index[-1], 'month']} ({len(valid)} months)")
