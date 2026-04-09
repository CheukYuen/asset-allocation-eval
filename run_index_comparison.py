"""Run index layer comparison only: 3.0 (AI-generated) vs 420_static (original 420)."""

import pandas as pd
from pathlib import Path
from src.load import load_csv, _melt_asset_returns, validate_weights, validate_eligibility
from src.calc import portfolio_monthly_returns, compute_all_metrics
from src.compare import compare_pair, summarize
from src.report import _df_to_md

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "index_3.0_vs_420"


def _save_csv(df, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  -> {path}")


def _save_md(content, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_text(content, encoding="utf-8")
    print(f"  -> {path}")

INDEX_PERIODS = {
    "1y": 12,
    "3y": 36,
    "5y": 60,
    "10y": 120,
    "20y": 240,
}


def run():
    print("Loading data...")
    weights = load_csv("strategy_weights.csv")
    asset_ret = _melt_asset_returns(load_csv("asset_returns.csv"))
    eligibility = load_csv("eligibility_matrix.csv")

    # Only validate index-layer strategies
    idx_weights = weights[weights["portfolio_type"].isin(["3.0", "420_static"])].copy()
    validate_weights(idx_weights)
    validate_eligibility(idx_weights, eligibility)
    print(f"Validated: {idx_weights['portfolio_type'].nunique()} strategies, "
          f"{idx_weights['profile_id'].nunique()} profiles each.\n")

    # Compute
    print("=== Index Layer: 3.0 vs 420_static ===")
    port_30 = portfolio_monthly_returns(idx_weights, asset_ret, "3.0", "asset_class")
    port_420 = portfolio_monthly_returns(idx_weights, asset_ret, "420_static", "asset_class")

    metrics_30 = compute_all_metrics(port_30, INDEX_PERIODS)
    metrics_420 = compute_all_metrics(port_420, INDEX_PERIODS)

    detail = compare_pair(metrics_30, metrics_420, "3.0", "420_static")
    summary = summarize(detail, "3.0", "420_static")

    # Output
    _save_csv(detail, "result_detail_index.csv")
    _save_csv(summary, "result_summary_index.csv")

    # Markdown
    md = "# Index Layer: 3.0 vs 420_static\n\n"
    md += _df_to_md(summary)
    _save_md(md, "summary_index.md")

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nDone. Check {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
