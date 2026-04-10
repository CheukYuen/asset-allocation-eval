"""
Asset Allocation Eval — Main entry point.

Usage:
    python generate_mock.py   # first time only
    python main.py
"""

import pandas as pd
from src.load import load_all, validate_weights, validate_eligibility
from src.calc import portfolio_monthly_returns, compute_all_metrics
from src.compare import compare_pair, summarize
from src.report import save_csv, generate_markdown, save_markdown


# ── Config ─────────────────────────────────────────────────────────────────

INDEX_PERIODS = {
    "1y": 12,
    "3y": 36,
    "5y": 60,
    "10y": 120,
    "20y": 240,
}

PRODUCT_PERIODS = {
    "5y": 60,
}


def run():
    print("Loading data...")
    data = load_all()
    weights = data["strategy_weights"]
    asset_ret = data["asset_returns"]
    product_ret = data["product_returns"]
    eligibility = data["eligibility_matrix"]

    validate_weights(weights)
    validate_eligibility(weights, eligibility)
    print("Weights validated (sum & eligibility).\n")

    # ── Index Layer: 3.0 vs 420_static ─────────────────────────────────────
    print("=== Index Layer ===")
    port_30_idx = portfolio_monthly_returns(weights, asset_ret, "3.0", "asset_class")
    port_420s = portfolio_monthly_returns(weights, asset_ret, "420_static", "asset_class")

    metrics_30_idx = compute_all_metrics(port_30_idx, INDEX_PERIODS)
    metrics_420s = compute_all_metrics(port_420s, INDEX_PERIODS)

    detail_idx = compare_pair(metrics_30_idx, metrics_420s, "3.0", "420_static")
    main_idx, winrate_idx = summarize(detail_idx, "3.0", "420_static")

    print("Index layer metrics computed.")

    # ── Product Layer: 3.0_mapped_product vs 420_online ────────────────────
    print("\n=== Product Layer ===")
    port_30_prd = portfolio_monthly_returns(weights, product_ret, "3.0_mapped_product", "product_code")
    port_420o = portfolio_monthly_returns(weights, product_ret, "420_online", "product_code")

    metrics_30_prd = compute_all_metrics(port_30_prd, PRODUCT_PERIODS)
    metrics_420o = compute_all_metrics(port_420o, PRODUCT_PERIODS)

    detail_prd = compare_pair(metrics_30_prd, metrics_420o, "3.0_mapped", "420_online")
    main_prd, winrate_prd = summarize(detail_prd, "3.0_mapped", "420_online")

    print("Product layer metrics computed.\n")

    # ── Output ─────────────────────────────────────────────────────────────
    print("Saving results...")
    save_csv(detail_idx, "result_detail_index.csv")
    save_csv(detail_prd, "result_detail_product.csv")
    save_csv(main_idx, "result_main_index.csv")
    save_csv(winrate_idx, "result_winrate_index.csv")
    save_csv(main_prd, "result_main_product.csv")
    save_csv(winrate_prd, "result_winrate_product.csv")

    md = generate_markdown(
        main_idx, winrate_idx,
        main_prd, winrate_prd,
        "3.0", "420_static",
        "3.0_mapped", "420_online",
    )
    save_markdown(md)

    print("\nDone. Check output/ directory.")


if __name__ == "__main__":
    run()
