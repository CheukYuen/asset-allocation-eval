# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Offline evaluation framework comparing smart portfolio strategies (智能投顾 3.0 vs 420). Two comparison layers:
- **Index layer**: 3.0 vs 420_static — asset class weights × index returns
- **Product layer**: 3.0_mapped_product vs 420_online — product weights × product returns

Both layers use buy-and-hold (static weights, no rebalancing) across 35 client segments (C1–C5 × S1–S7).

## Commands

```bash
python3 generate_mock.py   # generate mock data into data/ (idempotent, seed 42)
python3 main.py            # run full evaluation, outputs to output/
pip3 install -r requirements.txt  # pandas, numpy only
```

## Architecture

Data flows linearly: `data/*.csv → load → calc → compare → report → output/`

- **main.py** — orchestrates the pipeline: load data, compute portfolio returns for each layer, calculate metrics, compare strategy pairs, save results
- **src/load.py** — reads 6 input CSVs, validates weight sums to 1 and eligibility constraints
- **src/calc.py** — `portfolio_monthly_returns()` joins weights with returns via `asset_class` (index) or `product_code` (product); `compute_all_metrics()` computes annualized return/vol/sharpe/max drawdown per profile per lookback period
- **src/compare.py** — `compare_pair()` merges two strategies on (profile_id, period), computes deltas; `summarize()` aggregates means and win rates
- **src/report.py** — saves CSVs and markdown summary to `output/`
- **generate_mock.py** — standalone script producing all 6 input CSVs with realistic distributions

Key join: index layer joins on `asset_class`, product layer joins on `product_code`. The `join_col` parameter in `portfolio_monthly_returns()` controls this.

## Key Domain Concepts

- **profile_id**: client persona key = `{risk_level}_{life_stage}` (e.g. `C3_S2`), not a real customer ID
- **Risk anchor (风险锚)**: per-profile risk container — `sigma_min / sigma_mid / sigma_max` computed from `σ_stage_max × m_risk` formulas; strategy-agnostic referee layer
- **Eligibility (适当性约束)**: `eligibility_matrix.csv` defines which asset classes each risk level may invest in (C1: CASH/BOND only; C2: no EQUITY; C3–C5: all four)
- **Δσ = annualized_vol − sigma_mid**: core risk-fit metric; smaller |Δσ| = better client fit

## Output Files Explained

### result_detail.csv / result_detail_index.csv / result_detail_product.csv

Per-profile, per-period comparison. Each row = one profile × one lookback period. Columns include metrics for both strategies (suffixed `_3.0` / `_420_static` etc.) plus deltas:

| Metric | Definition |
|--------|-----------|
| `annualized_return` | `prod(1+r)^(12/n) - 1` — compounded monthly returns annualized |
| `annualized_vol` | `std(monthly, ddof=1) × √12` — annualized standard deviation |
| `sharpe_ratio` | `(ann_return - 0.02) / ann_vol` — risk-free rate hardcoded at 2% |
| `max_drawdown` | largest peak-to-trough decline in cumulative return series |
| `delta_return` | strategy A return minus strategy B return |
| `delta_sigma` | strategy A vol minus strategy B vol (positive = A is riskier) |
| `abs_delta_sigma` | absolute value of delta_sigma |

### result_summary.csv

One row per (period, layer). Aggregates across all 35 profiles:

| Metric | Definition |
|--------|-----------|
| `mean_return_*` | average annualized return across profiles |
| `mean_vol_*` | average annualized volatility across profiles |
| `mean_sharpe_*` | average Sharpe ratio across profiles |
| `mean_abs_delta_sigma` | average absolute vol difference |
| `win_rate_return` | fraction of profiles where strategy A has higher return |
| `win_rate_sharpe` | fraction of profiles where strategy A has higher Sharpe |
| `win_rate_abs_delta_sigma` | fraction of profiles where strategy A has lower vol |

### summary.md

Markdown tables with the same content as result_summary.csv, split by index/product layer.

## Conventions

- **Functional style**: no classes, pure functions taking/returning DataFrames
- **Minimal deps**: pandas + numpy only; matplotlib only if plotting added later
- Portfolio types: `3.0`, `420_static`, `420_online`, `3.0_mapped_product`
- Index periods: 1y/3y/5y/10y/20y; product periods: 5y (limited by data availability)
- Lookback uses last N months from sorted history (`monthly[-n_months:]`)
- Adding a new strategy: add rows to `strategy_weights.csv` (must satisfy eligibility), then add `portfolio_monthly_returns()` + `compute_all_metrics()` + `compare_pair()` calls in main.py
