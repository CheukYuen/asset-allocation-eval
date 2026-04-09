# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Offline evaluation framework comparing smart portfolio strategies (智能投顾 3.0 vs 420). Two comparison layers:
- **Index layer**: 3.0 vs 420_static — asset class weights × index returns
- **Product layer**: 3.0_mapped_product vs 420_online — product weights × product returns

Both layers use buy-and-hold (static weights, no rebalancing) across 35 client segments (C1–C5 × S1–S7).

## Commands

```bash
python3 build_asset_returns.py     # build data/asset_returns.csv from 建模月频序列.csv (Total Return口径)
python3 build_strategy_weights.py  # replace 3.0 weights in data/strategy_weights.csv with real weights from 420/
python3 build_ai_strategy_weights.py  # build 3.0 (AI weights) + 420_static into strategy_weights.csv
python3 run_index_comparison.py    # index layer only: 3.0 vs 420_static → output/index_3.0_vs_420/
python3 generate_mock.py           # generate mock data into data/ (idempotent, seed 42) — WARNING: overwrites real data
python3 main.py                    # run full evaluation (both layers), outputs to output/
pip3 install -r requirements.txt   # pandas, numpy only
```

## Architecture

Data flows linearly: `data/*.csv → load → calc → compare → report → output/`

- **main.py** — orchestrates the pipeline: load data, compute portfolio returns for each layer, calculate metrics, compare strategy pairs, save results
- **src/load.py** — reads 6 input CSVs, validates weight sums to 1 and eligibility constraints
- **src/calc.py** — `portfolio_monthly_returns()` joins weights with returns via `asset_class` (index) or `product_code` (product); `compute_all_metrics()` computes annualized return/vol/sharpe/max drawdown per profile per lookback period
- **src/compare.py** — `compare_pair()` merges two strategies on (profile_id, period), computes deltas; `summarize()` aggregates means and win rates
- **src/report.py** — saves CSVs and markdown summary to `output/`
- **build_asset_returns.py** — builds `data/asset_returns.csv` (wide format: month, CASH, BOND, EQUITY, ALT) from `data/建模月频序列.csv`; `src/load.py` melts it to long format on load
- **build_strategy_weights.py** — replaces the `3.0` portion of `data/strategy_weights.csv` with real weights from `420/420_growth_clients_35_minimal.csv` (maps lifecycle→S1–S7, risk_level→C1–C5, commodity→ALT)
- **build_ai_strategy_weights.py** — builds `3.0` (AI-extracted weights from `AI-invest/outputs/extracted_weights_v3.csv`) + `420_static` (original 420 weights) into `data/strategy_weights.csv`; normalizes weights to sum=1, handles failed extractions
- **run_index_comparison.py** — runs index layer comparison only (3.0 vs 420_static), outputs to `output/index_3.0_vs_420/`
- **generate_mock.py** — standalone script producing all 6 input CSVs with realistic distributions (WARNING: overwrites real data)

Key join: index layer joins on `asset_class`, product layer joins on `product_code`. The `join_col` parameter in `portfolio_monthly_returns()` controls this.

## Key Domain Concepts

- **profile_id**: client persona key = `{risk_level}_{life_stage}` (e.g. `C3_S2`), not a real customer ID
- **Risk anchor (风险锚)**: per-profile risk container — `sigma_min / sigma_mid / sigma_max` computed from `σ_stage_max × m_risk` formulas; strategy-agnostic referee layer
- **Eligibility (适当性约束)**: `eligibility_matrix.csv` defines which asset classes each risk level may invest in (C1: CASH/BOND only; C2: no EQUITY; C3–C5: all four)
- **Δσ = annualized_vol − sigma_mid**: core risk-fit metric; smaller |Δσ| = better client fit

## Data Provenance (Index Layer)

| 数据 | 来源 | 状态 |
|------|------|------|
| asset_returns.csv | `data/建模月频序列.csv` → `build_asset_returns.py` | ✅ 真实数据 |
| 3.0 权重 (strategy_weights.csv) | `420/420_growth_clients_35_minimal.csv` → `build_strategy_weights.py` | ✅ 真实数据 |
| 420_static 权重 (strategy_weights.csv) | — | ❌ 仍为 mock，待替换 |

Asset return口径 (Total Return):
- CASH: `CBA02201`, BOND: `CBOND_NEW_COMPOSITE_WEALTH`, EQUITY: `CSI300_TR`
- ALT: `70% × AU9999 + 30% × NHCI`
- 月收益率: `P_t / P_{t-1} - 1`

## Output Files Explained

Comparison results are organized by layer under `output/`:
- `output/index_3.0_vs_420/` — index layer: 3.0 (AI) vs 420_static

### result_detail_index.csv

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

### result_summary_index.csv

One row per period. Aggregates across all 35 profiles:

| Metric | Definition |
|--------|-----------|
| `mean_return_*` | average annualized return across profiles |
| `mean_vol_*` | average annualized volatility across profiles |
| `mean_sharpe_*` | average Sharpe ratio across profiles |
| `mean_abs_delta_sigma` | average absolute vol difference |
| `win_rate_return` | fraction of profiles where strategy A has higher return |
| `win_rate_sharpe` | fraction of profiles where strategy A has higher Sharpe |
| `win_rate_abs_delta_sigma` | fraction of profiles where strategy A has lower vol |

### summary_index.md

Markdown table with the same content as result_summary_index.csv.

## AI-invest: Two-Stage Batch Generation

`AI-invest/` contains a standalone batch pipeline that calls LLMs to generate and extract asset allocation proposals for the same 35 client segments.

### How it works

1. **Stage 1** (`qwen3-235b-a22b-instruct-2507`): For each of the 35 clients, injects their profile (risk_level, lifecycle, allowed_asset_classes) into the system prompt template (`市场分析-提示词模板.md`) and generates a full allocation proposal in natural language.
2. **Stage 2** (`qwen3.6-plus`): Extracts the "final recommended allocation" (CASH/BOND/EQUITY/ALT weights summing to 100) from each Stage 1 output as structured JSON.

### Commands

```bash
cd AI-invest
pip install -r requirements.txt       # openai, python-dotenv
python3 batch_generate_allocations.py  # runs both stages, outputs to AI-invest/outputs/
```

### Key files

- `AI-invest/batch_generate_allocations.py` — main script (two-stage pipeline)
- `AI-invest/市场分析-提示词模板.md` — Stage 1 system prompt template (macro data + client profile placeholder)
- `420/420_growth_clients_35_minimal.csv` — 35-row real 3.0 weights (id, lifecycle, risk_level, cash_pct/bond_pct/equity_pct/commodity_pct)

### Outputs (in `AI-invest/outputs/`)

| File | Content |
|------|---------|
| `stage1_raw_generations*.jsonl` | 35 full LLM outputs with metadata |
| `stage2_extracted_weights*.jsonl` | 35 extraction results with raw stage2 output |
| `extracted_weights*.csv` | Final comparison table: original 420 weights vs AI-extracted weights |

### Eligibility rules (适当性约束)

Applied when building `allowed_asset_classes` per client:
- C1: `["CASH", "BOND"]`
- C2: `["CASH", "BOND", "ALT"]`
- C3/C4/C5: `["CASH", "BOND", "EQUITY", "ALT"]`

### Stage 2 parser robustness

The `parse_weights()` function handles multiple LLM output formats:
- Bare JSON, code-fenced JSON, JSON with surrounding text
- Percent strings (`"70%"`), string numbers (`"70"`), 0–1 decimals (`0.7`)
- Auto-detects ratio mode (sum ≈ 1) vs percentage mode (sum ≈ 100) and normalizes
- Tolerance: weight_sum in `[99.5, 100.5]` → `success`

### Environment

Reads from `.env`:
- `DASHSCOPE_API_KEY` (or `OPENAI_API_KEY`) — API key
- `OPENAI_BASE_URL` — defaults to DashScope compatible endpoint

## Conventions

- **Functional style**: no classes, pure functions taking/returning DataFrames
- **Minimal deps**: pandas + numpy only; matplotlib only if plotting added later
- Portfolio types: `3.0`, `420_static`, `420_online`, `3.0_mapped_product`
- Index periods: 1y/3y/5y/10y/20y; product periods: 5y (limited by data availability)
- Lookback uses last N months from sorted history (`monthly[-n_months:]`)
- Adding a new strategy: add rows to `strategy_weights.csv` (must satisfy eligibility), then add `portfolio_monthly_returns()` + `compute_all_metrics()` + `compare_pair()` calls in main.py
