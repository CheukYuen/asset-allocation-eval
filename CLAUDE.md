# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Offline evaluation framework comparing smart portfolio strategies (智能投顾 3.0 vs 420). Two comparison layers:
- **Index layer**: 3.0 vs 420_static — asset class weights × index returns
- **Product layer**: 3.0_mapped_product vs 420_online — product weights × product returns ⚠️ **暂停** — 依赖 `product_returns.csv`（真实产品净值）和 product_code 粒度的权重，数据尚未就绪

Both layers use buy-and-hold (static weights, no rebalancing) across 35 client segments (C1–C5 × S1–S7).

## Commands

```bash
python3 build_asset_returns.py     # build data/asset_returns.csv + data/rf_series.csv from 建模月频序列.csv
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
- **src/load.py** — reads input CSVs, validates weight sums to 1 and eligibility constraints; `load_rf_series()` loads `data/rf_series.csv` as a date-indexed CGB_1Y Series for dynamic Sharpe
- **src/calc.py** — `portfolio_monthly_returns()` joins weights with returns via `asset_class`; `compute_all_metrics()` computes annualized return/vol/sharpe/max drawdown per profile per lookback period; accepts optional `rf_series` for dynamic Sharpe (see below)
- **src/compare.py** — `compare_pair()` merges two strategies on (profile_id, period), computes deltas; `summarize()` aggregates means and win rates
- **src/report.py** — saves CSVs and markdown summary to `output/`
- **build_asset_returns.py** — builds `data/asset_returns.csv` (wide format: month, CASH, BOND, EQUITY, ALT) and `data/rf_series.csv` (month, CGB_1Y) from `data/建模月频序列.csv`; `src/load.py` melts asset_returns to long format on load
- **build_strategy_weights.py** — replaces the `3.0` portion of `data/strategy_weights.csv` with real weights from `420/420_growth_clients_35_minimal.csv` (maps lifecycle→S1–S7, risk_level→C1–C5, commodity→ALT)
- **build_ai_strategy_weights.py** — builds `3.0` (AI-extracted weights from `AI-invest/outputs/extracted_weights_v3.csv`) + `420_static` (original 420 weights) into `data/strategy_weights.csv`; normalizes weights to sum=1, handles failed extractions
- **run_index_comparison.py** — runs index layer comparison only (3.0 vs 420_static), outputs to `output/index_3.0_vs_420/`
- **generate_mock.py** — standalone script producing all 6 input CSVs with realistic distributions (WARNING: overwrites real data)

Key join: both layers join on `asset_class`. `product_returns.csv` is wide format (date, CASH, BOND, EQUITY, ALT) with values averaged across products per asset class per month; `src/load.py` melts it to long format on load.

## Key Domain Concepts

- **profile_id**: client persona key = `{risk_level}_{life_stage}` (e.g. `C3_S2`), not a real customer ID
- **Risk anchor (风险锚)**: per-profile risk container — `sigma_min / sigma_mid / sigma_max` computed from `σ_stage_max × m_risk` formulas; strategy-agnostic referee layer
- **Eligibility (适当性约束)**: `eligibility_matrix.csv` defines which asset classes each risk level may invest in (C1: CASH/BOND only; C2: no EQUITY; C3–C5: all four)
- **Δσ = annualized_vol − sigma_mid**: core risk-fit metric; smaller |Δσ| = better client fit

## Data Provenance (Index Layer)

| 数据 | 来源 | 状态 |
|------|------|------|
| asset_returns.csv | `data/建模月频序列.csv` → `build_asset_returns.py` | ✅ 真实数据 |
| rf_series.csv | `data/建模月频序列.csv` → `build_asset_returns.py`（CGB_1Y列，ffill） | ✅ 真实数据 |
| 3.0 权重 (strategy_weights.csv) | `420/420_growth_clients_35_minimal.csv` → `build_strategy_weights.py` | ✅ 真实数据 |
| 420_static 权重 (strategy_weights.csv) | — | ❌ 仍为 mock，待替换 |

Asset return口径 (Total Return):
- CASH: `CBA02201`, BOND: `CBOND_NEW_COMPOSITE_WEALTH`, EQUITY: `CSI300_TR`
- ALT: `70% × AU9999 + 30% × NHCI`
- 月收益率: `P_t / P_{t-1} - 1`

## Output Files Explained

Comparison results are organized by layer under `output/`:
- `output/index_3.0_vs_420/` — index layer: 3.0 (AI) vs 420_static

Output follows a **正文主表 + 胜率表**两层结构:

### result_detail_index.csv

Per-profile, per-period comparison. Each row = one profile × one lookback period. Columns include metrics for both strategies (suffixed `_3.0` / `_420_static` etc.) plus deltas and risk-anchor columns:

| Metric | Definition |
|--------|-----------|
| `annualized_return` | `prod(1+r)^(12/n) - 1` — compounded monthly returns annualized |
| `annualized_vol` | `std(monthly, ddof=1) × √12` — annualized standard deviation |
| `sharpe_ratio` | `mean(R_p - rf_t) / std(R_p - rf_t, ddof=1) × √12` — 逐月超额收益法；rf_t 来自 CGB_1Y 动态序列，`(1 + CGB_1Y_t/100)^(1/12) - 1` |
| `max_drawdown` | largest peak-to-trough decline in cumulative return series (negative value) |
| `delta_return` | strategy A return minus strategy B return |
| `delta_sigma` | strategy A vol minus strategy B vol (positive = A is riskier) |
| `abs_delta_sigma_3.0` | `\|vol_3.0 − σ_mid\|` — 3.0 策略波动率与客户风险中枢的偏差 |
| `abs_delta_sigma_420_static` | `\|vol_420 − σ_mid\|` — 420_static 策略波动率与客户风险中枢的偏差 |
| `sigma_mid` | 该画像的风险中枢目标波动率（来自 risk_anchor.csv） |
| `max_drawdown_tolerance` | 该画像的最大回撤红线（来自 risk_anchor.csv，负值） |

### result_main_index.csv（正文主表）

One row per period. Shows absolute performance levels across all 35 profiles:

| Metric | Definition |
|--------|-----------|
| `mean_return_*` | average annualized return across profiles |
| `mean_sharpe_*` | average Sharpe ratio across profiles |
| `mean_abs_delta_sigma_*` | 平均风险偏差 mean(`\|vol − σ_mid\|`)，越小越匹配客户 |
| `exceed_rate_maxdd_*` | 超回撤红线画像占比 — `actual_mdd < max_drawdown_tolerance` 的画像比例 |

### result_winrate_index.csv（胜率表）

One row per period. Shows how broadly strategy A (3.0) beats strategy B (420_static):

| Metric | Definition |
|--------|-----------|
| `win_rate_return` | fraction of profiles where 3.0 has higher return |
| `win_rate_sharpe` | fraction of profiles where 3.0 has higher Sharpe |
| `win_rate_risk_match` | 3.0 风险匹配更优的画像占比（逐画像比较 `\|vol − σ_mid\|`，更小者胜） |

### summary_index.md

Markdown with both tables (正文主表 + 胜率表).

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
- **Sharpe rf 口径**: 动态 CGB_1Y（`data/rf_series.csv`），逐月超额收益法；`compute_all_metrics()` 必须传入 `rf_series=load_rf_series()`，禁止固定常数 rf
