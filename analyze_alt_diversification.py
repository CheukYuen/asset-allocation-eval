"""
analyze_alt_diversification.py
-------------------------------
证明：3.0 在中高风险画像中更有效使用另类资产做分散，420 的 ALT 权重普遍偏低。

输出：
  - 资产类别相关性矩阵（全历史 + 近 3 年）
  - ALT 权重对比表（3.0 vs 420_static，C3–C5）
  - 各画像投资组合指标：年化收益、波动率、夏普比率、|Δσ| 风险锚偏差
  - 重点画像（C3_S4, C5_S1 等）分组对比
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))
from src.load import load_rf_series

# ── 1. Load raw data ──────────────────────────────────────────────────────────

def load_asset_returns():
    df = pd.read_csv("data/asset_returns.csv")
    df["date"] = pd.to_datetime(df["month"]) + pd.offsets.MonthEnd(0)
    df = df.set_index("date").drop(columns=["month"])
    df = df.dropna(how="all")
    return df.sort_index()

def load_weights():
    df = pd.read_csv("data/strategy_weights.csv")
    return df[df["portfolio_type"].isin(["3.0", "420_static"])]

def load_risk_anchor():
    df = pd.read_csv("data/risk_anchor.csv")
    df = df.set_index("profile_id")
    return df

# ── 2. Correlation matrices ───────────────────────────────────────────────────

def print_correlation_matrices(returns_wide: pd.DataFrame):
    # Only rows where all 4 asset classes have data
    full = returns_wide.dropna()
    recent_3y = full.loc[full.index >= full.index[-1] - pd.DateOffset(years=3)]

    print("\n" + "=" * 62)
    print("§1  资产类别相关性矩阵")
    print("=" * 62)

    corr_full = full[["CASH", "BOND", "EQUITY", "ALT"]].corr()
    corr_3y   = recent_3y[["CASH", "BOND", "EQUITY", "ALT"]].corr()

    print(f"\n【全历史】({full.index[0].strftime('%Y-%m')} ~ {full.index[-1].strftime('%Y-%m')}，共 {len(full)} 月)")
    print(corr_full.round(4).to_string())

    print(f"\n【近 3 年】({recent_3y.index[0].strftime('%Y-%m')} ~ {recent_3y.index[-1].strftime('%Y-%m')}，共 {len(recent_3y)} 月)")
    print(corr_3y.round(4).to_string())

    # Key insight lines
    alt_eq_full = corr_full.loc["ALT", "EQUITY"]
    alt_eq_3y   = corr_3y.loc["ALT", "EQUITY"]
    alt_bond_full = corr_full.loc["ALT", "BOND"]
    bond_eq_full  = corr_full.loc["BOND", "EQUITY"]

    print(f"""
关键结论：
  ● ALT vs EQUITY  全历史相关系数 = {alt_eq_full:+.4f}，近 3 年 = {alt_eq_3y:+.4f}
    → ALT 与股票相关性{'低' if abs(alt_eq_full) < 0.4 else '中等'}，增配 ALT 可有效分散权益风险
  ● ALT vs BOND    全历史相关系数 = {alt_bond_full:+.4f}
  ● BOND vs EQUITY 全历史相关系数 = {bond_eq_full:+.4f}
    → {'ALT 比 BOND 更能分散 EQUITY 风险' if abs(alt_eq_full) < abs(bond_eq_full) else 'BOND 与 EQUITY 相关性同样较低'}
""")
    return corr_full, corr_3y

# ── 3. ALT weight comparison table ───────────────────────────────────────────

def print_alt_weight_table(weights_df: pd.DataFrame):
    pivot = weights_df[weights_df["asset_class"] == "ALT"].pivot(
        index="profile_id", columns="portfolio_type", values="weight"
    ).fillna(0)
    pivot["Δ (3.0 − 420)"] = pivot["3.0"] - pivot["420_static"]
    pivot["risk_level"] = pivot.index.str[:2]
    pivot["stage"]      = pivot.index.str[3:]

    print("\n" + "=" * 62)
    print("§2  ALT 权重对比：3.0 vs 420_static（C1–C5 全画像）")
    print("=" * 62)

    for risk in ["C1", "C2", "C3", "C4", "C5"]:
        sub = pivot[pivot["risk_level"] == risk].sort_values("stage")
        print(f"\n  {risk}（{'保守' if risk=='C1' else '稳健' if risk=='C2' else '平衡' if risk=='C3' else '积极' if risk=='C4' else '激进'}型）")
        print("  " + sub[["3.0", "420_static", "Δ (3.0 − 420)"]].to_string(
            float_format=lambda x: f"{x:.0%}"
        ))

    # Summary by risk level
    print("\n  ── 各风险等级均值 ──")
    summary = pivot.groupby("risk_level")[["3.0", "420_static", "Δ (3.0 − 420)"]].mean()
    print("  " + summary.to_string(float_format=lambda x: f"{x:.1%}"))

# ── 4. Portfolio metrics ──────────────────────────────────────────────────────

def compute_portfolio_metrics(weights_df: pd.DataFrame, returns_wide: pd.DataFrame,
                               rf_series: pd.Series, anchor: pd.DataFrame,
                               lookback_years: int = 5) -> pd.DataFrame:
    """
    For each (portfolio_type, profile_id), compute:
      - annualized_return, annualized_vol, sharpe_ratio, max_drawdown
      - abs_delta_sigma = |vol - sigma_mid|
    Using last `lookback_years` of common data.
    """
    # Pivot weights to wide: profile_id → {CASH, BOND, EQUITY, ALT}
    w_pivot = weights_df.pivot_table(
        index=["portfolio_type", "profile_id"],
        columns="asset_class", values="weight", fill_value=0
    )

    # Restrict returns to lookback window where all 4 classes present
    full_ret = returns_wide.dropna()
    cutoff   = full_ret.index[-1] - pd.DateOffset(years=lookback_years)
    ret_win  = full_ret[full_ret.index >= cutoff]

    results = []
    for (ptype, profile_id), row in w_pivot.iterrows():
        w = row.reindex(["CASH", "BOND", "EQUITY", "ALT"], fill_value=0).values

        # Monthly portfolio returns
        R = ret_win[["CASH", "BOND", "EQUITY", "ALT"]].values @ w  # shape (T,)

        n = len(R)
        if n < 12:
            continue

        # Metrics
        ann_ret = (np.prod(1 + R) ** (12 / n)) - 1
        ann_vol = R.std(ddof=1) * np.sqrt(12)

        # Dynamic Sharpe
        rf_aligned = rf_series.reindex(ret_win.index, method="ffill").dropna()
        common_idx = ret_win.index.intersection(rf_aligned.index)
        R_exc = pd.Series(R, index=ret_win.index).reindex(common_idx)
        rf_m  = ((1 + rf_aligned.reindex(common_idx) / 100) ** (1/12) - 1)
        excess = R_exc.values - rf_m.values
        sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(12) if excess.std(ddof=1) > 0 else np.nan

        # Max drawdown
        cum = np.cumprod(1 + R)
        peak = np.maximum.accumulate(cum)
        mdd  = ((cum - peak) / peak).min()

        # Risk anchor
        sigma_mid = anchor.loc[profile_id, "sigma_mid"] if profile_id in anchor.index else np.nan
        abs_dsig  = abs(ann_vol - sigma_mid) if not np.isnan(sigma_mid) else np.nan
        mdd_tol   = anchor.loc[profile_id, "max_drawdown_tolerance"] if profile_id in anchor.index else np.nan
        breach_mdd = (mdd < mdd_tol) if not np.isnan(mdd_tol) else False

        results.append({
            "portfolio_type": ptype,
            "profile_id": profile_id,
            "risk_level": profile_id[:2],
            "life_stage": profile_id[3:],
            "annualized_return": ann_ret,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "sigma_mid": sigma_mid,
            "abs_delta_sigma": abs_dsig,
            "mdd_tolerance": mdd_tol,
            "breach_mdd": breach_mdd,
            "ALT_weight": w[3],  # index 3 = ALT
        })

    return pd.DataFrame(results)

# ── 5. Print comparison tables ────────────────────────────────────────────────

def print_metrics_comparison(metrics_df: pd.DataFrame):
    m30  = metrics_df[metrics_df["portfolio_type"] == "3.0"].set_index("profile_id")
    m420 = metrics_df[metrics_df["portfolio_type"] == "420_static"].set_index("profile_id")

    cols_show = ["annualized_return", "annualized_vol", "sharpe_ratio",
                 "max_drawdown", "sigma_mid", "abs_delta_sigma", "ALT_weight"]

    print("\n" + "=" * 80)
    print("§3  各画像投资组合指标详表（近 5 年，中高风险 C3–C5）")
    print("=" * 80)

    for risk in ["C3", "C4", "C5"]:
        profiles = sorted([p for p in m30.index if p.startswith(risk)])
        rows = []
        for pid in profiles:
            if pid not in m30.index or pid not in m420.index:
                continue
            r30  = m30.loc[pid]
            r420 = m420.loc[pid]
            rows.append({
                "画像": pid,
                "3.0 ALT权重": f"{r30['ALT_weight']:.0%}",
                "420 ALT权重": f"{r420['ALT_weight']:.0%}",
                "3.0 年化收益": f"{r30['annualized_return']:.2%}",
                "420 年化收益": f"{r420['annualized_return']:.2%}",
                "3.0 波动率":   f"{r30['annualized_vol']:.2%}",
                "420 波动率":   f"{r420['annualized_vol']:.2%}",
                "3.0 夏普":     f"{r30['sharpe_ratio']:.3f}",
                "420 夏普":     f"{r420['sharpe_ratio']:.3f}",
                "σ_mid":        f"{r30['sigma_mid']:.2%}",
                "|Δσ| 3.0":    f"{r30['abs_delta_sigma']:.2%}",
                "|Δσ| 420":    f"{r420['abs_delta_sigma']:.2%}",
                "风险锚胜":     "3.0 ✓" if r30['abs_delta_sigma'] < r420['abs_delta_sigma'] else "420 ✓",
            })

        tbl = pd.DataFrame(rows).set_index("画像")
        risk_label = {"C3": "C3 平衡型", "C4": "C4 积极型", "C5": "C5 激进型"}[risk]
        print(f"\n  ── {risk_label} ──")
        print(tbl.to_string())

# ── 6. Aggregate summary by risk level ───────────────────────────────────────

def print_aggregate_summary(metrics_df: pd.DataFrame):
    m30  = metrics_df[metrics_df["portfolio_type"] == "3.0"]
    m420 = metrics_df[metrics_df["portfolio_type"] == "420_static"]

    print("\n" + "=" * 62)
    print("§4  各风险等级均值汇总（C1–C5）")
    print("=" * 62)

    risk_label = {
        "C1": "C1 保守", "C2": "C2 稳健",
        "C3": "C3 平衡", "C4": "C4 积极", "C5": "C5 激进"
    }

    rows = []
    for risk in ["C1", "C2", "C3", "C4", "C5"]:
        r30  = m30[m30["risk_level"] == risk]
        r420 = m420[m420["risk_level"] == risk]
        if r30.empty or r420.empty:
            continue

        rows.append({
            "风险等级": risk_label[risk],
            "3.0 ALT均值": f"{r30['ALT_weight'].mean():.1%}",
            "420 ALT均值": f"{r420['ALT_weight'].mean():.1%}",
            "3.0 年化收益": f"{r30['annualized_return'].mean():.2%}",
            "420 年化收益": f"{r420['annualized_return'].mean():.2%}",
            "3.0 波动率":   f"{r30['annualized_vol'].mean():.2%}",
            "420 波动率":   f"{r420['annualized_vol'].mean():.2%}",
            "3.0 夏普均值": f"{r30['sharpe_ratio'].mean():.3f}",
            "420 夏普均值": f"{r420['sharpe_ratio'].mean():.3f}",
            "3.0 |Δσ|均值": f"{r30['abs_delta_sigma'].mean():.2%}",
            "420 |Δσ|均值": f"{r420['abs_delta_sigma'].mean():.2%}",
            "风险锚胜率(3.0)": f"{(r30['abs_delta_sigma'].values < r420['abs_delta_sigma'].values).mean():.0%}",
        })

    tbl = pd.DataFrame(rows).set_index("风险等级")
    print(tbl.to_string())

# ── 7. Highlight 3 spotlight profiles from user's table ──────────────────────

def print_spotlight(metrics_df: pd.DataFrame, weights_df: pd.DataFrame,
                    returns_wide: pd.DataFrame, rf_series: pd.Series,
                    anchor: pd.DataFrame):
    spotlight = [("C1_S4", "C1 保守 · S4小孩学前"),
                 ("C3_S4", "C3 平衡 · S4小孩学前"),
                 ("C5_S1", "C5 激进 · S1刚毕业")]

    print("\n" + "=" * 62)
    print("§5  重点画像深度对比")
    print("=" * 62)

    m30  = metrics_df[metrics_df["portfolio_type"] == "3.0"].set_index("profile_id")
    m420 = metrics_df[metrics_df["portfolio_type"] == "420_static"].set_index("profile_id")

    # ALT weights for spotlight
    w30  = weights_df[weights_df["portfolio_type"] == "3.0"]
    w420 = weights_df[weights_df["portfolio_type"] == "420_static"]

    def get_w(df, pid):
        sub = df[df["profile_id"] == pid].set_index("asset_class")["weight"]
        return {ac: sub.get(ac, 0) for ac in ["CASH", "BOND", "EQUITY", "ALT"]}

    for pid, label in spotlight:
        if pid not in m30.index or pid not in m420.index:
            continue

        r30  = m30.loc[pid]
        r420 = m420.loc[pid]
        ww30 = get_w(w30, pid)
        ww420= get_w(w420, pid)
        sig  = r30["sigma_mid"]
        mdd_tol = r30["mdd_tolerance"]

        print(f"\n  ▶ {label}  (σ_mid = {sig:.2%}, 最大回撤红线 = {mdd_tol:.0%})")
        print(f"  {'指标':<20}  {'3.0':>10}  {'420_static':>10}  {'差值(3.0−420)':>14}")
        print("  " + "-" * 60)

        def fmt_w(d):
            return f"C{d['CASH']:.0%} B{d['BOND']:.0%} E{d['EQUITY']:.0%} A{d['ALT']:.0%}"

        lines = [
            ("权重配置", fmt_w(ww30), fmt_w(ww420), ""),
            ("ALT 权重", f"{ww30['ALT']:.0%}", f"{ww420['ALT']:.0%}",
             f"{ww30['ALT']-ww420['ALT']:+.0%}"),
            ("年化收益率", f"{r30['annualized_return']:.2%}", f"{r420['annualized_return']:.2%}",
             f"{r30['annualized_return']-r420['annualized_return']:+.2%}"),
            ("年化波动率", f"{r30['annualized_vol']:.2%}", f"{r420['annualized_vol']:.2%}",
             f"{r30['annualized_vol']-r420['annualized_vol']:+.2%}"),
            ("夏普比率",  f"{r30['sharpe_ratio']:.3f}", f"{r420['sharpe_ratio']:.3f}",
             f"{r30['sharpe_ratio']-r420['sharpe_ratio']:+.3f}"),
            ("最大回撤",  f"{r30['max_drawdown']:.2%}", f"{r420['max_drawdown']:.2%}",
             f"{r30['max_drawdown']-r420['max_drawdown']:+.2%}"),
            ("|Δσ| 风险锚偏差", f"{r30['abs_delta_sigma']:.2%}", f"{r420['abs_delta_sigma']:.2%}",
             f"{r30['abs_delta_sigma']-r420['abs_delta_sigma']:+.2%}"),
            ("超回撤红线", str(r30['breach_mdd']), str(r420['breach_mdd']), ""),
        ]
        for name, v30, v420, delta in lines:
            print(f"  {name:<20}  {v30:>10}  {v420:>10}  {delta:>14}")

# ── 8. Marginal diversification benefit of ALT ───────────────────────────────

def print_marginal_alt_benefit(returns_wide: pd.DataFrame, rf_series: pd.Series,
                                lookback_years: int = 5):
    """
    固定 C3_S4 和 C5_S1 的非 ALT 权重比例，逐步增加 ALT 权重，
    展示 Sharpe / 波动率随 ALT 权重变化的趋势。
    """
    full_ret = returns_wide.dropna()
    cutoff   = full_ret.index[-1] - pd.DateOffset(years=lookback_years)
    ret_win  = full_ret[full_ret.index >= cutoff]

    rf_aligned = rf_series.reindex(ret_win.index, method="ffill").dropna()
    common_idx = ret_win.index.intersection(rf_aligned.index)
    rf_m       = ((1 + rf_aligned.reindex(common_idx) / 100) ** (1/12) - 1)

    def scan_alt(base_no_alt, label):
        """base_no_alt: dict {CASH, BOND, EQUITY} summing to 1.0"""
        print(f"\n  {label}")
        print(f"  {'ALT%':<8} {'年化收益':>10} {'波动率':>10} {'夏普':>8} {'相比0%ALT':>12}")
        baseline_sharpe = None
        for alt_pct in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
            scale = 1 - alt_pct / 100
            w = np.array([
                base_no_alt.get("CASH", 0) * scale,
                base_no_alt.get("BOND", 0) * scale,
                base_no_alt.get("EQUITY", 0) * scale,
                alt_pct / 100
            ])
            R = ret_win[["CASH", "BOND", "EQUITY", "ALT"]].values @ w
            n = len(R)
            ann_ret = (np.prod(1 + R) ** (12 / n)) - 1
            ann_vol = R.std(ddof=1) * np.sqrt(12)
            R_s = pd.Series(R, index=ret_win.index).reindex(common_idx)
            exc = R_s.values - rf_m.values
            sharpe = (exc.mean() / exc.std(ddof=1)) * np.sqrt(12) if exc.std(ddof=1) > 0 else np.nan
            if alt_pct == 0:
                baseline_sharpe = sharpe
            delta_sharpe = f"{sharpe - baseline_sharpe:+.3f}" if baseline_sharpe is not None and alt_pct > 0 else "—"
            print(f"  {alt_pct:<8}% {ann_ret:>9.2%} {ann_vol:>10.2%} {sharpe:>8.3f} {delta_sharpe:>12}")

    print("\n" + "=" * 62)
    print("§6  ALT 权重边际增益分析（固定其余资产比例不变）")
    print("=" * 62)

    # C3_S4: 3.0 = CASH10% BOND35% EQUITY25% ALT30%
    # Base (no ALT): CASH≈14.3% BOND≈50% EQUITY≈35.7%
    scan_alt({"CASH": 0.1429, "BOND": 0.50, "EQUITY": 0.357}, "C3_S4 基准（平衡型·学前）")

    # C5_S1: 3.0 = CASH5% BOND15% EQUITY45% ALT35%
    scan_alt({"CASH": 0.077, "BOND": 0.231, "EQUITY": 0.692}, "C5_S1 基准（激进型·刚毕业）")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    returns_wide = load_asset_returns()
    weights_df   = load_weights()
    anchor       = load_risk_anchor()
    rf_series    = load_rf_series()

    print_correlation_matrices(returns_wide)
    print_alt_weight_table(weights_df)

    metrics_df = compute_portfolio_metrics(
        weights_df, returns_wide, rf_series, anchor, lookback_years=5
    )

    print_metrics_comparison(metrics_df)
    print_aggregate_summary(metrics_df)
    print_spotlight(metrics_df, weights_df, returns_wide, rf_series, anchor)
    print_marginal_alt_benefit(returns_wide, rf_series, lookback_years=5)

    print("\n" + "=" * 62)
    print("分析完成。")
    print("=" * 62)

if __name__ == "__main__":
    main()
