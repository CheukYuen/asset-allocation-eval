"""Generate mock input data for asset-allocation-eval."""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────
RISK_LEVELS = ["C1", "C2", "C3", "C4", "C5"]
LIFE_STAGES = [f"S{i}" for i in range(1, 8)]
ASSET_CLASSES = ["CASH", "BOND", "EQUITY", "ALT"]

# 生命周期波动率上限（参考 glide path 方法论）
SIGMA_STAGE_MAX = {
    "S1": 0.16,  # 刚毕业
    "S2": 0.15,  # 单身青年
    "S3": 0.14,  # 二人世界
    "S4": 0.14,  # 小孩学前
    "S5": 0.13,  # 小孩成年前
    "S6": 0.12,  # 子女成年
    "S7": 0.08,  # 退休
}

# 风险等级乘数
M_RISK = {
    "C1": 0.45,  # 保守型
    "C2": 0.60,  # 稳健型
    "C3": 0.75,  # 平衡型
    "C4": 0.90,  # 进取型
    "C5": 1.00,  # 激进型
}

# 最大回撤容忍度（仅由风险等级决定）
MAX_DD = {
    "C1": -0.05,
    "C2": -0.10,
    "C3": -0.15,
    "C4": -0.20,
    "C5": -0.25,
}

# 适当性约束：每个风险等级可投资的资产类别
ELIGIBLE_ASSETS = {
    "C1": {"CASH", "BOND"},
    "C2": {"CASH", "BOND", "ALT"},
    "C3": {"CASH", "BOND", "EQUITY", "ALT"},
    "C4": {"CASH", "BOND", "EQUITY", "ALT"},
    "C5": {"CASH", "BOND", "EQUITY", "ALT"},
}

# Products per asset class
PRODUCTS = {
    "CASH": ["MMF_001"],
    "BOND": ["BOND_001", "BOND_002"],
    "EQUITY": ["EQ_001", "EQ_002"],
    "ALT": ["ALT_001"],
}

# Monthly return params (mean, std) — annualized sense, will convert to monthly
ASSET_PARAMS = {
    "CASH":   (0.025, 0.005),
    "BOND":   (0.045, 0.040),
    "EQUITY": (0.090, 0.180),
    "ALT":    (0.065, 0.120),
}

INDEX_MONTHS = 240  # 20 years
PRODUCT_MONTHS = 60  # 5 years


# ── 1. client_profiles.csv ────────────────────────────────────────────────
def gen_client_profiles() -> pd.DataFrame:
    rows = []
    for r in RISK_LEVELS:
        for s in LIFE_STAGES:
            rows.append({
                "profile_id": f"{r}_{s}",
                "risk_level": r,
                "life_stage": s,
            })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "client_profiles.csv", index=False)
    print(f"client_profiles.csv: {len(df)} rows")
    return df


# ── 2. eligibility_matrix.csv ────────────────────────────────────────────
def gen_eligibility_matrix() -> pd.DataFrame:
    rows = []
    for r in RISK_LEVELS:
        for ac in ASSET_CLASSES:
            rows.append({
                "risk_level": r,
                "asset_class": ac,
                "eligible": 1 if ac in ELIGIBLE_ASSETS[r] else 0,
            })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "eligibility_matrix.csv", index=False)
    print(f"eligibility_matrix.csv: {len(df)} rows")
    return df


# ── 3. strategy_weights.csv ──────────────────────────────────────────────
def _risk_tilt(risk_level: str) -> dict[str, float]:
    """Return target allocation tilts based on risk level, respecting eligibility."""
    eligible = ELIGIBLE_ASSETS[risk_level]
    idx = RISK_LEVELS.index(risk_level)

    # Base tilts (before eligibility filtering)
    raw = {
        "CASH":   0.40 - idx * 0.08,   # 0.40 .. 0.08
        "BOND":   0.35 - idx * 0.05,   # 0.35 .. 0.15
        "EQUITY": 0.10 + idx * 0.15,   # 0.10 .. 0.70
        "ALT":    0.15,                 # residual
    }

    # Zero out ineligible assets, redistribute to eligible ones
    filtered = {ac: (v if ac in eligible else 0.0) for ac, v in raw.items()}
    total = sum(filtered.values())
    if total == 0:
        # Fallback: equal weight among eligible
        for ac in eligible:
            filtered[ac] = 1.0 / len(eligible)
    else:
        filtered = {ac: v / total for ac, v in filtered.items()}

    return filtered


def gen_strategy_weights(clients: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, c in clients.iterrows():
        pid = c["profile_id"]
        risk = c["risk_level"]
        eligible = ELIGIBLE_ASSETS[risk]
        tilt = _risk_tilt(risk)

        # ── Index layer: 3.0 ──
        noise_30 = {k: max(0.01, v + np.random.uniform(-0.03, 0.03)) if v > 0 else 0.0
                     for k, v in tilt.items()}
        total = sum(noise_30.values())
        for ac in ASSET_CLASSES:
            rows.append({
                "portfolio_type": "3.0",
                "profile_id": pid,
                "asset_class": ac,
                "product_code": "",
                "weight": round(noise_30[ac] / total, 6) if total > 0 else 0.0,
            })

        # ── Index layer: 420_static ──
        noise_420 = {k: max(0.01, v + np.random.uniform(-0.05, 0.05)) if v > 0 else 0.0
                      for k, v in tilt.items()}
        total = sum(noise_420.values())
        for ac in ASSET_CLASSES:
            rows.append({
                "portfolio_type": "420_static",
                "profile_id": pid,
                "asset_class": ac,
                "product_code": "",
                "weight": round(noise_420[ac] / total, 6) if total > 0 else 0.0,
            })

        # ── Product layer: 420_online ──
        prod_weights_420 = _gen_product_weights(tilt, eligible, jitter=0.05)
        for ac, prods in prod_weights_420.items():
            for pc, w in prods.items():
                rows.append({
                    "portfolio_type": "420_online",
                    "profile_id": pid,
                    "asset_class": ac,
                    "product_code": pc,
                    "weight": w,
                })

        # ── Product layer: 3.0_mapped_product ──
        prod_weights_30 = _gen_product_weights(tilt, eligible, jitter=0.03)
        for ac, prods in prod_weights_30.items():
            for pc, w in prods.items():
                rows.append({
                    "portfolio_type": "3.0_mapped_product",
                    "profile_id": pid,
                    "asset_class": ac,
                    "product_code": pc,
                    "weight": w,
                })

    df = pd.DataFrame(rows)

    # Remove zero-weight rows (ineligible assets)
    df = df[df["weight"] > 0].copy()

    # Fix rounding: normalize weights per (portfolio_type, profile_id)
    for (pt, pid), grp in df.groupby(["portfolio_type", "profile_id"]):
        total = grp["weight"].sum()
        df.loc[grp.index, "weight"] = grp["weight"] / total

    df["weight"] = df["weight"].round(6)
    df.to_csv(DATA_DIR / "strategy_weights.csv", index=False)
    print(f"strategy_weights.csv: {len(df)} rows")
    return df


def _gen_product_weights(
    tilt: dict, eligible: set, jitter: float,
) -> dict[str, dict[str, float]]:
    result = {}
    all_weights = []

    for ac in ASSET_CLASSES:
        if ac not in eligible or tilt[ac] == 0:
            continue
        prods = PRODUCTS[ac]
        base = max(0.01, tilt[ac] + np.random.uniform(-jitter, jitter))
        if len(prods) == 1:
            result[ac] = {prods[0]: base}
        else:
            split = np.random.dirichlet(np.ones(len(prods)))
            result[ac] = {p: round(base * s, 6) for p, s in zip(prods, split)}
        all_weights.extend(result[ac].values())

    # Normalize
    total = sum(all_weights)
    for ac in result:
        for pc in result[ac]:
            result[ac][pc] = round(result[ac][pc] / total, 6)

    return result


# ── 4. asset_returns.csv ──────────────────────────────────────────────────
def gen_asset_returns() -> pd.DataFrame:
    dates = pd.date_range("2006-01-31", periods=INDEX_MONTHS, freq="ME")
    rows = []
    for ac in ASSET_CLASSES:
        ann_mu, ann_sig = ASSET_PARAMS[ac]
        monthly_mu = ann_mu / 12
        monthly_sig = ann_sig / np.sqrt(12)
        rets = np.random.normal(monthly_mu, monthly_sig, INDEX_MONTHS)
        for d, r in zip(dates, rets):
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "asset_class": ac,
                "return": round(r, 8),
            })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "asset_returns.csv", index=False)
    print(f"asset_returns.csv: {len(df)} rows")
    return df


# ── 5. product_returns.csv ────────────────────────────────────────────────
def gen_product_returns() -> pd.DataFrame:
    dates = pd.date_range("2021-01-31", periods=PRODUCT_MONTHS, freq="ME")
    rows = []
    for ac, prods in PRODUCTS.items():
        ann_mu, ann_sig = ASSET_PARAMS[ac]
        monthly_mu = ann_mu / 12
        monthly_sig = ann_sig / np.sqrt(12)
        for pc in prods:
            # Add tracking error relative to index
            rets = np.random.normal(monthly_mu, monthly_sig, PRODUCT_MONTHS)
            tracking = np.random.normal(0, 0.002, PRODUCT_MONTHS)
            rets = rets + tracking
            for d, r in zip(dates, rets):
                rows.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "product_code": pc,
                    "asset_class": ac,
                    "return": round(r, 8),
                })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "product_returns.csv", index=False)
    print(f"product_returns.csv: {len(df)} rows")
    return df


# ── 6. risk_anchor.csv ───────────────────────────────────────────────────
def gen_risk_anchor(clients: pd.DataFrame) -> pd.DataFrame:
    """
    风险锚体系：
        σ_cap = σ_stage_max × m_risk
        σ_mid = 0.8 × σ_cap
        σ_min = 0.6 × σ_cap
    """
    rows = []
    for _, c in clients.iterrows():
        risk = c["risk_level"]
        stage = c["life_stage"]
        sigma_stage_max = SIGMA_STAGE_MAX[stage]
        m_risk = M_RISK[risk]

        sigma_cap = sigma_stage_max * m_risk
        sigma_mid = 0.8 * sigma_cap
        sigma_min = 0.6 * sigma_cap

        rows.append({
            "profile_id": c["profile_id"],
            "risk_level": risk,
            "life_stage": stage,
            "sigma_min": round(sigma_min, 6),
            "sigma_mid": round(sigma_mid, 6),
            "sigma_max": round(sigma_cap, 6),
            "max_drawdown_tolerance": MAX_DD[risk],
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "risk_anchor.csv", index=False)
    print(f"risk_anchor.csv: {len(df)} rows")
    return df


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating mock data...\n")
    clients = gen_client_profiles()
    gen_eligibility_matrix()
    gen_strategy_weights(clients)
    gen_asset_returns()
    gen_product_returns()
    gen_risk_anchor(clients)
    print("\nDone. Files saved to data/")
