"""
Generate comparison charts for 智能投顾 3.0 vs 420_static index layer evaluation.
Output: output/index_3.0_vs_420/charts/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Config ────────────────────────────────────────────────────────────────────
MAIN_CSV = "output/index_3.0_vs_420/result_main_index.csv"
WINRATE_CSV = "output/index_3.0_vs_420/result_winrate_index.csv"
OUT_DIR = "output/index_3.0_vs_420/charts"

COLOR_30 = "#1f77b4"      # 3.0  深蓝实线
COLOR_420 = "#999999"     # 420  中灰虚线
PERIOD_ORDER = ["1y", "3y", "5y", "10y", "20y"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    main = pd.read_csv(MAIN_CSV)
    wr = pd.read_csv(WINRATE_CSV)
    main["window"] = pd.Categorical(main["window"], categories=PERIOD_ORDER, ordered=True)
    wr["window"] = pd.Categorical(wr["window"], categories=PERIOD_ORDER, ordered=True)
    main = main.sort_values("window").reset_index(drop=True)
    wr = wr.sort_values("window").reset_index(drop=True)
    return main, wr


def dual_line(ax, x, y_a, y_b, label_a, label_b, fmt_fn=None, annotate=True):
    """Draw two lines on ax, with optional value annotations."""
    ax.plot(x, y_a, color=COLOR_30, linewidth=2, marker="o", markersize=6, label=label_a, zorder=3)
    ax.plot(x, y_b, color=COLOR_420, linewidth=2, marker="s", markersize=6,
            linestyle="--", label=label_b, zorder=3)
    if annotate and fmt_fn:
        for xi, (ya, yb) in enumerate(zip(y_a, y_b)):
            ax.annotate(fmt_fn(ya), (xi, ya), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=COLOR_30)
            ax.annotate(fmt_fn(yb), (xi, yb), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8, color=COLOR_420)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


# ── 图1：平均年化收益 ────────────────────────────────────────────────────────
def chart1_return(main):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = PERIOD_ORDER
    dual_line(ax, x,
              main["mean_return_3.0"].values * 100,
              main["mean_return_420_static"].values * 100,
              "智能投顾 3.0", "420_static",
              fmt_fn=lambda v: f"{v:.1f}%")
    ax.set_title("图1  平均年化收益对比", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("年化收益（%）", fontsize=10)
    ax.set_xlabel("回测窗口", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/chart1_return.png")
    plt.close(fig)
    print("✓ chart1_return.png")


# ── 图2：平均夏普比率 ────────────────────────────────────────────────────────
def chart2_sharpe(main):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = PERIOD_ORDER
    dual_line(ax, x,
              main["mean_sharpe_3.0"].values,
              main["mean_sharpe_420_static"].values,
              "智能投顾 3.0", "420_static",
              fmt_fn=lambda v: f"{v:.2f}")
    # 在 20y 点标注 420 反超
    ax.annotate("420 反超", xy=(4, main["mean_sharpe_420_static"].iloc[4]),
                xytext=(3.4, main["mean_sharpe_420_static"].iloc[4] + 0.06),
                fontsize=8, color=COLOR_420,
                arrowprops=dict(arrowstyle="->", color=COLOR_420, lw=0.8))
    ax.set_title("图2  平均夏普比率对比", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("夏普比率", fontsize=10)
    ax.set_xlabel("回测窗口", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/chart2_sharpe.png")
    plt.close(fig)
    print("✓ chart2_sharpe.png")


# ── 图3：平均 |Δσ|（越低越好）───────────────────────────────────────────────
def chart3_delta_sigma(main):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = PERIOD_ORDER
    dual_line(ax, x,
              main["mean_abs_delta_sigma_3.0"].values * 100,
              main["mean_abs_delta_sigma_420_static"].values * 100,
              "智能投顾 3.0", "420_static",
              fmt_fn=lambda v: f"{v:.2f}%")
    ax.set_title("图3  平均 |Δσ| 对比（越低越匹配客户）", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("|Δσ|（%）", fontsize=10)
    ax.set_xlabel("回测窗口", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    # 添加"越低越好"注释
    ax.text(0.98, 0.95, "↓ 越低越好", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#555555", style="italic")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/chart3_delta_sigma.png")
    plt.close(fig)
    print("✓ chart3_delta_sigma.png")


# ── 图4：exceed_rate_maxdd（分组柱状图）────────────────────────────────────
def chart4_maxdd(main):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(PERIOD_ORDER))
    w = 0.35
    vals_30 = main["exceed_rate_maxdd_3.0"].values * 100
    vals_420 = main["exceed_rate_maxdd_420_static"].values * 100

    bars_30 = ax.bar(x - w / 2, vals_30, w, color=COLOR_30, label="智能投顾 3.0", zorder=3)
    bars_420 = ax.bar(x + w / 2, vals_420, w, color=COLOR_420, label="420_static",
                      alpha=0.85, zorder=3)

    # 标注数值（只在非零时显示）
    for bar, v in zip(bars_30, vals_30):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=COLOR_30, fontweight="bold")
    for bar, v in zip(bars_420, vals_420):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color="#555555", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(PERIOD_ORDER)
    ax.set_title("图4  MaxDD 红线触碰率对比", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("超回撤红线画像占比（%）", fontsize=10)
    ax.set_xlabel("回测窗口", fontsize=10)
    ax.set_ylim(0, 90)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # 在 1y-10y 区域加浅色背景，标注"均未触线"
    ax.axvspan(-0.5, 3.5, alpha=0.04, color="green", zorder=0)
    ax.text(1.5, 80, "1y–10y 均未触及红线", ha="center", fontsize=8,
            color="#2a7a2a", style="italic")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/chart4_maxdd.png")
    plt.close(fig)
    print("✓ chart4_maxdd.png")


# ── 图5：胜率对比（三折线）──────────────────────────────────────────────────
def chart5_winrate(wr):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(PERIOD_ORDER))

    colors = {"收益胜率": "#1f77b4", "夏普胜率": "#e07b39", "风险匹配胜率": "#2ca02c"}
    cols = {
        "收益胜率": "win_rate_return",
        "夏普胜率": "win_rate_sharpe",
        "风险匹配胜率": "win_rate_risk_match",
    }
    markers = {"收益胜率": "o", "夏普胜率": "^", "风险匹配胜率": "s"}

    for label, col in cols.items():
        vals = wr[col].values * 100
        ax.plot(x, vals, color=colors[label], linewidth=2,
                marker=markers[label], markersize=7, label=label, zorder=3)
        for xi, v in enumerate(vals):
            ax.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7.5, color=colors[label])

    # 50% 基准线
    ax.axhline(50, color="#cccccc", linewidth=1, linestyle="--", zorder=1)
    ax.text(4.05, 50.5, "50%", fontsize=8, color="#aaaaaa")

    ax.set_xticks(list(x))
    ax.set_xticklabels(PERIOD_ORDER)
    ax.set_ylim(30, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("图5  胜率对比（3.0 优于 420_static 的画像占比）",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("胜率", fontsize=10)
    ax.set_xlabel("回测窗口", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/chart5_winrate.png")
    plt.close(fig)
    print("✓ chart5_winrate.png")


# ── 图6：管理层摘要（scorecard 热力格）──────────────────────────────────────
def chart6_summary(main, wr):
    """
    4 维度 × 5 窗口的胜负热力 scorecard。
    绿=3.0 胜，红=420 胜，格内标注数值和箭头符号。
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    dimensions = ["收益", "夏普", "风险匹配", "MaxDD 触线率"]
    windows = PERIOD_ORDER

    # 每个格子：(3.0值, 420值, 单位, 3.0胜=True)
    def cell_data():
        rows = []
        for w in windows:
            m = main[main["window"] == w].iloc[0]
            wr_row = wr[wr["window"] == w].iloc[0]
            rows.append({
                "window": w,
                "收益": (m["mean_return_3.0"] * 100, m["mean_return_420_static"] * 100, "%", True),
                "夏普": (m["mean_sharpe_3.0"], m["mean_sharpe_420_static"], "", True),
                "风险匹配": (m["mean_abs_delta_sigma_3.0"] * 100,
                              m["mean_abs_delta_sigma_420_static"] * 100, "%", False),  # 越小越好
                "MaxDD 触线率": (m["exceed_rate_maxdd_3.0"] * 100,
                                 m["exceed_rate_maxdd_420_static"] * 100, "%", False),  # 越小越好
            })
        return rows

    data = cell_data()

    ax.set_xlim(0, len(windows))
    ax.set_ylim(0, len(dimensions))
    ax.set_xticks([i + 0.5 for i in range(len(windows))])
    ax.set_xticklabels(windows, fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(len(dimensions))])
    ax.set_yticklabels(reversed(dimensions), fontsize=10)
    ax.tick_params(length=0)

    for xi, row in enumerate(data):
        for yi, dim in enumerate(dimensions):
            v30, v420, unit, higher_is_better = row[dim]
            win = (v30 > v420) if higher_is_better else (v30 < v420)
            tie = abs(v30 - v420) < 1e-6

            # Cell color
            if tie:
                fc = "#f5f5f5"
                symbol = "="
            elif win:
                fc = "#d4edda"  # 浅绿
                symbol = "▲"
            else:
                fc = "#f8d7da"  # 浅红
                symbol = "▼"

            rect = mpatches.FancyBboxPatch(
                (xi + 0.05, len(dimensions) - 1 - yi + 0.05),
                0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=fc, edgecolor="#cccccc", linewidth=0.8,
                zorder=2,
            )
            ax.add_patch(rect)

            # 格内文字：3.0 值 / 420 值
            sym_color = "#1a7a1a" if win else ("#cc0000" if not tie else "#555555")
            fmt = ".1f" if unit == "%" else ".2f"
            ax.text(xi + 0.5, len(dimensions) - 1 - yi + 0.62,
                    f"3.0: {v30:{fmt}}{unit}",
                    ha="center", va="center", fontsize=7.5, color="#1f77b4")
            ax.text(xi + 0.5, len(dimensions) - 1 - yi + 0.38,
                    f"420: {v420:{fmt}}{unit}",
                    ha="center", va="center", fontsize=7.5, color="#666666")
            ax.text(xi + 0.82, len(dimensions) - 1 - yi + 0.78,
                    symbol, ha="center", va="center", fontsize=9,
                    color=sym_color, fontweight="bold")

    ax.set_title("图6  管理层摘要  ▲=3.0 胜  ▼=420 胜", fontsize=13,
                 fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#d4edda", edgecolor="#aaaaaa", label="3.0 领先"),
        mpatches.Patch(facecolor="#f8d7da", edgecolor="#aaaaaa", label="420 领先"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=False,
              fontsize=8, bbox_to_anchor=(1.0, -0.08), ncol=2)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/chart6_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ chart6_summary.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main, wr = load_data()
    chart1_return(main)
    chart2_sharpe(main)
    chart3_delta_sigma(main)
    chart4_maxdd(main)
    chart5_winrate(wr)
    chart6_summary(main, wr)
    print(f"\nAll charts saved to {OUT_DIR}/")
