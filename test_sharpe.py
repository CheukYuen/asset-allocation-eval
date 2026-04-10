"""Minimal verification for dynamic Sharpe ratio implementation."""

import numpy as np
import pandas as pd
from src.calc import _sharpe_ratio_dynamic, _align_rf
from src.load import load_rf_series


def test_dynamic_vs_static():
    """Dynamic rf differs from fixed 2% rf when CGB_1Y != 2%."""
    np.random.seed(42)
    monthly = np.random.normal(0.005, 0.03, 60)
    # Fixed rf: 2% annual → ~0.1655% monthly
    rf_fixed = (1 + 0.02) ** (1 / 12) - 1
    sharpe_fixed = (np.mean(monthly - rf_fixed) / np.std(monthly - rf_fixed, ddof=1)) * np.sqrt(12)

    # Dynamic rf: 3% annual for all months
    rf_monthly = np.full(60, (1 + 0.03) ** (1 / 12) - 1)
    sharpe_dynamic = _sharpe_ratio_dynamic(monthly, rf_monthly)

    assert not np.isclose(sharpe_fixed, sharpe_dynamic), \
        "Dynamic Sharpe should differ from static 2% when rf=3%"
    print(f"  static  (rf=2%): {sharpe_fixed:.4f}")
    print(f"  dynamic (rf=3%): {sharpe_dynamic:.4f}")
    print("  PASS: dynamic rf produces different result than fixed 2%")


def test_length_mismatch_raises():
    monthly = np.array([0.01, 0.02, 0.03])
    rf_monthly = np.array([0.001, 0.001])
    try:
        _sharpe_ratio_dynamic(monthly, rf_monthly)
        print("  FAIL: should have raised ValueError")
    except ValueError as e:
        print(f"  PASS: raised ValueError as expected: {e}")


def test_load_rf_series():
    rf = load_rf_series()
    assert isinstance(rf, pd.Series), "Should return a pd.Series"
    assert rf.index.dtype == "datetime64[ns]", "Index should be datetime"
    assert not rf.isna().any(), "No NaNs after ffill (CGB_1Y is complete)"
    assert (rf > 0).all(), "All CGB_1Y rates should be positive"
    print(f"  rf_series loaded: {len(rf)} months, {rf.index[0].date()} ~ {rf.index[-1].date()}")
    print(f"  range: {rf.min():.2f}% ~ {rf.max():.2f}%")
    print("  PASS: load_rf_series()")


def test_align_rf_matches_dates():
    """_align_rf returns correct monthly rates for given dates."""
    # Build synthetic rf_series: 1.2% annual for all months in 2020
    dates_monthly = pd.date_range("2020-01-31", periods=12, freq="ME")
    rf_series = pd.Series(1.2, index=dates_monthly)  # 1.2% annual

    dates = dates_monthly.values
    rf_monthly = _align_rf(dates, rf_series)

    expected = (1 + 1.2 / 100) ** (1 / 12) - 1
    assert np.allclose(rf_monthly, expected), f"Expected {expected:.6f}, got {rf_monthly}"
    print(f"  PASS: _align_rf correctly converts 1.2% annual → {expected:.6f} monthly")


def test_align_rf_missing_leading_raises():
    """_align_rf should raise if CGB_1Y unavailable before first portfolio date."""
    # rf only starts 2010, but portfolio dates start 2005
    rf_series = pd.Series(2.0, index=pd.date_range("2010-01-31", periods=12, freq="ME"))
    early_dates = pd.date_range("2005-01-31", periods=6, freq="ME").values
    try:
        _align_rf(early_dates, rf_series)
        print("  FAIL: should have raised ValueError for missing leading rf")
    except ValueError as e:
        print(f"  PASS: raised ValueError for missing leading rf: {e}")


if __name__ == "__main__":
    print("=== Dynamic Sharpe Verification ===\n")
    print("[1] dynamic vs static rf:")
    test_dynamic_vs_static()

    print("\n[2] length mismatch guard:")
    test_length_mismatch_raises()

    print("\n[3] load_rf_series():")
    test_load_rf_series()

    print("\n[4] _align_rf date alignment:")
    test_align_rf_matches_dates()

    print("\n[5] _align_rf missing leading rf raises:")
    test_align_rf_missing_leading_raises()

    print("\nAll checks done.")
