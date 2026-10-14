"""تست‌های واحد آزمون روندپیروی و مومنتوم — دیتای مصنوعی با پاسخ قطعی."""
import math

import numpy as np
import pandas as pd
import pytest

from backtest import trend_test as tt
from backtest.trend_report import write_outputs


def panel_rw(n=1500, k=12, seed=0, vol=0.03, drift=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    cols = {}
    for i in range(k):
        mu = 0.0 if drift is None else drift[i]
        cols[f"S{i}/USDT"] = 100 * np.exp(np.cumsum(rng.normal(mu, vol, n)))
    return pd.DataFrame(cols, index=idx)


# ------------------------------------------------------------------ بدون نگاه‌به‌آینده
@pytest.mark.parametrize("kind", ["tsmom_ls", "tsmom_lo", "csmom"])
def test_weights_do_not_use_future_prices(kind):
    P = panel_rw(n=600)
    full = tt.target_weights(P, kind, 30, 7, 200)
    cut = tt.target_weights(P.iloc[:400], kind, 30, 7, 200)
    pd.testing.assert_frame_equal(full.iloc[:400], cut)        # وزن‌های ۴۰۰ روز اول با قطع داده‌ی آینده عوض نمی‌شود


def test_strategy_return_uses_previous_day_weights():
    P = panel_rw(n=500)
    w = tt.target_weights(P, "tsmom_ls", 30, 7, 100)
    s = tt.daily_series(P, w)
    t = 300
    expect = float((w.iloc[t - 1] * P.pct_change().iloc[t]).sum())
    assert s["gross"].iloc[t] == pytest.approx(expect)


# ------------------------------------------------------------------ وزن‌ها
def test_csmom_is_market_neutral_with_unit_gross():
    P = panel_rw(n=600)
    w = tt.target_weights(P, "csmom", 30, 7, 200).iloc[250:]
    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-12)
    assert np.allclose(w.abs().sum(axis=1), 1.0, atol=1e-12)


def test_long_only_has_no_shorts_and_long_short_gross_le_one():
    P = panel_rw(n=600)
    lo = tt.target_weights(P, "tsmom_lo", 30, 7, 200)
    ls = tt.target_weights(P, "tsmom_ls", 30, 7, 200)
    assert (lo >= -1e-12).all().all()
    assert (ls.abs().sum(axis=1) <= 1.0 + 1e-12).all()


def test_weights_constant_between_rebalances():
    P = panel_rw(n=600)
    w = tt.target_weights(P, "tsmom_ls", 30, 7, 200)
    block = w.iloc[207:214]                                    # از یک بازتنظیم تا قبل از بعدی
    assert (block.nunique() <= 1).all()
    assert (w.iloc[:200] == 0).all().all()                    # قبل از شروع ارزیابی پوزیشنی نیست


# ------------------------------------------------------------------ هزینه
def test_turnover_and_costs_exact():
    idx = pd.date_range("2022-01-01", periods=6, freq="D")
    close = pd.DataFrame({"A": [100, 101, 102, 103, 104, 105.0], "B": [100, 99, 98, 97, 96, 95.0]}, index=idx)
    w = pd.DataFrame({"A": [0.0, 0.5, 0.5, -0.5, -0.5, -0.5], "B": [0.0, 0.5, 0.5, 0.5, 0.5, 0.5]}, index=idx)
    s = tt.daily_series(close, w)
    # روز ۲: تغییر وزن از ۰ به (۰.۵,۰.۵) → گردش ۱ ؛ روز ۴: A از +۰.۵ به −۰.۵ → گردش ۱
    assert np.allclose(s["turnover"], [0, 0, 1.0, 0, 1.0, 0])
    assert np.allclose(s["short"], [0, 0, 0, 0, 0.5, 0.5])
    n = tt.net_returns(s, fee_pct=0.1, borrow_daily_pct=0.05)
    assert n.iloc[2] == pytest.approx(s["gross"].iloc[2] - 1.0 * 0.001)
    assert n.iloc[4] == pytest.approx(s["gross"].iloc[4] - 1.0 * 0.001 - 0.5 * 0.0005)


# ------------------------------------------------------------------ آمار
def test_newey_west_calibration_and_autocorrelation():
    rng = np.random.default_rng(1)
    fp = 0
    for _ in range(300):
        fp += tt.nw_tstat(rng.normal(0, 1, 500))[1] < 0.05
    assert fp / 300 < 0.10
    ar = np.zeros(3000)
    e = rng.normal(0, 1, 3000)
    for i in range(1, 3000):
        ar[i] = 0.5 * ar[i - 1] + e[i]
    naive_t = ar.mean() / (ar.std(ddof=1) / math.sqrt(len(ar)))
    assert abs(tt.nw_tstat(ar)[0]) < abs(naive_t) + 1e-9      # خودهمبستگی: خطای استاندارد NW بزرگ‌تر از ساده


def test_alpha_beta_recovers_parameters():
    rng = np.random.default_rng(2)
    b = rng.normal(0.0005, 0.03, 3000)
    y = 0.0008 + 0.7 * b + rng.normal(0, 0.01, 3000)
    r = tt.alpha_beta(y, b)
    assert r["beta"] == pytest.approx(0.7, abs=0.03)
    assert r["alpha_ann"] == pytest.approx(0.0008 * 365, abs=0.25)      # خطای استاندارد α سالانه ≈ ۰.۰۷


def test_equity_stats_known_values():
    r = pd.Series([0.01] * 365)
    s = tt.equity_stats(r)
    assert s["mdd"] == pytest.approx(0.0, abs=1e-12) and s["total"] == pytest.approx(1.01 ** 365 - 1, rel=1e-9)
    down = tt.equity_stats(pd.Series([0.1] * 50 + [-0.5] + [0.0] * 50))
    assert down["mdd"] == pytest.approx(-0.5, abs=1e-9)


# ------------------------------------------------------------------ کل پایپ‌لاین
def test_real_trend_is_detected_and_pure_noise_is_not():
    # ۱۲ نماد با روند پایدار مثبت یا منفی و نوسان معقول → روندپیروی لانگ‌شورت باید برتری نشان دهد
    rng = np.random.default_rng(5)
    drift = [0.004 if i % 2 == 0 else -0.004 for i in range(12)]
    res = tt.run(panel_rw(n=1500, drift=drift, vol=0.02, seed=5), 5, 365)
    best = [r for r in res["rows"] if r["name"] == "tsmom_ls_90_7"][0]
    assert best["t_gross"] > 5 and best["verdict"] in ("edge_net_pos", "edge_below_cost")
    sig = 0
    tot = 0
    for seed in range(6):
        res = tt.run(panel_rw(n=1500, seed=100 + seed), 5, 365)
        sig += sum(r["p_bonf"] < 0.05 for r in res["rows"])
        tot += len(res["rows"])
    assert sig / tot < 0.12                                    # روی راه‌رفتن تصادفی: نرخ مثبت کاذب کوچک


def test_write_outputs_creates_all_files(tmp_path):
    P = panel_rw(n=1500, seed=9)
    res = tt.run(P, 5, 365)
    write_outputs(P, res, str(tmp_path), 0.10, 0.0)
    for f in ["summary.md", "dashboard.html", "variants.csv", "daily_returns.csv.gz", "meta.json"]:
        assert (tmp_path / f).exists(), f
    v = pd.read_csv(tmp_path / "variants.csv")
    assert len(v) == len(tt.VARIANTS)
    assert "tsmom_ls_90_7" in (tmp_path / "summary.md").read_text(encoding="utf-8")
