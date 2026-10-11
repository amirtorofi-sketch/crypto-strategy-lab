"""تست‌های واحد آزمون سیگنال (Event Study) — دیتای مصنوعی با پاسخ قطعی."""
import math

import numpy as np
import pandas as pd
import pytest

from backtest import signal_report as sr
from backtest.signal_test import HORIZONS, SESSIONS, collect_events, prepare_symbol
from strategies.base import Signal, Strategy


def make15(n=2000, start="2026-01-01 00:00", fn=None):
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    close = np.array([100.0 + 0.01 * k for k in range(n)]) if fn is None else fn(n)
    df = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0}, index=idx)
    return df


class FireAt(Strategy):
    """در کندل‌های مشخص (با زمان باز شدن) سیگنال می‌دهد."""
    category = "test"
    min_bars = 5
    timeframe = "15m"

    def __init__(self, fire, side="long", name="test_fire"):
        self.fire = set(fire)
        self._side = side
        self.name = name

    def generate_signal(self, df):
        if df.index[-1] in self.fire:
            c = float(df["close"].iloc[-1])
            if self._side == "long":
                return Signal(side="long", entry=c, stop_loss=c * 0.99, take_profit=c * 1.02)
            return Signal(side="short", entry=c, stop_loss=c * 1.01, take_profit=c * 0.98)
        return None


# ------------------------------------------------------------------ بازده آینده
def test_forward_returns_alignment_and_edges():
    df = make15()
    a = prepare_symbol(df)
    c = df["close"].to_numpy()
    for lab, h in HORIZONS.items():
        r = a.ret[lab]
        assert r[10] == pytest.approx((c[10 + h] / c[10] - 1) * 1e4, rel=1e-9)
        assert np.isnan(r[-1]) and np.isnan(r[-h])         # انتهای داده: افق کامل نیست
        assert not np.isnan(r[-h - 1])


def test_gap_makes_returns_nan_instead_of_wrong():
    df = make15()
    df = df.drop(df.index[100])                       # یک کندل گم‌شده
    a = prepare_symbol(df)
    # کندل‌هایی که افق ۴ کندلی‌شان از روی حفره رد می‌شود NaN هستند
    assert np.isnan(a.ret["1h"][99]) and np.isnan(a.ret["1h"][97])
    assert not np.isnan(a.ret["1h"][90])


def test_baseline_is_session_mean():
    df = make15(n=96 * 20)
    a = prepare_symbol(df, baseline="year")
    r = a.ret["4h"]
    t_close = df.index.as_unit("ns").asi8 + 900 * 10**9
    hours = (t_close // (3600 * 10**9)) % 24
    asia = (hours >= 0) & (hours < 7) & ~np.isnan(r)
    k_asia = int(np.where(asia)[0][5])
    assert a.base["4h"][k_asia] == pytest.approx(r[asia].mean())


def _regime_series(n):
    """ژانویه: صعود تند ، فوریه: ریزش تند (رژیم‌های بازار)؛ هر کندل ±۱۰ bps."""
    idx = pd.date_range("2026-01-01 00:00", periods=n, freq="15min", tz="UTC")
    step = np.where(idx.month == 1, 0.001, -0.001)
    close = 100 * np.exp(np.cumsum(step))
    return pd.DataFrame({"open": close, "high": close * 1.0005, "low": close * 0.9995, "close": close, "volume": 1.0},
                        index=idx)


def test_monthly_baseline_follows_the_market_regime():
    df = _regime_series(96 * 59)                        # ژانویه + فوریه (۵۹ روز)
    am = prepare_symbol(df, baseline="month")
    ay = prepare_symbol(df, baseline="year")
    k_jan, k_feb = 96 * 10, 96 * 45
    assert am.base["4h"][k_jan] == pytest.approx(16 * 10.0, abs=15)      # حدود +۱۶۰ bps
    assert am.base["4h"][k_feb] == pytest.approx(-16 * 10.0, abs=15)     # حدود −۱۶۰ bps
    assert abs(ay.base["4h"][k_jan]) < 80                                # baseline یک‌ساله هر دو رژیم را میانگین می‌گیرد


def test_regime_following_is_not_credited_as_skill_with_monthly_baseline():
    df = _regime_series(96 * 59)
    am = prepare_symbol(df, baseline="month")
    ay = prepare_symbol(df, baseline="year")
    rng = np.random.default_rng(0)
    k = np.concatenate([rng.integers(96 * 5, 96 * 25, 200), rng.integers(96 * 35, 96 * 55, 200)])
    side = np.where(df.index[k].month == 1, 1.0, -1.0)    # «سیگنال» فقط جهت رژیم را دنبال می‌کند
    exc_m = np.mean(side * (am.ret["4h"][k] - am.base["4h"][k]))
    exc_y = np.mean(side * (ay.ret["4h"][k] - ay.base["4h"][k]))
    raw = np.mean(side * am.ret["4h"][k])
    assert raw > 100 and exc_y > 100 and abs(exc_m) < 10   # baseline سالانه مهارت کاذب می‌سازد ، ماهانه نه


def test_small_cells_fall_back_without_nan():
    df = make15(n=96 * 3)
    a = prepare_symbol(df, baseline="month")
    ok = ~np.isnan(a.ret["1h"])
    assert not np.isnan(a.base["1h"][ok]).any()


def test_invalid_baseline_rejected():
    with pytest.raises(ValueError):
        prepare_symbol(make15(n=500), baseline="decade")


def test_mfe_mae_for_long_ramp():
    df = make15()
    a = prepare_symbol(df)
    # بالای ۴ ساعت بعد: max(high[k+1..k+16]) / close[k]
    k = 50
    exp_up = (df["high"].iloc[k + 1:k + 17].max() / df["close"].iloc[k] - 1) * 100
    exp_dn = (df["low"].iloc[k + 1:k + 17].min() / df["close"].iloc[k] - 1) * 100
    assert a.mfe["4h"][k] == pytest.approx(exp_up) and a.mae["4h"][k] == pytest.approx(exp_dn)


# ------------------------------------------------------------------ رویدادها
def test_consecutive_same_side_signals_count_once():
    df = make15()
    a = prepare_symbol(df)
    fire = [df.index[200], df.index[201], df.index[202], df.index[300]]
    ev = collect_events(FireAt(fire), df, a, "T/USDT", window_size=110, min_reward_pct=0.0)
    assert len(ev) == 2                                # {200,201,202} -> یکی ، {300} -> یکی


def test_event_anchored_at_candle_close_with_correct_session():
    df = make15()
    a = prepare_symbol(df)
    k = 11 * 4 + 3 + 96 * 2                            # کندل ۱۱:۴۵ (بسته‌شدنش ۱۲:۰۰ => London-NY Overlap)
    assert df.index[k].hour == 11 and df.index[k].minute == 45
    ev = collect_events(FireAt([df.index[k]]), df, a, "T/USDT", 110, 0.0)
    assert len(ev) == 1
    e = ev[0]
    assert SESSIONS[e["session"]] == "London-NY Overlap"
    assert e["t"] == df.index[k].value + 900 * 10**9
    assert e["r_1h"] == pytest.approx(a.ret["1h"][k])   # مبدأ = close خود کندل سیگنال


def test_short_side_flips_mfe_mae():
    df = make15()
    a = prepare_symbol(df)
    k = 300
    lo = collect_events(FireAt([df.index[k]], "long"), df, a, "T/USDT", 110, 0.0)[0]
    sh = collect_events(FireAt([df.index[k]], "short"), df, a, "T/USDT", 110, 0.0)[0]
    assert sh["mfe_4h"] == pytest.approx(-lo["mae_4h"]) and sh["mae_4h"] == pytest.approx(-lo["mfe_4h"])


# ------------------------------------------------------------------ آمار
def test_cluster_se_is_larger_than_naive_when_days_share_shocks():
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(50), 40)                # ۵۰ روز ، هر روز ۴۰ رویداد
    shock = rng.normal(0, 10, 50)[days]                # شوک مشترک روزانه (بازار هم‌بسته)
    x = shock + rng.normal(0, 5, len(days))
    mean, se, t, p, G = sr.cluster_mean_test(x, days)
    naive_se = x.std(ddof=1) / math.sqrt(len(x))
    assert G == 50 and se > 2 * naive_se


def test_null_calibration_and_power():
    rng = np.random.default_rng(2)
    days = np.repeat(np.arange(100), 20)
    false_pos = 0
    for _ in range(200):
        x = rng.normal(0, 30, len(days)) + rng.normal(0, 10, 100)[days]
        false_pos += sr.cluster_mean_test(x, days)[3] < 0.05
    assert false_pos / 200 < 0.12                       # سطح ۵٪ با تلورانس نمونه‌گیری
    x = rng.normal(5, 30, len(days)) + rng.normal(0, 10, 100)[days]   # برتری واقعی +۵ bps
    assert sr.cluster_mean_test(x, days)[2] > 3


def test_verdict_rules():
    assert sr._verdict(500, 6.0, 0.001, "pos", 3.0) == "tradable"
    assert sr._verdict(500, 6.0, 0.001, "pos", -2.0) == "below_cost"
    assert sr._verdict(500, -6.0, 0.001, "neg", -14.0) == "inverse"
    assert sr._verdict(500, 6.0, 0.001, "mixed", 3.0) == "hypothesis"
    assert sr._verdict(500, 6.0, 0.3, "pos", 3.0) == "hypothesis"
    assert sr._verdict(10, 50.0, 0.0, "pos", 40.0) == "low_n"


def _events(n=600, drift=0.0, seed=3):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-01-01", tz="UTC").value
    t = t0 + (np.sort(rng.integers(0, 120 * 86400, n)) * 10**9)
    ev = pd.DataFrame({"strategy": rng.choice(["s1", "s2"], n), "symbol": rng.choice(["A/USDT", "B/USDT"], n),
                       "tf": "1h", "side": rng.choice([1, -1], n), "t": t, "session": rng.integers(0, 5, n)})
    for lab in HORIZONS:
        ev["r_" + lab] = rng.normal(0, 50, n) + ev["side"] * drift
        ev["b_" + lab] = 0.0
    for w in ("4h", "24h"):
        ev["mfe_" + w] = rng.uniform(0, 2, n)
        ev["mae_" + w] = -rng.uniform(0, 2, n)
    return ev


def test_injected_drift_is_detected_and_null_is_not():
    for drift, expect_sig in ((0.0, False), (25.0, True)):
        ev = sr.prepare_events(_events(n=4000, drift=drift))
        mid = int(ev["t"].quantile(0.5))
        rows = sr.build_table(ev, ["strategy"], 0.04, 0.35, mid)
        r = [x for x in rows if x["key"][1] == "4h"][0]
        assert (r["q"] is not None and r["q"] < 0.05) == expect_sig


def test_write_outputs_creates_all_files(tmp_path):
    ev = _events(n=1500, drift=10.0)
    params = {"min_reward_pct": 0.15, "windows": {"1h": 110}, "strategies": ["s1", "s2"], "skipped": ["x_5m"],
              "symbols_requested": 2, "symbols_with_data": 2, "symbols_missing": [], "fee_pct_a": 0.04, "fee_pct_b": 0.35}
    sr.write_outputs(ev, {"AUSDT": {}}, params, str(tmp_path))
    for f in ["summary.md", "dashboard.html", "by_strategy.csv", "by_strategy_session.csv", "by_strategy_symbol.csv",
              "by_strategy_side.csv", "meta.json"]:
        assert (tmp_path / f).exists(), f
    df = pd.read_csv(tmp_path / "by_strategy.csv")
    assert set(df["horizon"]) == set(HORIZONS) and set(df["strategy"]) == {"s1", "s2"}
    assert "s1" in (tmp_path / "summary.md").read_text(encoding="utf-8")
