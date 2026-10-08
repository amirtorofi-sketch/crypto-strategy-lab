"""تست‌های واحد بک‌تستر ترکیب‌ها — سناریوهای قطعی با دیتای مصنوعی."""
import math

import numpy as np
import pandas as pd
import pytest

from backtest import combo_stats as cs
from backtest.combo_backtest import resample_tf, simulate
from backtest.combo_report import write_outputs
from strategies.base import Signal, Strategy

SIGNAL_BAR = 59   # simulate از ایندکس max(min_bars, 60)-1 شروع می‌کند؛ این ایندکس = ساعت ۱۱:۰۰ UTC در دیتای زیر


class FixedSignal(Strategy):
    name = "test_fixed"
    category = "test"
    min_bars = 5
    timeframe = "1h"

    def __init__(self, target_ts, signal):
        self.target_ts = target_ts
        self._sig = signal

    def generate_signal(self, df):
        return self._sig if df.index[-1] == self.target_ts else None


class EveryBar(Strategy):
    name = "test_every"
    category = "test"
    min_bars = 5
    timeframe = "1h"

    def generate_signal(self, df):
        return Signal(side="long", entry=100.0, stop_loss=99.0, take_profit=102.0)


def make_df(n=100):
    idx = pd.date_range("2026-01-01 00:00", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0}, index=idx)
    return df


def run(strategy, df, allowed=("London-NY Overlap",), min_reward=0.0, fee=0.04, slip=0.0):
    return simulate(strategy, df, "TEST/USDT", set(allowed), set(allowed), window_size=110,
                    min_reward_pct=min_reward, trade_value=10.0, fee_pct=fee, slippage_pct=slip)


def long_signal(tp=102.0):
    return Signal(side="long", entry=100.0, stop_loss=99.0, take_profit=tp)


# ------------------------------------------------------------------ شبیه‌ساز
def test_session_uses_candle_close_time():
    df = make_df()
    strat = FixedSignal(df.index[SIGNAL_BAR], long_signal())
    df.iloc[SIGNAL_BAR + 2, df.columns.get_loc("high")] = 102.5
    # کندل ۱۱:۰۰ بسته‌شدنش ۱۲:۰۰ است => London-NY Overlap (نه London)
    assert len(run(strat, df, allowed=("London-NY Overlap",))) == 1
    assert run(strat, df, allowed=("London",)) == []


def test_exit_is_not_checked_on_signal_candle():
    df = make_df()
    df.iloc[SIGNAL_BAR, df.columns.get_loc("low")] = 98.5            # کندل سیگنال به SL رسیده بود
    df.iloc[SIGNAL_BAR + 2, df.columns.get_loc("high")] = 102.5      # TP دو کندل بعد
    t = run(FixedSignal(df.index[SIGNAL_BAR], long_signal()), df)
    assert len(t) == 1
    assert t[0]["exit_reason"] == "tp" and t[0]["win"] is True
    assert pd.Timestamp(t[0]["exit_time"]) > pd.Timestamp(t[0]["entry_time"])


def test_sl_wins_when_both_touched_in_same_candle():
    df = make_df()
    df.iloc[SIGNAL_BAR + 1, df.columns.get_loc("high")] = 102.5
    df.iloc[SIGNAL_BAR + 1, df.columns.get_loc("low")] = 98.5
    t = run(FixedSignal(df.index[SIGNAL_BAR], long_signal()), df)
    assert t[0]["exit_reason"] == "sl" and t[0]["win"] is False


def test_fee_math_and_fixed_size():
    df = make_df()
    df.iloc[SIGNAL_BAR + 1, df.columns.get_loc("high")] = 102.5
    t = run(FixedSignal(df.index[SIGNAL_BAR], long_signal()), df)[0]
    assert t["pnl_pct"] == pytest.approx(2.0 - 0.08)
    assert t["pnl_usdt"] == pytest.approx(10 * (2.0 - 0.08) / 100, abs=1e-4)


def test_win_is_decided_by_pnl_sign_not_tp_hit():
    df = make_df()
    df.iloc[SIGNAL_BAR + 1, df.columns.get_loc("high")] = 100.2
    t = run(FixedSignal(df.index[SIGNAL_BAR], long_signal(tp=100.05)), df, min_reward=0.0)[0]
    assert t["exit_reason"] == "tp"           # TP خورد ...
    assert t["win"] is False                  # ... ولی بعد از کارمزد ضرر است
    assert t["pnl_usdt"] < 0


def test_min_reward_pct_rejects_tiny_targets():
    df = make_df()
    df.iloc[SIGNAL_BAR + 1, df.columns.get_loc("high")] = 100.2
    assert run(FixedSignal(df.index[SIGNAL_BAR], long_signal(tp=100.05)), df, min_reward=0.15) == []


def test_single_open_position_per_strategy_and_symbol():
    df = make_df(100)
    df.iloc[80, df.columns.get_loc("high")] = 102.5      # فقط یک TP در کل بازه
    t = run(EveryBar(), df, allowed=("Asia", "London", "London-NY Overlap", "New York", "Off-hours"))
    # بین ورود و خروج سیگنال تازه‌ای باز نمی‌شود؛ ورودها پشت‌سرهم و غیرهم‌پوشان‌اند
    starts = [pd.Timestamp(x["entry_time"]) for x in t]
    ends = [pd.Timestamp(x["exit_time"]) for x in t]
    assert all(starts[i + 1] >= ends[i] for i in range(len(t) - 1))


def test_resample_keeps_only_complete_hours():
    idx = pd.date_range("2026-01-01 00:00", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=idx)
    df = df.drop(df.index[5])                      # یک کندل ۱۵ دقیقه‌ای گم‌شده در ساعت دوم
    h = resample_tf(df, "1h")
    assert list(h.index) == [pd.Timestamp("2026-01-01 00:00", tz="UTC")]


# ------------------------------------------------------------------ آمار
def test_wilson_interval():
    lo, hi = cs.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert cs.wilson(0, 0) == (0.0, 0.0)


def test_binomial_two_sided_matches_known_value():
    # مقدار مرجع scipy.stats.binomtest(60, 100, 0.5).pvalue = 0.05688793...
    assert cs.binom_two_sided_p(60, 100, 0.5) == pytest.approx(0.0568879, abs=1e-6)
    assert cs.binom_two_sided_p(5, 10, 0.5) == pytest.approx(1.0)
    assert cs.binom_two_sided_p(1, 0, 0.5) is None


def test_max_drawdown_and_streak():
    assert cs.max_drawdown(np.array([1.0, -2.0, -1.0, 4.0, -1.0])) == pytest.approx(3.0)
    assert cs.longest_losing_streak(np.array([True, False, False, True, False])) == 2


def test_bh_adjust_is_monotone_and_bounded():
    rows = [{"p": 0.001}, {"p": 0.02}, {"p": 0.04}, {"p": 0.5}, {"p": None}]
    cs.bh_adjust(rows)
    qs = [r["q"] for r in rows[:4]]
    assert qs == sorted(qs) and all(0 <= q <= 1 for q in qs) and rows[4]["q"] is None


def test_verdict_requires_sample_stability_and_significance():
    base = {"n": 150, "q": 0.01, "stable": "pos", "exp": 0.05}
    assert cs.verdict(base) == "reliable_pos"
    assert cs.verdict({**base, "stable": "mixed"}) == "hypothesis"
    assert cs.verdict({**base, "q": 0.2}) == "hypothesis"
    assert cs.verdict({**base, "n": 20}) == "low_n"
    assert cs.verdict({**base, "stable": "neg", "exp": -0.05}) == "reliable_neg"


# ------------------------------------------------------------------ خروجی‌ها
def test_write_outputs_creates_all_files(tmp_path):
    rng = np.random.default_rng(0)
    n = 80
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for i in range(n):
        pnl = float(rng.choice([-0.1, 0.2], p=[0.6, 0.4]))
        rows.append({"strategy": "s1" if i % 2 else "s2", "category": "x", "symbol": "BTC/USDT" if i % 3 else "ETH/USDT",
                     "timeframe": "1h", "session": "Asia", "selected": True, "side": "long",
                     "entry_time": (t0 + pd.Timedelta(hours=i)).isoformat(),
                     "exit_time": (t0 + pd.Timedelta(hours=i + 2)).isoformat(),
                     "entry_price": 100.0, "stop_loss": 99.0, "take_profit": 102.0, "exit_price": 101.0,
                     "exit_reason": "tp", "pnl_pct": pnl * 10, "pnl_usdt": pnl, "win": pnl > 0,
                     "duration_minutes": 120.0, "reason": ""})
    params = {"fee_pct": 0.04, "slippage_pct": 0.0, "trade_value": 10.0, "min_reward_pct": 0.15, "windows": {"1h": 110},
              "all_sessions": False, "symbols_requested": 2, "symbols_with_data": 2, "symbols_missing": [],
              "combos": {"s1": ["Asia"], "s2": ["Asia"]}}
    write_outputs(pd.DataFrame(rows), {"BTCUSDT": {}}, params, str(tmp_path))
    for f in ["summary.md", "dashboard.html", "by_combo.csv", "by_strategy.csv", "by_combo_symbol.csv",
              "by_symbol.csv", "trades.csv.gz", "meta.json"]:
        assert (tmp_path / f).exists(), f
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "s1" in md and "s2" in md
    df = pd.read_csv(tmp_path / "by_combo.csv")
    assert set(df["strategy"]) == {"s1", "s2"} and math.isclose(df["n"].sum(), n)
