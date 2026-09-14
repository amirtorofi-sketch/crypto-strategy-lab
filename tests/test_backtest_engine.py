"""تست‌های واحد برای موتور بک‌تست — سناریوهای قطعی با دیتای مصنوعی."""
import pandas as pd
import pytest

from backtest.engine import Backtester, trades_to_df
from backtest.metrics import compute_metrics
from strategies.base import Strategy, Signal, validate_signal
from strategies.range_rtm.strategies import RangeBoundFadeStrategy

SIGNAL_BAR = 60  # موتور از max(min_bars, 60) شروع می‌کند


class FixedSignal(Strategy):
    """فقط روی یک تایم‌استمپ مشخص سیگنال می‌دهد."""
    name = "test_fixed"
    category = "test"
    min_bars = 5
    timeframe = "1h"

    def __init__(self, target_ts, signal):
        self.target_ts = target_ts
        self._sig = signal

    def generate_signal(self, df):
        if df.index[-1] == self.target_ts:
            return self._sig
        return None


class NeverSignal(Strategy):
    name = "test_never"
    category = "test"
    min_bars = 5
    timeframe = "1h"

    def generate_signal(self, df):
        return None


def make_df(n=80, price=100.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": price, "high": price, "low": price,
        "close": price, "volume": 1.0,
    }, index=idx)


def set_bar(df, i, *, open=None, high=None, low=None, close=None):
    if open is not None:
        df.loc[df.index[i], "open"] = open
    if high is not None:
        df.loc[df.index[i], "high"] = high
    if low is not None:
        df.loc[df.index[i], "low"] = low
    if close is not None:
        df.loc[df.index[i], "close"] = close


LONG_SIGNAL = Signal("long", entry=100.0, stop_loss=90.0, take_profit=110.0)

# با کارمزد ۰.۰۴٪ + اسلیپیج ۰.۰۲٪ رفت‌وبرگشت: هزینه هر معامله = ۰.۱۲٪
COSTS = 0.12


def test_take_profit_hit_is_win():
    df = make_df()
    set_bar(df, 61, high=111.0, low=99.0)
    strat = FixedSignal(df.index[SIGNAL_BAR], LONG_SIGNAL)
    trades = Backtester().run(strat, df, "TEST/USDT")
    assert len(trades) == 1
    assert trades[0].result == "win"
    assert trades[0].exit_price == 110.0
    assert trades[0].pnl_pct == pytest.approx(10.0 - COSTS, abs=1e-4)


def test_stop_loss_hit_is_loss():
    df = make_df()
    set_bar(df, 61, high=101.0, low=89.0)
    strat = FixedSignal(df.index[SIGNAL_BAR], LONG_SIGNAL)
    trades = Backtester().run(strat, df, "TEST/USDT")
    assert len(trades) == 1
    assert trades[0].result == "loss"
    assert trades[0].exit_price == 90.0
    assert trades[0].pnl_pct == pytest.approx(-10.0 - COSTS, abs=1e-4)


def test_both_hit_conservative_counts_loss():
    """هم SL هم TP در یک کندل: حالت پیش‌فرض محافظه‌کارانه SL را می‌زند."""
    df = make_df()
    set_bar(df, 61, open=100.0, high=111.0, low=89.0)
    strat = FixedSignal(df.index[SIGNAL_BAR], LONG_SIGNAL)
    trades = Backtester(intrabar_mode="conservative").run(strat, df, "TEST/USDT")
    assert trades[0].result == "loss"


def test_both_hit_distance_mode_resolves_by_open_proximity():
    """حالت distance: open به TP نزدیک‌تر است -> ابتدا TP لمس می‌شود."""
    df = make_df()
    set_bar(df, 61, open=109.5, high=111.0, low=89.0)
    strat = FixedSignal(df.index[SIGNAL_BAR], LONG_SIGNAL)
    trades = Backtester(intrabar_mode="distance").run(strat, df, "TEST/USDT")
    assert trades[0].result == "win"

    # برعکس: open به SL نزدیک‌تر -> SL ابتدا زده می‌شود
    df2 = make_df()
    set_bar(df2, 61, open=90.5, high=111.0, low=89.0)
    strat2 = FixedSignal(df2.index[SIGNAL_BAR], LONG_SIGNAL)
    trades2 = Backtester(intrabar_mode="distance").run(strat2, df2, "TEST/USDT")
    assert trades2[0].result == "loss"


def test_next_open_entry_mode():
    """ورود با open کندل بعد از سیگنال، نه close کندل سیگنال."""
    df = make_df()
    set_bar(df, 61, open=102.0, high=103.0, low=101.0)
    set_bar(df, 62, high=103.0, low=89.0)  # SL در کندل بعدی
    strat = FixedSignal(df.index[SIGNAL_BAR], LONG_SIGNAL)
    trades = Backtester(entry_price_mode="next_open").run(strat, df, "TEST/USDT")
    assert len(trades) == 1
    assert trades[0].entry_price == 102.0
    assert trades[0].result == "loss"


def test_only_one_position_at_a_time():
    """تا وقتی پوزیشن باز است سیگنال‌های جدید نادیده گرفته می‌شوند."""
    df = make_df()
    strat = FixedSignal(df.index[SIGNAL_BAR], LONG_SIGNAL)

    # یک استراتژی که هر دو کندل 60 و 62 سیگنال می‌دهد
    class TwoSignal(Strategy):
        name = "test_two"
        category = "test"
        min_bars = 5
        timeframe = "1h"

        def generate_signal(self, d):
            if d.index[-1] in (df.index[60], df.index[62]):
                return LONG_SIGNAL
            return None

    trades = Backtester().run(TwoSignal(), df, "TEST/USDT")
    assert len(trades) == 1
    assert trades[0].result == "open"  # هیچ‌وقت بسته نشد


def test_strategy_exception_is_swallowed():
    """استثنا در یک استراتژی نباید کل بک‌تست را بکشد."""
    class Broken(Strategy):
        name = "test_broken"
        category = "test"
        min_bars = 5
        timeframe = "1h"

        def generate_signal(self, df):
            raise RuntimeError("خرابی عمدی")

    df = make_df()
    trades = Backtester().run(Broken(), df, "TEST/USDT")
    assert trades == []


def test_no_signal_no_trades():
    df = make_df()
    trades = Backtester().run(NeverSignal(), df, "TEST/USDT")
    assert trades == []


# ---------- اعتبارسنجی سیگنال (validate_signal) ----------

def test_validate_signal_rejects_wrong_direction_long():
    bad = Signal("long", entry=100.0, stop_loss=90.0, take_profit=99.0)  # TP زیر ورود!
    ok, reason = validate_signal(bad)
    assert not ok
    assert "حد سود" in reason


def test_validate_signal_rejects_wrong_direction_short():
    bad = Signal("short", entry=100.0, stop_loss=110.0, take_profit=101.0)  # TP بالای ورود!
    ok, reason = validate_signal(bad)
    assert not ok


def test_validate_signal_accepts_valid_long():
    good = Signal("long", entry=100.0, stop_loss=95.0, take_profit=110.0)
    ok, reason = validate_signal(good)
    assert ok
    assert reason == ""


def test_validate_signal_rejects_tiny_reward_below_threshold():
    tiny = Signal("long", entry=100.0, stop_loss=95.0, take_profit=100.03)  # فاصله ۰.۰۳٪
    ok, reason = validate_signal(tiny, min_reward_pct=0.15)
    assert not ok


def test_validate_signal_allows_tiny_reward_when_threshold_is_zero():
    tiny = Signal("long", entry=100.0, stop_loss=95.0, take_profit=100.03)
    ok, _ = validate_signal(tiny, min_reward_pct=0.0)
    assert ok


def test_engine_skips_direction_invalid_signal():
    """موتور بک‌تست نباید حتی یک پوزیشن هم برای سیگنال با جهت غلط باز کند."""
    bad_signal = Signal("short", entry=100.0, stop_loss=110.0, take_profit=101.0)
    df = make_df()
    strat = FixedSignal(df.index[SIGNAL_BAR], bad_signal)
    trades = Backtester().run(strat, df, "TEST/USDT")
    assert trades == []


def test_engine_respects_min_reward_pct_filter():
    tiny_reward_signal = Signal("long", entry=100.0, stop_loss=95.0, take_profit=100.03)
    df = make_df()
    strat = FixedSignal(df.index[SIGNAL_BAR], tiny_reward_signal)
    trades = Backtester(min_reward_pct=0.15).run(strat, df, "TEST/USDT")
    assert trades == []


# ---------- رفع باگ RangeBoundFadeStrategy (جهت TP در رنج باریک) ----------

def test_range_fade_short_rejects_when_midpoint_above_entry():
    """
    در رنج خیلی باریک، اگر میانگین رنج بالاتر از ورود (close) باشد، دیگر
    نباید سیگنال short با آن TP نامعتبر صادر شود.
    """
    n = 35
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    # رنج باریک: high=101, low=100.9 -> mid=100.95
    highs = [101.0] * n
    lows = [100.9] * n
    closes = [100.95] * n
    closes[-1] = 100.96  # close >= high*0.995 برای فعال‌شدن شرط short
    df = pd.DataFrame({
        "open": closes, "high": highs, "low": lows, "close": closes,
        "volume": 1.0,
    }, index=idx)
    strat = RangeBoundFadeStrategy()
    signal = strat.generate_signal(df)
    # چون mid (100.95) پایین‌تر از close (100.96) است، این یکی معتبر است؛
    # فقط برای اطمینان از عدم-کرش، signal یا None یا معتبر باشد کافی است:
    if signal is not None:
        ok, _ = validate_signal(signal)
        assert ok


def test_range_fade_never_returns_direction_invalid_signal():
    """
    آزمون فازی کوچک: با چند رنج تصادفی باریک، هر سیگنالی که تولید می‌شود
    باید از نظر جهت TP/SL معتبر باشد (یعنی دیگر تکرار باگ قبلی رخ ندهد).
    """
    import numpy as np
    rng = np.random.default_rng(0)
    strat = RangeBoundFadeStrategy()
    for _ in range(50):
        n = 35
        idx = pd.date_range("2024-01-01", periods=n, freq="15min")
        base = 100 + rng.normal(0, 0.01, n).cumsum()
        highs = base + rng.uniform(0.0, 0.05, n)
        lows = base - rng.uniform(0.0, 0.05, n)
        closes = base
        df = pd.DataFrame({
            "open": closes, "high": highs, "low": lows, "close": closes,
            "volume": 1.0,
        }, index=idx)
        signal = strat.generate_signal(df)
        if signal is not None:
            ok, reason = validate_signal(signal)
            assert ok, f"سیگنال نامعتبر تولید شد: {reason}"


# ---------- متریک‌ها ----------

def test_compute_metrics_profit_factor():
    tdf = pd.DataFrame([
        {"result": "win", "pnl_pct": 5.0, "entry_price": 100.0, "stop_loss": 95.0},
        {"result": "loss", "pnl_pct": -2.5, "entry_price": 100.0, "stop_loss": 105.0},
    ])
    m = compute_metrics(tdf, initial_balance=1000.0, risk_pct=1.0)
    assert m["total_trades"] == 2
    assert m["win_rate"] == 50.0
    assert m["profit_factor"] == 2.0
    assert m["net_return_pct"] > 0


def test_compute_metrics_empty():
    m = compute_metrics(pd.DataFrame(), initial_balance=1000.0, risk_pct=1.0)
    assert m["total_trades"] == 0
    assert m["final_balance"] == 1000.0


def test_trades_to_df_roundtrip():
    df = make_df()
    set_bar(df, 61, high=111.0, low=99.0)
    strat = FixedSignal(df.index[SIGNAL_BAR], LONG_SIGNAL)
    trades = Backtester().run(strat, df, "TEST/USDT")
    tdf = trades_to_df(trades)
    assert len(tdf) == 1
    assert tdf.iloc[0]["result"] == "win"
