"""تست‌های حالت معکوس (--invert): جهت برعکس، SL/TP آینه‌ای حول ورود، مثل ربات‌های دست‌ترید."""
import numpy as np
import pandas as pd
import pytest

from backtest import combo_backtest as cb
from backtest.combo_backtest import mirror_signal, simulate
from strategies.base import Signal, Strategy

from tests.test_combo_15m_all import _dump, _fake_trades

SIGNAL_BAR = 59


class FixedSignal(Strategy):
    name = "test_fixed"
    category = "test"
    min_bars = 5
    timeframe = "1h"

    def __init__(self, target_ts, signal):
        self.target_ts, self._sig = target_ts, signal

    def generate_signal(self, df):
        return self._sig if df.index[-1] == self.target_ts else None


def _df(n=100):
    idx = pd.date_range("2026-01-01 00:00", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0}, index=idx)


def _run(strat, df, invert, fee=0.0):
    allowed = {"London-NY Overlap"}
    return simulate(strat, df, "TEST/USDT", allowed, allowed, window_size=110, min_reward_pct=0.0,
                    trade_value=10.0, fee_pct=fee, slippage_pct=0.0, invert=invert)


def test_mirror_signal_long_and_short():
    assert mirror_signal("long", 100.0, 99.0, 102.0) == ("short", 101.0, 98.0)
    assert mirror_signal("short", 100.0, 101.0, 98.0) == ("long", 99.0, 102.0)


def test_mirror_keeps_risk_reward():
    side, sl, tp = mirror_signal("long", 250.0, 247.5, 257.5)
    sig = Signal(side=side, entry=250.0, stop_loss=sl, take_profit=tp)
    assert sig.rr == Signal(side="long", entry=250.0, stop_loss=247.5, take_profit=257.5).rr == 3.0
    assert side == "short" and sl > 250.0 > tp


def test_simulate_normal_vs_inverted_on_same_price_path():
    df = _df()
    df.iloc[SIGNAL_BAR + 2, df.columns.get_loc("high")] = 102.5     # قیمت بالا می‌رود: TP لانگ (۱۰۲) می‌خورد
    strat = FixedSignal(df.index[SIGNAL_BAR], Signal(side="long", entry=100.0, stop_loss=99.0, take_profit=102.0))
    normal = _run(strat, df, invert=False)
    inv = _run(strat, df, invert=True)
    assert len(normal) == len(inv) == 1
    n, v = normal[0], inv[0]
    assert (n["side"], n["raw_side"], n["exit_reason"]) == ("long", "long", "tp")
    assert (v["side"], v["raw_side"], v["exit_reason"]) == ("short", "long", "sl")
    assert n["pnl_pct"] == pytest.approx(2.0) and v["pnl_pct"] == pytest.approx(-1.0)      # +۲ ریسک‌واحد ... -۱ در معکوس
    assert (v["stop_loss"], v["take_profit"]) == (101.0, 98.0)


def test_simulate_inverted_wins_when_price_falls():
    df = _df()
    df.iloc[SIGNAL_BAR + 2, df.columns.get_loc("low")] = 97.5       # قیمت پایین می‌رود: TP معکوس (۹۸) می‌خورد
    strat = FixedSignal(df.index[SIGNAL_BAR], Signal(side="long", entry=100.0, stop_loss=99.0, take_profit=102.0))
    n, v = _run(strat, df, invert=False)[0], _run(strat, df, invert=True)[0]
    assert (n["exit_reason"], v["exit_reason"]) == ("sl", "tp")
    assert n["pnl_pct"] == pytest.approx(-1.0) and v["pnl_pct"] == pytest.approx(2.0)


def test_inversion_charges_the_same_fee():
    df = _df()
    df.iloc[SIGNAL_BAR + 2, df.columns.get_loc("low")] = 97.5
    strat = FixedSignal(df.index[SIGNAL_BAR], Signal(side="long", entry=100.0, stop_loss=99.0, take_profit=102.0))
    v0 = _run(strat, df, invert=True, fee=0.0)[0]["pnl_pct"]
    v1 = _run(strat, df, invert=True, fee=0.04)[0]["pnl_pct"]
    assert v0 - v1 == pytest.approx(0.08)


def test_merge_refuses_mixed_invert_shards(tmp_path):
    _dump(tmp_path / "shard_1.pkl.gz", _fake_trades(50, 0, "BTC/USDT"), "BTCUSDT", invert=False)
    _dump(tmp_path / "shard_2.pkl.gz", _fake_trades(50, 500, "ETH/USDT"), "ETHUSDT", invert=True)
    with pytest.raises(SystemExit):
        cb.merge_shards([str(tmp_path / "shard_*.pkl.gz")], str(tmp_path / "out"))


def test_report_shows_inverted_warning(tmp_path):
    tr = _fake_trades(200, 0, "BTC/USDT")
    _dump(tmp_path / "shard_1.pkl.gz", tr, "BTCUSDT", invert=True)
    out = tmp_path / "out"
    cb.merge_shards([str(tmp_path / "shard_1.pkl.gz")], str(out))
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "حالت معکوس" in md
    _dump(tmp_path / "shard_2.pkl.gz", tr, "BTCUSDT", invert=False)
    out2 = tmp_path / "out2"
    cb.merge_shards([str(tmp_path / "shard_2.pkl.gz")], str(out2))
    assert "حالت معکوس" not in (out2 / "summary.md").read_text(encoding="utf-8")
