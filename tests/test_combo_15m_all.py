"""تست‌های حالت «همه‌ی استراتژی‌ها روی ۱۵ دقیقه، نمادهای پرنقدینگی، اجرای تکه‌تکه و ادغام»."""
import gzip
import pickle

import numpy as np
import pandas as pd
import pytest

from backtest import combo_backtest as cb
from backtest.combos import (BOT_SYMBOLS, EXCLUDED_STRATEGIES, LIQUID_SYMBOLS, SESSIONS, parse_shard,
                             pick_symbols)
from strategies.registry import get_all_strategies, get_by_name


# ------------------------------------------------------------------ نمادها و تکه‌ها
def test_liquid_symbols_are_valid():
    assert 30 <= len(LIQUID_SYMBOLS) <= 40
    assert len(set(LIQUID_SYMBOLS)) == len(LIQUID_SYMBOLS)
    assert set(LIQUID_SYMBOLS) <= set(BOT_SYMBOLS)


def test_shards_partition_the_symbols_exactly():
    n = 6
    parts = [pick_symbols("liquid", f"{i}/{n}") for i in range(1, n + 1)]
    flat = [s for p in parts for s in p]
    assert sorted(flat) == sorted(LIQUID_SYMBOLS) and len(flat) == len(set(flat))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_symbol_set_all_and_explicit():
    assert len(pick_symbols("all")) == len(BOT_SYMBOLS)
    assert pick_symbols("liquid", None, ["BTC/USDT", "ETHUSDT"]) == ["BTCUSDT", "ETHUSDT"]
    with pytest.raises(ValueError):
        pick_symbols("nope")


@pytest.mark.parametrize("bad", ["0/6", "7/6", "a/b", "3", "1/0"])
def test_parse_shard_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        parse_shard(bad)


def test_excluded_strategies_exist_and_twenty_remain():
    names = {s.name for s in get_all_strategies()}
    assert EXCLUDED_STRATEGIES <= names
    assert len(EXCLUDED_STRATEGIES) == 3
    assert len(names - EXCLUDED_STRATEGIES) == 20


# ------------------------------------------------------------------ تایم‌فریم
def test_resample_rejects_timeframes_not_buildable_from_15m():
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
                      index=pd.date_range("2026-01-01", periods=10, freq="15min", tz="UTC"))
    for tf in ("5m", "1d"):
        with pytest.raises(ValueError):
            cb.resample_tf(df, tf)
    assert cb.resample_tf(df, "15m") is df


def _write_csv(path, n=2500, seed=1):
    r = np.random.default_rng(seed)
    t = pd.date_range("2026-03-01", periods=n, freq="15min")
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    pd.DataFrame({"open_time": t, "open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999,
                  "close": c, "volume": r.uniform(10, 100, n)}).to_csv(path, index=False)


def test_run_symbol_timeframe_override_does_not_touch_registry(tmp_path):
    p = tmp_path / "TESTUSDT_15m.csv"
    _write_csv(p)
    name = "classic_bollinger_mean_reversion"        # استراتژی ۱ساعته‌ی سبک
    native = get_by_name(name).timeframe
    assert native != "15m"
    job = {"symbol": "TESTUSDT", "path": str(p), "allowed": {name: list(SESSIONS)}, "selected": {}, "windows": {},
           "tf_override": "15m", "buffer_bars": 50, "max_fetch": 1000, "min_reward_pct": 0.0,
           "trade_value": 10.0, "fee_pct": 0.04, "slippage_pct": 0.0, "independent": True}
    sym, rows, meta = cb.run_symbol(job)
    assert sym == "TESTUSDT" and meta["trades"] == len(rows)
    assert rows, "روی داده‌ی تصادفی باید دست‌کم چند معامله ثبت شود"
    assert {r["timeframe"] for r in rows} == {"15m"}
    assert not any(r["selected"] for r in rows)
    assert get_by_name(name).timeframe == native       # کپی سطحی؛ رجیستری دست‌نخورده


# ------------------------------------------------------------------ ادغام تکه‌ها
def _fake_trades(n, offset_h, symbol):
    rng = np.random.default_rng(offset_h)
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for i in range(n):
        pnl = float(rng.choice([-0.1, 0.2], p=[0.6, 0.4]))
        rows.append({"strategy": "s1" if i % 2 else "s2", "category": "x", "symbol": symbol, "timeframe": "15m",
                     "session": SESSIONS[i % 5], "selected": False, "side": "long",
                     "entry_time": (t0 + pd.Timedelta(hours=offset_h + i)).isoformat(),
                     "exit_time": (t0 + pd.Timedelta(hours=offset_h + i + 2)).isoformat(),
                     "entry_price": 100.0, "stop_loss": 99.0, "take_profit": 102.0, "exit_price": 101.0,
                     "exit_reason": "tp", "pnl_pct": pnl * 10, "pnl_usdt": pnl, "win": pnl > 0,
                     "duration_minutes": 120.0, "reason": ""})
    return pd.DataFrame(rows)


def _dump(path, trades, sym, **over):
    params = {"fee_pct": 0.04, "slippage_pct": 0.0, "trade_value": 10.0, "min_reward_pct": 0.15, "windows": {},
              "all_sessions": True, "timeframe_override": "15m", "strategies": ["s1", "s2"],
              "symbols_requested": 1, "symbols_with_data": 1, "symbols_missing": [], "combos": {}}
    params.update(over)
    with gzip.open(path, "wb") as f:
        pickle.dump({"trades": trades, "metas": {sym: {"trades": len(trades)}}, "params": params}, f)


def test_merge_shards_builds_report_without_selected_combos(tmp_path):
    _dump(tmp_path / "shard_1.pkl.gz", _fake_trades(300, 0, "BTC/USDT"), "BTCUSDT")
    _dump(tmp_path / "shard_2.pkl.gz", _fake_trades(300, 500, "ETH/USDT"), "ETHUSDT")
    out = tmp_path / "out"
    cb.merge_shards([str(tmp_path / "shard_*.pkl.gz")], str(out))
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "همه‌ی ترکیب‌ها" in md and "تایم‌فریم **15m**" in md
    assert "چک سوگیری انتخاب" not in md
    df = pd.read_csv(out / "by_combo.csv")
    assert df["n"].sum() == 600 and set(df["strategy"]) == {"s1", "s2"}
    assert set(pd.read_csv(out / "by_symbol.csv")["symbol"]) == {"BTC/USDT", "ETH/USDT"}
    html = (out / "dashboard.html").read_text(encoding="utf-8")
    assert "sel_label" in html


def test_merge_shards_refuses_mismatched_parameters(tmp_path):
    _dump(tmp_path / "shard_1.pkl.gz", _fake_trades(50, 0, "BTC/USDT"), "BTCUSDT")
    _dump(tmp_path / "shard_2.pkl.gz", _fake_trades(50, 500, "ETH/USDT"), "ETHUSDT", fee_pct=0.35)
    with pytest.raises(SystemExit):
        cb.merge_shards([str(tmp_path / "shard_*.pkl.gz")], str(tmp_path / "out"))


# ------------------------------------------------------------------ حالت «تایم‌فریم اصلی هر استراتژی»
def test_split_buildable_skips_5m_and_1d_only():
    names = [s.name for s in get_all_strategies() if s.name not in EXCLUDED_STRATEGIES]
    ok, skipped = cb.split_buildable(names)
    assert set(skipped) == {"ict_judas_swing", "classic_golden_cross_50_200"}
    assert len(ok) == 18 and {get_by_name(n).timeframe for n in ok} == {"15m", "1h", "4h"}


def test_report_lists_native_timeframes(tmp_path):
    tr = _fake_trades(200, 0, "BTC/USDT")
    tr["timeframe"] = np.where(tr["strategy"] == "s1", "1h", "4h")
    _dump(tmp_path / "shard_1.pkl.gz", tr, "BTCUSDT", timeframe_override=None, skipped_unbuildable=["ict_judas_swing"])
    out = tmp_path / "out"
    cb.merge_shards([str(tmp_path / "shard_1.pkl.gz")], str(out))
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "تایم‌فریم اصلی خودش" in md and "ict_judas_swing" in md and "**1h**" in md and "**4h**" in md
