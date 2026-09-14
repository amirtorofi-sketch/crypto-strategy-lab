"""تست‌های واحد برای strategies/indicators.py — همه با دیتای مصنوعی، بدون شبکه."""
import numpy as np
import pandas as pd

from strategies import indicators as ind


def make_df(opens, highs, lows, closes, start="2024-01-01", freq="1h"):
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [1.0] * len(closes),
    }, index=idx)


def flat_df(n=100, price=100.0):
    return make_df([price] * n, [price] * n, [price] * n, [price] * n)


# ---------- اندیکاتورهای کلاسیک ----------

def test_sma_known_values():
    df = flat_df(5)
    df["close"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    s = ind.sma(df["close"], 3)
    assert s.iloc[:2].isna().all()
    assert s.iloc[-1] == (3 + 4 + 5) / 3


def test_ema_rising_series_monotonic():
    closes = pd.Series(np.linspace(100, 200, 50))
    e = ind.ema(closes, 10)
    assert e.iloc[0] == closes.iloc[0]
    assert e.iloc[-1] > e.iloc[-2] > e.iloc[-10]


def test_rsi_bounds_and_warmup():
    rng = np.random.default_rng(42)
    closes = 100 + rng.normal(0, 1, 200).cumsum()
    r = ind.rsi(pd.Series(closes), 14)
    valid = r.dropna()
    assert r.iloc[:13].isna().all()
    assert not valid.empty
    assert valid.between(0, 100).all()


def test_atr_constant_range():
    n = 40
    df = make_df([100.0] * n, [101.0] * n, [99.0] * n, [100.0] * n)
    a = ind.atr(df)
    assert a.iloc[:13].isna().all()
    assert a.iloc[-1] == 2.0  # TR = high - low = 2 در همه کندل‌ها


def test_macd_returns_three_series():
    df = flat_df(60)
    m, s, h = ind.macd(df["close"])
    assert len(m) == len(s) == len(h) == 60


def test_bollinger_bands_width():
    rng = np.random.default_rng(7)
    closes = pd.Series(100 + rng.normal(0, 2, 100).cumsum())
    upper, mid, lower = ind.bollinger_bands(closes)
    assert (upper.dropna() > mid.dropna()).all()
    assert (mid.dropna() > lower.dropna()).all()


# ---------- ابزارهای ساختار بازار ----------

def test_swing_points_finds_single_swing_high():
    n = 11
    highs = [9.0] * n
    highs[5] = 10.0
    df = make_df([8.5] * n, highs, [8.0] * n, [8.5] * n)
    sh, sl = ind.swing_points(df, left=2, right=2)
    assert sh.iloc[5]
    assert sh.sum() == 1


def test_detect_bos_choch_bullish_bos():
    # زیگزاگ با دو سوئینگ های (مقادیر 12 در ایندکس 2 و 7) و شکست در کندل آخر
    highs = [10, 11, 12, 11, 10, 10, 11, 12, 11, 10, 13.5]
    lows = [8, 7, 6, 7, 8, 8, 7, 6, 7, 8, 12.0]
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    closes[-1] = 13.0
    df = make_df([c - 0.1 for c in closes], highs, lows, closes)
    assert ind.detect_bos_choch(df) == "bullish_bos"


def test_detect_bos_choch_none_inside_range():
    highs = [10, 11, 12, 11, 10, 10, 11, 12, 11, 10, 12.5]
    lows = [8, 7, 6, 7, 8, 8, 7, 6, 7, 8, 11.0]
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    closes[-1] = 11.0  # بین آخرین سوئینگ های (12) و لو (6): بدون شکست
    df = make_df([c - 0.1 for c in closes], highs, lows, closes)
    assert ind.detect_bos_choch(df) is None


def test_find_fair_value_gaps_bullish():
    opens = [9.0, 9.6, 11.6, 11.6, 11.7, 11.8]
    highs = [10.0, 10.0, 12.0, 12.0, 12.0, 12.0]
    lows = [8.0, 9.0, 11.0, 11.2, 11.4, 11.5]
    closes = [9.5, 9.6, 11.8, 11.8, 11.8, 11.9]
    df = make_df(opens, highs, lows, closes)
    gaps = ind.find_fair_value_gaps(df)
    assert len(gaps) >= 1
    assert any(g["type"] == "bullish" and g["bottom"] == 10.0 and g["top"] == 11.0 for g in gaps)
    assert all(g["top"] > g["bottom"] for g in gaps)


def test_find_order_blocks_detects_bullish_ob():
    n = 10
    opens = [100.0] * n
    highs = [100.6] * n
    lows = [99.4] * n
    closes = [99.9 + 0.2 * (i % 2) for i in range(n)]
    # کندل ۱ نزولیِ کوچک، کندل ۲ حرکت قوی صعودی -> bullish OB روی کندل ۱
    opens[1], closes[1] = 100.0, 99.0
    highs[1], lows[1] = 100.5, 98.8
    opens[2], closes[2] = 99.2, 100.8
    highs[2], lows[2] = 101.0, 99.0
    df = make_df(opens, highs, lows, closes)
    obs = ind.find_order_blocks(df)
    assert len(obs) >= 1
    assert any(o["type"] == "bullish" for o in obs)


def test_liquidity_sweep_buy_side():
    # یک سوئینگ های واحد در ایندکس ۵ (مقدار ۱۰)، کندل آخر wick بالای آن و close برگشته
    n = 11
    highs = [9.0] * n
    highs[5] = 10.0
    highs[-1] = 11.0
    lows = [8.5] * n
    closes = [8.7] * n
    closes[-1] = 9.5
    opens = [8.6] * n
    opens[-1] = 9.0
    df = make_df(opens, highs, lows, closes)
    assert ind.liquidity_sweep(df) == "buy_side_sweep"


def test_liquidity_sweep_sell_side():
    n = 11
    lows = [8.5] * n
    lows[5] = 8.0
    lows[-1] = 7.5
    highs = [9.5] * n
    closes = [9.3] * n
    closes[-1] = 8.8
    opens = [9.4] * n
    opens[-1] = 8.0
    df = make_df(opens, highs, lows, closes)
    assert ind.liquidity_sweep(df) == "sell_side_sweep"


def test_liquidity_sweep_none_without_wick():
    df = flat_df(11, price=100.0)
    assert ind.liquidity_sweep(df) is None
