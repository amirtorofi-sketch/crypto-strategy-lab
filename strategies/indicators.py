"""توابع کمکی: اندیکاتورهای کلاسیک + ابزارهای تشخیص ساختار بازار (برای ICT/SMC)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length).mean()


def bollinger_bands(series: pd.Series, length: int = 20, std_mult: float = 2.0):
    mid = sma(series, length)
    std = series.rolling(length).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = atr(df, length)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / length).mean()


def stochastic(df: pd.DataFrame, k_length=14, d_length=3):
    low_min = df["low"].rolling(k_length).min()
    high_max = df["high"].rolling(k_length).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(d_length).mean()
    return k, d


def donchian(df: pd.DataFrame, length: int = 20):
    upper = df["high"].rolling(length).max()
    lower = df["low"].rolling(length).min()
    return upper, lower


def ichimoku(df: pd.DataFrame):
    high, low = df["high"], df["low"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, span_a, span_b


# ---------- ابزارهای ساختار بازار (Market Structure) برای ICT / Smart Money ----------

def swing_points(df: pd.DataFrame, left: int = 2, right: int = 2):
    """نقاط swing high / swing low ساده (فرکتال) - نسخه‌ی وکتورایز با numpy برای سرعت."""
    n = len(df)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    is_sh = np.zeros(n, dtype=bool)
    is_sl = np.zeros(n, dtype=bool)
    win = left + right + 1
    if n >= win:
        # حداکثر/حداقل غلتان متمرکز (centered rolling) با numpy stride tricks
        rmax = pd.Series(highs).rolling(win, center=True).max().to_numpy()
        rmin = pd.Series(lows).rolling(win, center=True).min().to_numpy()
        candidate_sh = highs == rmax
        candidate_sl = lows == rmin
        candidate_sh[:left] = False
        candidate_sh[n - right:] = False
        candidate_sl[:left] = False
        candidate_sl[n - right:] = False
        is_sh = candidate_sh
        is_sl = candidate_sl
    return pd.Series(is_sh, index=df.index), pd.Series(is_sl, index=df.index)


def detect_bos_choch(df: pd.DataFrame, left=2, right=2):
    """
    تشخیص ساده Break of Structure (BOS) و Change of Character (CHoCH)
    بر اساس آخرین سوئینگ‌ها. خروجی: "bullish_bos" | "bearish_bos" |
    "bullish_choch" | "bearish_choch" | None
    """
    sh, sl = swing_points(df, left, right)
    swing_highs = df["high"][sh]
    swing_lows = df["low"][sl]
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    last_close = df["close"].iloc[-1]
    last_high = swing_highs.iloc[-2]  # آخرین سوئینگ‌های تثبیت‌شده (نه لبه)
    last_low = swing_lows.iloc[-2]
    prev_high = swing_highs.iloc[-3] if len(swing_highs) >= 3 else None
    prev_low = swing_lows.iloc[-3] if len(swing_lows) >= 3 else None

    if last_close > last_high:
        if prev_high is not None and last_high < prev_high:
            return "bullish_choch"
        return "bullish_bos"
    if last_close < last_low:
        if prev_low is not None and last_low > prev_low:
            return "bearish_choch"
        return "bearish_bos"
    return None


def find_fair_value_gaps(df: pd.DataFrame, lookback: int = 30):
    """
    Fair Value Gap (FVG) سه‌کندلی: گپ بین high کندل ۱ و low کندل ۳ (بولیش)
    یا بین low کندل ۱ و high کندل ۳ (بریش).
    خروجی: لیست دیکشنری‌های {index, type, top, bottom}
    """
    gaps = []
    sub = df.iloc[-lookback:]
    for i in range(2, len(sub)):
        c1, c3 = sub.iloc[i - 2], sub.iloc[i]
        if c1["high"] < c3["low"]:
            gaps.append({"idx": sub.index[i], "type": "bullish", "top": c3["low"], "bottom": c1["high"]})
        elif c1["low"] > c3["high"]:
            gaps.append({"idx": sub.index[i], "type": "bearish", "top": c1["low"], "bottom": c3["high"]})
    return gaps


def find_order_blocks(df: pd.DataFrame, lookback: int = 40):
    """
    Order Block ساده: آخرین کندل نزولی قبل از یک حرکت صعودی قوی (bullish OB)
    یا آخرین کندل صعودی قبل از حرکت نزولی قوی (bearish OB).
    """
    obs = []
    sub = df.iloc[-lookback:]
    body = (sub["close"] - sub["open"])
    avg_range = (sub["high"] - sub["low"]).mean()
    for i in range(1, len(sub) - 1):
        cur = sub.iloc[i]
        nxt = sub.iloc[i + 1]
        is_down = cur["close"] < cur["open"]
        is_up = cur["close"] > cur["open"]
        strong_up_next = (nxt["close"] - nxt["open"]) > avg_range * 0.8
        strong_down_next = (nxt["open"] - nxt["close"]) > avg_range * 0.8
        if is_down and strong_up_next:
            obs.append({"idx": sub.index[i], "type": "bullish", "top": cur["high"], "bottom": cur["low"]})
        if is_up and strong_down_next:
            obs.append({"idx": sub.index[i], "type": "bearish", "top": cur["high"], "bottom": cur["low"]})
    return obs


def liquidity_sweep(df: pd.DataFrame, left=3, right=3):
    """
    Liquidity Sweep / Stop Hunt: کندلی که فراتر از آخرین swing high/low می‌زند
    اما close داخل رنج قبلی برمی‌گردد (wick بلند = جمع‌آوری نقدینگی).
    خروجی: "buy_side_sweep" (بالای هایِ قبلی زده و برگشته -> بریش)،
            "sell_side_sweep" (زیر لوی قبلی زده و برگشته -> بولیش)، یا None
    """
    sh, sl = swing_points(df, left, right)
    swing_highs = df["high"][sh]
    swing_lows = df["low"][sl]
    if swing_highs.empty or swing_lows.empty:
        return None
    last = df.iloc[-1]
    prev_high = swing_highs.iloc[-1]
    prev_low = swing_lows.iloc[-1]
    if last["high"] > prev_high and last["close"] < prev_high:
        return "buy_side_sweep"
    if last["low"] < prev_low and last["close"] > prev_low:
        return "sell_side_sweep"
    return None
