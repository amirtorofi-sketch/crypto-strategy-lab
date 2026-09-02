from __future__ import annotations
import pandas as pd
from strategies.base import Strategy, Signal
from strategies import indicators as ind


class BullishBearishEngulfing(Strategy):
    name = "pa_engulfing_candle"
    category = "price_action"
    description = "کندل انگالفینگ صعودی/نزولی در جهت روند کوتاه‌مدت"
    min_bars = 25
    timeframe = "4h"

    def generate_signal(self, df: pd.DataFrame):
        c1, c2 = df.iloc[-2], df.iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        bullish = c1["close"] < c1["open"] and c2["close"] > c2["open"] and \
                  c2["close"] > c1["open"] and c2["open"] < c1["close"]
        bearish = c1["close"] > c1["open"] and c2["close"] < c2["open"] and \
                  c2["close"] < c1["open"] and c2["open"] > c1["close"]
        if bullish:
            return Signal("long", c2["close"], c2["low"] - 0.2 * atrv, c2["close"] + 2 * atrv, "انگالفینگ صعودی")
        if bearish:
            return Signal("short", c2["close"], c2["high"] + 0.2 * atrv, c2["close"] - 2 * atrv, "انگالفینگ نزولی")
        return None


class PinBarRejection(Strategy):
    name = "pa_pin_bar_rejection"
    category = "price_action"
    description = "پین‌بار (کندل با سایه بلند) در نزدیکی سطح حمایت/مقاومت"
    min_bars = 25
    timeframe = "4h"

    def generate_signal(self, df: pd.DataFrame):
        c = df.iloc[-1]
        body = abs(c["close"] - c["open"])
        upper_wick = c["high"] - max(c["close"], c["open"])
        lower_wick = min(c["close"], c["open"]) - c["low"]
        rng = c["high"] - c["low"]
        if rng == 0:
            return None
        atrv = ind.atr(df).iloc[-1]
        if lower_wick > body * 2 and lower_wick > rng * 0.6:
            return Signal("long", c["close"], c["low"] - 0.1 * atrv, c["close"] + 2 * atrv, "پین‌بار صعودی")
        if upper_wick > body * 2 and upper_wick > rng * 0.6:
            return Signal("short", c["close"], c["high"] + 0.1 * atrv, c["close"] - 2 * atrv, "پین‌بار نزولی")
        return None


class InsideBarBreakout(Strategy):
    name = "pa_inside_bar_breakout"
    category = "price_action"
    description = "شکست کندل inside bar (فشردگی نوسان)"
    min_bars = 20
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        mother, inside, cur = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        is_inside = inside["high"] < mother["high"] and inside["low"] > mother["low"]
        if not is_inside:
            return None
        if cur["close"] > inside["high"]:
            return Signal("long", cur["close"], inside["low"], cur["close"] + 2 * atrv, "شکست بالای inside bar")
        if cur["close"] < inside["low"]:
            return Signal("short", cur["close"], inside["high"], cur["close"] - 2 * atrv, "شکست پایین inside bar")
        return None


class DoubleTopBottom(Strategy):
    name = "pa_double_top_bottom"
    category = "price_action"
    description = "الگوی کلاسیک دابل تاپ/باتم روی سوئینگ‌های اخیر"
    min_bars = 60
    timeframe = "4h"

    def generate_signal(self, df: pd.DataFrame):
        sh, sl = ind.swing_points(df, 3, 3)
        highs = df["high"][sh].tail(2)
        lows = df["low"][sl].tail(2)
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if len(highs) == 2:
            h1, h2 = highs.iloc[0], highs.iloc[1]
            if abs(h1 - h2) / h1 < 0.01 and close < min(h1, h2) * 0.99:
                return Signal("short", close, max(h1, h2) + 0.5 * atrv, close - 2.5 * atrv, "دابل تاپ")
        if len(lows) == 2:
            l1, l2 = lows.iloc[0], lows.iloc[1]
            if abs(l1 - l2) / l1 < 0.01 and close > max(l1, l2) * 1.01:
                return Signal("long", close, min(l1, l2) - 0.5 * atrv, close + 2.5 * atrv, "دابل باتم")
        return None


class HeadAndShoulders(Strategy):
    name = "pa_head_and_shoulders"
    category = "price_action"
    description = "الگوی سر و شانه (ساده‌شده بر اساس ۳ سوئینگ های اخیر)"
    min_bars = 80
    timeframe = "1d"

    def generate_signal(self, df: pd.DataFrame):
        sh, _ = ind.swing_points(df, 3, 3)
        highs = df["high"][sh].tail(3)
        if len(highs) < 3:
            return None
        l_sh, head, r_sh = highs.iloc[0], highs.iloc[1], highs.iloc[2]
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        neckline = min(df["low"].iloc[-40:])
        if head > l_sh and head > r_sh and abs(l_sh - r_sh) / l_sh < 0.02 and close < neckline:
            return Signal("short", close, head, close - 2.5 * atrv, "سر و شانه سقف")
        return None


class SupportResistanceBounce(Strategy):
    name = "pa_support_resistance_bounce"
    category = "price_action"
    description = "برخورد و واکنش قیمت به سطح حمایت/مقاومت افقی معتبر (لمس چندباره)"
    min_bars = 60
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        window = df.iloc[-60:]
        atrv = ind.atr(df).iloc[-1]
        close = df["close"].iloc[-1]
        levels_high = window["high"].round(-int(len(str(int(close))) - 2))
        levels_low = window["low"].round(-int(len(str(int(close))) - 2))
        res = levels_high.mode()
        sup = levels_low.mode()
        if not res.empty and abs(close - res.iloc[0]) / close < 0.003 and df["close"].iloc[-2] < df["close"].iloc[-1]:
            return Signal("short", close, close + 1.5 * atrv, close - 3 * atrv, "واکنش به مقاومت افقی")
        if not sup.empty and abs(close - sup.iloc[0]) / close < 0.003 and df["close"].iloc[-2] > df["close"].iloc[-1]:
            return Signal("long", close, close - 1.5 * atrv, close + 3 * atrv, "واکنش به حمایت افقی")
        return None


class TrendlineBreak(Strategy):
    name = "pa_trendline_break"
    category = "price_action"
    description = "شکست خط روند رسم‌شده از دو سوئینگ اخیر هم‌جهت"
    min_bars = 60
    timeframe = "4h"

    def generate_signal(self, df: pd.DataFrame):
        sh, sl = ind.swing_points(df, 3, 3)
        lows = df["low"][sl].tail(2)
        highs = df["high"][sh].tail(2)
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if len(lows) == 2 and lows.iloc[1] > lows.iloc[0]:
            idx0, idx1 = df.index.get_loc(lows.index[0]), df.index.get_loc(lows.index[1])
            slope = (lows.iloc[1] - lows.iloc[0]) / max(idx1 - idx0, 1)
            proj = lows.iloc[1] + slope * (len(df) - 1 - idx1)
            if close < proj and df["close"].iloc[-2] >= proj:
                return Signal("short", close, close + 2 * atrv, close - 3 * atrv, "شکست خط روند صعودی (حمایت دینامیک)")
        if len(highs) == 2 and highs.iloc[1] < highs.iloc[0]:
            idx0, idx1 = df.index.get_loc(highs.index[0]), df.index.get_loc(highs.index[1])
            slope = (highs.iloc[1] - highs.iloc[0]) / max(idx1 - idx0, 1)
            proj = highs.iloc[1] + slope * (len(df) - 1 - idx1)
            if close > proj and df["close"].iloc[-2] <= proj:
                return Signal("long", close, close - 2 * atrv, close + 3 * atrv, "شکست خط روند نزولی (مقاومت دینامیک)")
        return None


class ThreeWhiteSoldiers(Strategy):
    name = "pa_three_white_soldiers_crows"
    category = "price_action"
    description = "سه سرباز سفید (صعودی) / سه کلاغ سیاه (نزولی)"
    min_bars = 20
    timeframe = "4h"

    def generate_signal(self, df: pd.DataFrame):
        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        bullish = all(c["close"] > c["open"] for c in [c1, c2, c3]) and \
                  c1["close"] < c2["close"] < c3["close"]
        bearish = all(c["close"] < c["open"] for c in [c1, c2, c3]) and \
                  c1["close"] > c2["close"] > c3["close"]
        if bullish:
            return Signal("long", c3["close"], c1["low"], c3["close"] + 2.5 * atrv, "سه سرباز سفید")
        if bearish:
            return Signal("short", c3["close"], c1["high"], c3["close"] - 2.5 * atrv, "سه کلاغ سیاه")
        return None


STRATEGIES = [
    BullishBearishEngulfing, PinBarRejection, InsideBarBreakout, DoubleTopBottom,
    HeadAndShoulders, SupportResistanceBounce, TrendlineBreak, ThreeWhiteSoldiers,
]
