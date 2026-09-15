from __future__ import annotations
import pandas as pd
from strategies.base import Strategy, Signal
from strategies import indicators as ind


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


STRATEGIES = [
    PinBarRejection, InsideBarBreakout, DoubleTopBottom,
    SupportResistanceBounce,
]
