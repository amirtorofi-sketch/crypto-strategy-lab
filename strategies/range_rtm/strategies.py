from __future__ import annotations
import pandas as pd
from strategies.base import Strategy, Signal
from strategies import indicators as ind


class SupplyDemandZoneReaction(Strategy):
    name = "rtm_supply_demand_zone"
    category = "range_rtm"
    description = "واکنش قیمت به ناحیه عرضه/تقاضای تازه (base قبل از حرکت قوی) - سبک RTM"
    min_bars = 50
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        window = df.iloc[-40:]
        atrv = ind.atr(df).iloc[-1]
        close = df["close"].iloc[-1]
        avg_body = (window["close"] - window["open"]).abs().mean()
        for i in range(len(window) - 4, 1, -1):
            base = window.iloc[i - 1]
            move = window.iloc[i]
            base_is_small = abs(base["close"] - base["open"]) < avg_body * 0.6
            move_up = (move["close"] - move["open"]) > avg_body * 1.5
            move_down = (move["open"] - move["close"]) > avg_body * 1.5
            zone_top, zone_bottom = base["high"], base["low"]
            if base_is_small and move_up and zone_bottom <= close <= zone_top * 1.01:
                return Signal("long", close, zone_bottom - 0.2 * atrv, close + 3 * atrv, "واکنش به ناحیه تقاضای تازه")
            if base_is_small and move_down and zone_top * 0.99 <= close <= zone_top:
                return Signal("short", close, zone_top + 0.2 * atrv, close - 3 * atrv, "واکنش به ناحیه عرضه تازه")
        return None


class RangeBoundFadeStrategy(Strategy):
    name = "range_fade_extremes"
    category = "range_rtm"
    description = "معامله در بازار رنج: فروش از سقف رنج و خرید از کف رنج (وقتی ADX پایین است = بدون روند)"
    min_bars = 40
    timeframe = "15m"

    def generate_signal(self, df: pd.DataFrame):
        adx_val = ind.adx(df).iloc[-1]
        if adx_val > 20:
            return None  # فقط در رنج معتبر
        window = df.iloc[-30:]
        high, low = window["high"].max(), window["low"].min()
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if close >= high * 0.995:
            return Signal("short", close, high + 0.5 * atrv, (high + low) / 2, "فروش از سقف رنج")
        if close <= low * 1.005:
            return Signal("long", close, low - 0.5 * atrv, (high + low) / 2, "خرید از کف رنج")
        return None


class RangeBreakoutRetest(Strategy):
    name = "range_breakout_retest"
    category = "range_rtm"
    description = "شکست رنج و سپس بازگشت (retest) موفق به مرز رنج شکسته‌شده"
    min_bars = 50
    timeframe = "15m"

    def generate_signal(self, df: pd.DataFrame):
        window = df.iloc[-30:-2]
        high, low = window["high"].max(), window["low"].min()
        close = df["close"].iloc[-1]
        prev2 = df["close"].iloc[-3]
        atrv = ind.atr(df).iloc[-1]
        broke_up = prev2 > high
        broke_down = prev2 < low
        if broke_up and abs(close - high) < atrv * 0.5 and close > high:
            return Signal("long", close, high - 0.5 * atrv, close + 3 * atrv, "ریتست موفق بعد از شکست سقف رنج")
        if broke_down and abs(close - low) < atrv * 0.5 and close < low:
            return Signal("short", close, low + 0.5 * atrv, close - 3 * atrv, "ریتست موفق بعد از شکست کف رنج")
        return None


class ImbalanceRebalanceRTM(Strategy):
    name = "rtm_imbalance_rebalance"
    category = "range_rtm"
    description = "بازگشت به ناحیه عدم تعادل (نقطه شروع حرکت impulsive) پیش از ادامه حرکت - سبک RTM"
    min_bars = 40
    timeframe = "15m"

    def generate_signal(self, df: pd.DataFrame):
        gaps = ind.find_fair_value_gaps(df, lookback=20)
        if not gaps:
            return None
        gap = gaps[-1]
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        mid = (gap["top"] + gap["bottom"]) / 2
        if gap["type"] == "bullish" and gap["bottom"] <= close <= mid:
            return Signal("long", close, gap["bottom"] - 0.3 * atrv, gap["top"] + 2 * atrv, "ورود از نیمه پایینی ایمبالانس صعودی")
        if gap["type"] == "bearish" and mid <= close <= gap["top"]:
            return Signal("short", close, gap["top"] + 0.3 * atrv, gap["bottom"] - 2 * atrv, "ورود از نیمه بالایی ایمبالانس نزولی")
        return None


STRATEGIES = [
    SupplyDemandZoneReaction, RangeBoundFadeStrategy, RangeBreakoutRetest, ImbalanceRebalanceRTM,
]
