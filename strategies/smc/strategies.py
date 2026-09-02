from __future__ import annotations
import pandas as pd
from strategies.base import Strategy, Signal
from strategies import indicators as ind


class BOSContinuation(Strategy):
    name = "smc_bos_continuation"
    category = "smc"
    description = "ورود در جهت روند بعد از تایید Break of Structure"
    min_bars = 60
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        structure = ind.detect_bos_choch(df)
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if structure == "bullish_bos":
            return Signal("long", close, close - 2 * atrv, close + 4 * atrv, "تایید BOS صعودی")
        if structure == "bearish_bos":
            return Signal("short", close, close + 2 * atrv, close - 4 * atrv, "تایید BOS نزولی")
        return None


class CHoCHReversal(Strategy):
    name = "smc_choch_reversal"
    category = "smc"
    description = "ورود بازگشتی بعد از Change of Character (تغییر ساختار)"
    min_bars = 60
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        structure = ind.detect_bos_choch(df)
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if structure == "bullish_choch":
            return Signal("long", close, close - 2 * atrv, close + 4 * atrv, "CHoCH صعودی - برگشت روند")
        if structure == "bearish_choch":
            return Signal("short", close, close + 2 * atrv, close - 4 * atrv, "CHoCH نزولی - برگشت روند")
        return None


class OrderBlockRetest(Strategy):
    name = "smc_order_block_retest"
    category = "smc"
    description = "بازگشت قیمت به آخرین Order Block معتبر و واکنش به آن"
    min_bars = 50
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        obs = ind.find_order_blocks(df, lookback=40)
        if not obs:
            return None
        ob = obs[-1]
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if ob["type"] == "bullish" and ob["bottom"] <= close <= ob["top"] * 1.01:
            return Signal("long", close, ob["bottom"] - 0.3 * atrv, close + 3 * atrv, "واکنش به بولیش Order Block")
        if ob["type"] == "bearish" and ob["top"] * 0.99 <= close <= ob["top"]:
            return Signal("short", close, ob["top"] + 0.3 * atrv, close - 3 * atrv, "واکنش به بریش Order Block")
        return None


class FairValueGapFill(Strategy):
    name = "smc_fvg_fill_entry"
    category = "smc"
    description = "ورود در بازگشت قیمت به سمت پر کردن Fair Value Gap"
    min_bars = 40
    timeframe = "15m"

    def generate_signal(self, df: pd.DataFrame):
        gaps = ind.find_fair_value_gaps(df, lookback=30)
        if not gaps:
            return None
        gap = gaps[-1]
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if gap["type"] == "bullish" and gap["bottom"] <= close <= gap["top"]:
            return Signal("long", close, gap["bottom"] - 0.3 * atrv, close + 2.5 * atrv, "ورود در FVG صعودی")
        if gap["type"] == "bearish" and gap["bottom"] <= close <= gap["top"]:
            return Signal("short", close, gap["top"] + 0.3 * atrv, close - 2.5 * atrv, "ورود در FVG نزولی")
        return None


class LiquiditySweepReversal(Strategy):
    name = "smc_liquidity_sweep_reversal"
    category = "smc"
    description = "برگشت بعد از جمع‌آوری نقدینگی (Stop Hunt) در بالای/پایین سوئینگ اخیر"
    min_bars = 40
    timeframe = "15m"

    def generate_signal(self, df: pd.DataFrame):
        sweep = ind.liquidity_sweep(df)
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        last = df.iloc[-1]
        if sweep == "sell_side_sweep":
            return Signal("long", close, last["low"] - 0.2 * atrv, close + 3 * atrv, "جمع‌آوری نقدینگی زیر لو + برگشت")
        if sweep == "buy_side_sweep":
            return Signal("short", close, last["high"] + 0.2 * atrv, close - 3 * atrv, "جمع‌آوری نقدینگی بالای های + برگشت")
        return None


class PremiumDiscountZone(Strategy):
    name = "smc_premium_discount_zone"
    category = "smc"
    description = "ورود از ناحیه دیسکانت (زیر ۵۰٪ رنج) در روند صعودی یا پریمیوم در روند نزولی"
    min_bars = 60
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        window = df.iloc[-50:]
        high, low = window["high"].max(), window["low"].min()
        mid = (high + low) / 2
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        structure = ind.detect_bos_choch(df)
        if structure in ("bullish_bos", "bullish_choch") and close < mid:
            return Signal("long", close, low - 0.3 * atrv, high, "خرید از ناحیه دیسکانت هم‌جهت با روند صعودی")
        if structure in ("bearish_bos", "bearish_choch") and close > mid:
            return Signal("short", close, high + 0.3 * atrv, low, "فروش از ناحیه پریمیوم هم‌جهت با روند نزولی")
        return None


class BreakerBlock(Strategy):
    name = "smc_breaker_block"
    category = "smc"
    description = "Breaker Block: بازگشت به Order Block شکسته‌شده که حالا نقش عکس دارد"
    min_bars = 60
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        obs = ind.find_order_blocks(df, lookback=50)
        structure = ind.detect_bos_choch(df)
        if not obs or structure is None:
            return None
        ob = obs[-1]
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        # اگر یک بولیش OB بود ولی ساختار بعداً نزولی شد -> بریکر نزولی (مقاومت جدید)
        if ob["type"] == "bullish" and structure in ("bearish_bos", "bearish_choch") \
                and ob["bottom"] <= close <= ob["top"] * 1.01:
            return Signal("short", close, ob["top"] + 0.3 * atrv, close - 3 * atrv, "بریکر بلاک نزولی")
        if ob["type"] == "bearish" and structure in ("bullish_bos", "bullish_choch") \
                and ob["top"] * 0.99 <= close <= ob["top"]:
            return Signal("long", close, ob["bottom"] - 0.3 * atrv, close + 3 * atrv, "بریکر بلاک صعودی")
        return None


class EqualHighsLowsLiquidity(Strategy):
    name = "smc_equal_highs_lows_grab"
    category = "smc"
    description = "شکار نقدینگی از های/لوهای برابر (Equal Highs/Lows)"
    min_bars = 50
    timeframe = "15m"

    def generate_signal(self, df: pd.DataFrame):
        sh, sl = ind.swing_points(df, 2, 2)
        highs = df["high"][sh].tail(3)
        lows = df["low"][sl].tail(3)
        close = df["close"].iloc[-1]
        last = df.iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if len(highs) >= 2:
            h = highs.iloc[-2:]
            if abs(h.iloc[0] - h.iloc[1]) / h.iloc[0] < 0.0015 and last["high"] > h.max() and close < h.max():
                return Signal("short", close, last["high"] + 0.2 * atrv, close - 3 * atrv, "شکار های‌های برابر")
        if len(lows) >= 2:
            l = lows.iloc[-2:]
            if abs(l.iloc[0] - l.iloc[1]) / l.iloc[0] < 0.0015 and last["low"] < l.min() and close > l.min():
                return Signal("long", close, last["low"] - 0.2 * atrv, close + 3 * atrv, "شکار لوهای برابر")
        return None


STRATEGIES = [
    BOSContinuation, CHoCHReversal, OrderBlockRetest, FairValueGapFill,
    LiquiditySweepReversal, PremiumDiscountZone, BreakerBlock, EqualHighsLowsLiquidity,
]
