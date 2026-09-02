from __future__ import annotations
import pandas as pd
from strategies.base import Strategy, Signal
from strategies import indicators as ind


class MACrossover(Strategy):
    name = "classic_ma_crossover_20_50"
    category = "classic"
    description = "کراس EMA20 و EMA50 (طلایی/مرگ کوچک)"
    min_bars = 60
    timeframe = "4h"

    def generate_signal(self, df: pd.DataFrame):
        e20 = ind.ema(df["close"], 20)
        e50 = ind.ema(df["close"], 50)
        if e20.iloc[-2] < e50.iloc[-2] and e20.iloc[-1] > e50.iloc[-1]:
            entry = df["close"].iloc[-1]
            atrv = ind.atr(df).iloc[-1]
            return Signal("long", entry, entry - 1.5 * atrv, entry + 3 * atrv, "کراس صعودی EMA20/50")
        if e20.iloc[-2] > e50.iloc[-2] and e20.iloc[-1] < e50.iloc[-1]:
            entry = df["close"].iloc[-1]
            atrv = ind.atr(df).iloc[-1]
            return Signal("short", entry, entry + 1.5 * atrv, entry - 3 * atrv, "کراس نزولی EMA20/50")
        return None


class GoldenCross50_200(Strategy):
    name = "classic_golden_cross_50_200"
    category = "classic"
    description = "کراس طلایی/مرگ SMA50 و SMA200"
    min_bars = 210
    timeframe = "1d"

    def generate_signal(self, df: pd.DataFrame):
        s50 = ind.sma(df["close"], 50)
        s200 = ind.sma(df["close"], 200)
        atrv = ind.atr(df).iloc[-1]
        entry = df["close"].iloc[-1]
        if s50.iloc[-2] < s200.iloc[-2] and s50.iloc[-1] > s200.iloc[-1]:
            return Signal("long", entry, entry - 2 * atrv, entry + 5 * atrv, "Golden Cross 50/200")
        if s50.iloc[-2] > s200.iloc[-2] and s50.iloc[-1] < s200.iloc[-1]:
            return Signal("short", entry, entry + 2 * atrv, entry - 5 * atrv, "Death Cross 50/200")
        return None


class RSIOversoldOverbought(Strategy):
    name = "classic_rsi_reversal_30_70"
    category = "classic"
    description = "برگشت از اشباع خرید/فروش RSI(14)"
    min_bars = 30
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        r = ind.rsi(df["close"], 14)
        atrv = ind.atr(df).iloc[-1]
        entry = df["close"].iloc[-1]
        if r.iloc[-2] < 30 and r.iloc[-1] >= 30:
            return Signal("long", entry, entry - 1.5 * atrv, entry + 3 * atrv, "خروج RSI از اشباع فروش")
        if r.iloc[-2] > 70 and r.iloc[-1] <= 70:
            return Signal("short", entry, entry + 1.5 * atrv, entry - 3 * atrv, "خروج RSI از اشباع خرید")
        return None


class MACDCrossover(Strategy):
    name = "classic_macd_signal_cross"
    category = "classic"
    description = "کراس خط MACD و خط سیگنال"
    min_bars = 40
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        macd_line, signal_line, _ = ind.macd(df["close"])
        atrv = ind.atr(df).iloc[-1]
        entry = df["close"].iloc[-1]
        if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            return Signal("long", entry, entry - 1.5 * atrv, entry + 3 * atrv, "کراس صعودی MACD")
        if macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            return Signal("short", entry, entry + 1.5 * atrv, entry - 3 * atrv, "کراس نزولی MACD")
        return None


class BollingerBandBounce(Strategy):
    name = "classic_bollinger_mean_reversion"
    category = "classic"
    description = "برگشت قیمت از باند بولینگر به سمت میانگین"
    min_bars = 30
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        upper, mid, lower = ind.bollinger_bands(df["close"])
        close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        if prev_close < lower.iloc[-2] and close > lower.iloc[-1]:
            return Signal("long", close, lower.iloc[-1] * 0.995, mid.iloc[-1], "برگشت از باند پایین بولینگر")
        if prev_close > upper.iloc[-2] and close < upper.iloc[-1]:
            return Signal("short", close, upper.iloc[-1] * 1.005, mid.iloc[-1], "برگشت از باند بالای بولینگر")
        return None


class BollingerBreakout(Strategy):
    name = "classic_bollinger_breakout"
    category = "classic"
    description = "شکست باند بولینگر همراه با انبساط باند (اسکوییز)"
    min_bars = 40
    timeframe = "4h"

    def generate_signal(self, df: pd.DataFrame):
        upper, mid, lower = ind.bollinger_bands(df["close"])
        bandwidth = (upper - lower) / mid
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if bandwidth.iloc[-2] < bandwidth.rolling(50).mean().iloc[-2] * 0.7:
            if close > upper.iloc[-1]:
                return Signal("long", close, mid.iloc[-1], close + 2 * atrv, "شکست بولینگر بعد از اسکوییز")
            if close < lower.iloc[-1]:
                return Signal("short", close, mid.iloc[-1], close - 2 * atrv, "شکست بولینگر بعد از اسکوییز")
        return None


class StochasticCross(Strategy):
    name = "classic_stochastic_cross"
    category = "classic"
    description = "کراس %K و %D استوکاستیک در نواحی اشباع"
    min_bars = 30
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        k, d = ind.stochastic(df)
        atrv = ind.atr(df).iloc[-1]
        entry = df["close"].iloc[-1]
        if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 30:
            return Signal("long", entry, entry - 1.5 * atrv, entry + 3 * atrv, "کراس صعودی استوکاستیک از اشباع فروش")
        if k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 70:
            return Signal("short", entry, entry + 1.5 * atrv, entry - 3 * atrv, "کراس نزولی استوکاستیک از اشباع خرید")
        return None


class DonchianBreakout(Strategy):
    name = "classic_donchian_channel_breakout"
    category = "classic"
    description = "شکست کانال دانچیان ۲۰ کندلی (سبک ترتل تریدرز)"
    min_bars = 30
    timeframe = "4h"

    def generate_signal(self, df: pd.DataFrame):
        upper, lower = ind.donchian(df, 20)
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if close > upper.iloc[-2]:
            return Signal("long", close, close - 2 * atrv, close + 4 * atrv, "شکست سقف کانال دانچیان")
        if close < lower.iloc[-2]:
            return Signal("short", close, close + 2 * atrv, close - 4 * atrv, "شکست کف کانال دانچیان")
        return None


class ADXTrendFollowing(Strategy):
    name = "classic_adx_trend_following"
    category = "classic"
    description = "ورود در جهت روند وقتی ADX>25 و جهت +DI/-DI تایید کند"
    min_bars = 40
    timeframe = "4h"

    def generate_signal(self, df: pd.DataFrame):
        adx_val = ind.adx(df)
        e20 = ind.ema(df["close"], 20)
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if adx_val.iloc[-1] > 25:
            if close > e20.iloc[-1] and df["close"].iloc[-2] <= e20.iloc[-2]:
                return Signal("long", close, close - 1.5 * atrv, close + 3 * atrv, "روند قوی ADX + بازگشت به EMA20")
            if close < e20.iloc[-1] and df["close"].iloc[-2] >= e20.iloc[-2]:
                return Signal("short", close, close + 1.5 * atrv, close - 3 * atrv, "روند قوی ADX + بازگشت به EMA20")
        return None


class IchimokuKumoBreakout(Strategy):
    name = "classic_ichimoku_kumo_breakout"
    category = "classic"
    description = "شکست ابر کومو ایچیموکو همراه با تنکان/کیجون"
    min_bars = 60
    timeframe = "4h"

    def generate_signal(self, df: pd.DataFrame):
        tenkan, kijun, span_a, span_b = ind.ichimoku(df)
        close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        top_cloud = max(span_a.iloc[-1], span_b.iloc[-1])
        bottom_cloud = min(span_a.iloc[-1], span_b.iloc[-1])
        atrv = ind.atr(df).iloc[-1]
        if prev_close <= top_cloud and close > top_cloud and tenkan.iloc[-1] > kijun.iloc[-1]:
            return Signal("long", close, bottom_cloud, close + 3 * atrv, "شکست بالای ابر کومو")
        if prev_close >= bottom_cloud and close < bottom_cloud and tenkan.iloc[-1] < kijun.iloc[-1]:
            return Signal("short", close, top_cloud, close - 3 * atrv, "شکست پایین ابر کومو")
        return None


class VolumeSpikeBreakout(Strategy):
    name = "classic_volume_spike_breakout"
    category = "classic"
    description = "شکست همراه با اسپایک حجم (بیش از ۲ برابر میانگین)"
    min_bars = 30
    timeframe = "15m"

    def generate_signal(self, df: pd.DataFrame):
        vol_avg = df["volume"].rolling(20).mean()
        high20 = df["high"].rolling(20).max()
        low20 = df["low"].rolling(20).min()
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if df["volume"].iloc[-1] > 2 * vol_avg.iloc[-2] and close > high20.iloc[-2]:
            return Signal("long", close, close - 2 * atrv, close + 4 * atrv, "شکست سقف با اسپایک حجم")
        if df["volume"].iloc[-1] > 2 * vol_avg.iloc[-2] and close < low20.iloc[-2]:
            return Signal("short", close, close + 2 * atrv, close - 4 * atrv, "شکست کف با اسپایک حجم")
        return None


STRATEGIES = [
    MACrossover, GoldenCross50_200, RSIOversoldOverbought, MACDCrossover,
    BollingerBandBounce, BollingerBreakout, StochasticCross, DonchianBreakout,
    ADXTrendFollowing, IchimokuKumoBreakout, VolumeSpikeBreakout,
]
