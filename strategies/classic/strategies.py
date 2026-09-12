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


STRATEGIES = [
    MACrossover, GoldenCross50_200, RSIOversoldOverbought, MACDCrossover,
    BollingerBandBounce, StochasticCross,
]
