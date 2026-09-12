from __future__ import annotations
import pandas as pd
from strategies.base import Strategy, Signal
from strategies import indicators as ind


class OptimalTradeEntry(Strategy):
    name = "ict_optimal_trade_entry_ote"
    category = "ict"
    description = "ورود در ناحیه OTE (فیبو ۶۲-۷۹٪) بعد از BOS در جهت روند"
    min_bars = 60
    timeframe = "15m"

    def generate_signal(self, df: pd.DataFrame):
        structure = ind.detect_bos_choch(df)
        if structure not in ("bullish_bos", "bearish_bos"):
            return None
        window = df.iloc[-30:]
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if structure == "bullish_bos":
            leg_low, leg_high = window["low"].min(), window["high"].max()
            rng = leg_high - leg_low
            fib_62 = leg_high - 0.62 * rng
            fib_79 = leg_high - 0.79 * rng
            if fib_79 <= close <= fib_62:
                return Signal("long", close, leg_low, leg_high, "ورود در ناحیه OTE صعودی")
        else:
            leg_low, leg_high = window["low"].min(), window["high"].max()
            rng = leg_high - leg_low
            fib_62 = leg_low + 0.62 * rng
            fib_79 = leg_low + 0.79 * rng
            if fib_62 <= close <= fib_79:
                return Signal("short", close, leg_high, leg_low, "ورود در ناحیه OTE نزولی")
        return None


class JudasSwing(Strategy):
    name = "ict_judas_swing"
    category = "ict"
    description = "حرکت فریبنده (Judas Swing) در باز شدن سشن که سریع برمی‌گردد"
    min_bars = 30
    timeframe = "5m"

    def generate_signal(self, df: pd.DataFrame):
        ts = df.index[-1]
        ny_hour = (ts.tz_convert("America/New_York").hour if ts.tzinfo else ts.hour)
        if not (8 <= ny_hour <= 9):
            return None
        sweep = ind.liquidity_sweep(df, left=2, right=2)
        close = df["close"].iloc[-1]
        last = df.iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if sweep == "sell_side_sweep":
            return Signal("long", close, last["low"] - 0.2 * atrv, close + 2.5 * atrv, "Judas Swing صعودی در باز شدن سشن")
        if sweep == "buy_side_sweep":
            return Signal("short", close, last["high"] + 0.2 * atrv, close - 2.5 * atrv, "Judas Swing نزولی در باز شدن سشن")
        return None


class MitigationBlock(Strategy):
    name = "ict_mitigation_block"
    category = "ict"
    description = "بازگشت قیمت به آخرین ناحیه Mitigation (اولین OB خلاف روند قبل از BOS)"
    min_bars = 60
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        structure = ind.detect_bos_choch(df)
        obs = ind.find_order_blocks(df, lookback=50)
        if structure is None or not obs:
            return None
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        ob = obs[0]  # اولین (قدیمی‌ترین) در بازه به عنوان نقطه mitigation
        if structure.startswith("bullish") and ob["type"] == "bearish" and ob["bottom"] <= close <= ob["top"]:
            return Signal("long", close, ob["bottom"] - 0.2 * atrv, close + 3 * atrv, "میتیگیشن بلاک صعودی")
        if structure.startswith("bearish") and ob["type"] == "bullish" and ob["bottom"] <= close <= ob["top"]:
            return Signal("short", close, ob["top"] + 0.2 * atrv, close - 3 * atrv, "میتیگیشن بلاک نزولی")
        return None


class LondonKillzoneBreakout(Strategy):
    name = "ict_london_killzone_breakout"
    category = "ict"
    description = "شکست رنج ساعت افتتاحیه لندن (Opening Range) در پنجره کیل‌زون لندن"
    min_bars = 60
    timeframe = "15m"

    def generate_signal(self, df: pd.DataFrame):
        ts = df.index[-1]
        has_tz = ts.tzinfo is not None
        london_hour = ts.tz_convert("Europe/London").hour if has_tz else ts.hour
        # پنجره شکست: ساعت ۸ تا ۱۰ لندن (بعد از یک ساعت رنج افتتاحیه ۷-۸)
        if not (8 <= london_hour < 10):
            return None

        idx_london = df.index.tz_convert("Europe/London") if has_tz else df.index
        today = ts.tz_convert("Europe/London").date() if has_tz else ts.date()
        mask = (idx_london.date == today) & (idx_london.hour == 7)
        opening_range = df[mask]
        if opening_range.empty:
            return None

        range_high = opening_range["high"].max()
        range_low = opening_range["low"].min()
        rng = range_high - range_low
        if rng <= 0:
            return None

        close = df["close"].iloc[-1]
        if close > range_high:
            return Signal("long", close, range_low, close + 2 * rng, "شکست بالای رنج افتتاحیه لندن")
        if close < range_low:
            return Signal("short", close, range_high, close - 2 * rng, "شکست پایین رنج افتتاحیه لندن")
        return None


STRATEGIES = [
    OptimalTradeEntry, JudasSwing, MitigationBlock, LondonKillzoneBreakout,
]
