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


class SilverBulletWindow(Strategy):
    name = "ict_silver_bullet_ny_am"
    category = "ict"
    description = "استراتژی Silver Bullet: FVG داخل پنجره زمانی ۱۰-۱۱ نیویورک بعد از تغییر ساختار"
    min_bars = 40
    timeframe = "5m"

    def generate_signal(self, df: pd.DataFrame):
        ts = df.index[-1]
        ny_hour = (ts.tz_convert("America/New_York").hour if ts.tzinfo else ts.hour)
        if not (10 <= ny_hour < 11):
            return None
        gaps = ind.find_fair_value_gaps(df, lookback=15)
        structure = ind.detect_bos_choch(df)
        if not gaps or structure is None:
            return None
        gap = gaps[-1]
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if gap["type"] == "bullish" and structure.startswith("bullish"):
            return Signal("long", close, gap["bottom"], close + 3 * atrv, "Silver Bullet - FVG صعودی در پنجره NY AM")
        if gap["type"] == "bearish" and structure.startswith("bearish"):
            return Signal("short", close, gap["top"], close - 3 * atrv, "Silver Bullet - FVG نزولی در پنجره NY AM")
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


class PowerOf3Model(Strategy):
    name = "ict_power_of_three"
    category = "ict"
    description = "مدل Power of Three: تجمع -> دستکاری (sweep) -> توزیع در جهت روز"
    min_bars = 50
    timeframe = "1h"

    def generate_signal(self, df: pd.DataFrame):
        recent = df.iloc[-20:]
        rng = recent["high"].max() - recent["low"].min()
        avg_range = ind.atr(df).iloc[-20:].mean()
        is_accumulation = rng < avg_range * 3
        sweep = ind.liquidity_sweep(df)
        structure = ind.detect_bos_choch(df)
        close = df["close"].iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if is_accumulation and sweep == "sell_side_sweep" and structure and structure.startswith("bullish"):
            return Signal("long", close, recent["low"].min() - 0.2 * atrv, close + 3.5 * atrv, "Power of 3: manipulation پایین سپس توزیع صعودی")
        if is_accumulation and sweep == "buy_side_sweep" and structure and structure.startswith("bearish"):
            return Signal("short", close, recent["high"].max() + 0.2 * atrv, close - 3.5 * atrv, "Power of 3: manipulation بالا سپس توزیع نزولی")
        return None


class BreakawayGapImbalance(Strategy):
    name = "ict_breakaway_imbalance"
    category = "ict"
    description = "حرکت پرقدرت (imbalance/inefficiency) که انتظار پرشدن جزئی دارد قبل از ادامه روند"
    min_bars = 30
    timeframe = "15m"

    def generate_signal(self, df: pd.DataFrame):
        c = df.iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        body = abs(c["close"] - c["open"])
        if body < atrv * 1.8:
            return None
        close = c["close"]
        if c["close"] > c["open"]:
            entry = close - body * 0.3
            return Signal("long", close, c["low"], close + 2.5 * atrv, "کندل ایمبالانس صعودی - ادامه روند")
        else:
            return Signal("short", close, c["high"], close - 2.5 * atrv, "کندل ایمبالانس نزولی - ادامه روند")


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


class NewYorkKillzoneReversal(Strategy):
    name = "ict_ny_killzone_reversal"
    category = "ict"
    description = "برگشت قیمت در پنجره کیل‌زون نیویورک (۱۰-۱۱ صبح NY) بعد از جمع‌آوری نقدینگی + تایید ساختار"
    min_bars = 50
    timeframe = "15m"

    def generate_signal(self, df: pd.DataFrame):
        ts = df.index[-1]
        ny_hour = ts.tz_convert("America/New_York").hour if ts.tzinfo else ts.hour
        if not (10 <= ny_hour < 11):
            return None

        sweep = ind.liquidity_sweep(df, left=2, right=2)
        structure = ind.detect_bos_choch(df)
        if sweep is None or structure is None:
            return None

        close = df["close"].iloc[-1]
        last = df.iloc[-1]
        atrv = ind.atr(df).iloc[-1]
        if sweep == "sell_side_sweep" and structure.startswith("bullish"):
            return Signal("long", close, last["low"] - 0.2 * atrv, close + 3 * atrv,
                          "برگشت صعودی در کیل‌زون نیویورک بعد از جمع‌آوری نقدینگی")
        if sweep == "buy_side_sweep" and structure.startswith("bearish"):
            return Signal("short", close, last["high"] + 0.2 * atrv, close - 3 * atrv,
                          "برگشت نزولی در کیل‌زون نیویورک بعد از جمع‌آوری نقدینگی")
        return None


class AsianRangeLiquidityGrab(Strategy):
    name = "ict_asian_range_liquidity_grab"
    category = "ict"
    description = "جمع‌آوری نقدینگی از رنج سشن آسیایی و برگشت در باز شدن سشن لندن"
    min_bars = 60
    timeframe = "15m"

    def generate_signal(self, df: pd.DataFrame):
        ts = df.index[-1]
        has_tz = ts.tzinfo is not None
        london_hour = ts.tz_convert("Europe/London").hour if has_tz else ts.hour
        # فقط در دو ساعت اول باز شدن سشن لندن بررسی کن (وقتی رنج آسیایی تازه شکار می‌شود)
        if not (7 <= london_hour < 9):
            return None

        idx_london = df.index.tz_convert("Europe/London") if has_tz else df.index
        today = ts.tz_convert("Europe/London").date() if has_tz else ts.date()
        # رنج آسیایی تقریبی: نیمه‌شب تا ۷ صبح لندن (تقریباً سشن توکیو/سیدنی)
        mask = (idx_london.date == today) & (idx_london.hour >= 0) & (idx_london.hour < 7)
        asian_bars = df[mask]
        if asian_bars.empty:
            return None

        asian_high = asian_bars["high"].max()
        asian_low = asian_bars["low"].min()
        rng = asian_high - asian_low
        if rng <= 0:
            return None

        last = df.iloc[-1]
        close = last["close"]
        atrv = ind.atr(df).iloc[-1]
        if last["high"] > asian_high and close < asian_high:
            return Signal("short", close, last["high"] + 0.2 * atrv, close - 2 * rng,
                          "جمع‌آوری نقدینگی بالای رنج آسیایی و برگشت")
        if last["low"] < asian_low and close > asian_low:
            return Signal("long", close, last["low"] - 0.2 * atrv, close + 2 * rng,
                          "جمع‌آوری نقدینگی زیر رنج آسیایی و برگشت")
        return None


STRATEGIES = [
    OptimalTradeEntry, SilverBulletWindow, JudasSwing, PowerOf3Model,
    BreakawayGapImbalance, MitigationBlock,
    LondonKillzoneBreakout, NewYorkKillzoneReversal, AsianRangeLiquidityGrab,
]
