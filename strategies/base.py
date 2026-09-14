"""
هر استراتژی یک کلاس است که از Strategy ارث‌بری می‌کند و متد
`generate_signal(df)` را پیاده می‌کند.

خروجی generate_signal یک شیء Signal یا None است.
df: یک pandas.DataFrame با ستون‌های open/high/low/close/volume
    که آخرین ردیف = آخرین کندل بسته‌شده است.

این طراحی عمداً ساده نگه داشته شده تا اضافه‌کردن استراتژی جدید
(برای رسیدن به ۱۰۰ تا) فقط یعنی یک فایل جدید کوچک با همین الگو.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
import pandas as pd


@dataclass
class Signal:
    side: Literal["long", "short"]
    entry: float
    stop_loss: float
    take_profit: float
    reason: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def rr(self) -> float:
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.take_profit - self.entry)
        return round(reward / risk, 2) if risk else 0.0


class Strategy:
    name: str = "base"
    category: str = "generic"   # ict | smc | price_action | classic | range_rtm
    description: str = ""
    min_bars: int = 50          # حداقل تعداد کندل مورد نیاز
    timeframe: str = "1h"       # تایم‌فریم پیشنهادی این استراتژی (بر اساس منطق خودش)

    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        raise NotImplementedError

    def __repr__(self):
        return f"<Strategy {self.name} ({self.category}, {self.timeframe})>"


def validate_signal(signal: Optional[Signal], min_reward_pct: float = 0.0) -> tuple[bool, str]:
    """
    اعتبارسنجی مرکزی یک سیگنال - هم در بک‌تست و هم در اجرای زنده استفاده
    می‌شود تا هر دو دقیقاً یک منطق را رعایت کنند.

    دو نوع بررسی انجام می‌دهد:
    ۱) صحت جهت (همیشه چک می‌شود، صرف نظر از min_reward_pct): برای یک
       پوزیشن long، حد سود باید بالاتر و حد ضرر باید پایین‌تر از ورود
       باشد (برعکسش برای short). این یک باگ واقعی را می‌گیرد - نه یک
       تنظیم قابل‌بحث - چون اگر رعایت نشود، حتی رسیدن قیمت به "حد سود"
       ممکن است در واقعیت به‌معنی ضرر باشد.
    ۲) حداقل فاصله‌ی سودآوری (فقط اگر min_reward_pct > 0 داده شود): اگر
       فاصله‌ی حد سود تا ورود آن‌قدر کوچک باشد که کارمزد رفت‌وبرگشت کل
       سود را بخورد، سیگنال رد می‌شود.

    خروجی: (معتبر است؟, دلیل رد در صورت نامعتبر بودن)
    """
    if signal is None:
        return False, "سیگنال خالی است"
    if signal.entry <= 0:
        return False, "قیمت ورود باید مثبت باشد"

    if signal.side == "long":
        if signal.take_profit <= signal.entry:
            return False, "برای پوزیشن long، حد سود باید بالاتر از قیمت ورود باشد"
        if signal.stop_loss >= signal.entry:
            return False, "برای پوزیشن long، حد ضرر باید پایین‌تر از قیمت ورود باشد"
    elif signal.side == "short":
        if signal.take_profit >= signal.entry:
            return False, "برای پوزیشن short، حد سود باید پایین‌تر از قیمت ورود باشد"
        if signal.stop_loss <= signal.entry:
            return False, "برای پوزیشن short، حد ضرر باید بالاتر از قیمت ورود باشد"
    else:
        return False, f"side نامعتبر: {signal.side!r}"

    if min_reward_pct > 0:
        reward_pct = abs(signal.take_profit - signal.entry) / signal.entry * 100
        if reward_pct < min_reward_pct:
            return False, (
                f"فاصله‌ی حد سود تا ورود ({reward_pct:.4f}%) کمتر از حداقل "
                f"لازم برای پوشش کارمزد ({min_reward_pct}%) است"
            )

    return True, ""
