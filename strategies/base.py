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
