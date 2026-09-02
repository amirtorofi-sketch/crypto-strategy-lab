"""
همه استراتژی‌های تعریف‌شده در زیرپوشه‌ها را جمع می‌کند.
برای اضافه‌کردن استراتژی جدید: کلاس آن را در فایل مناسب دسته بنویس
(یا فایل جدید بساز) و آن را به لیست STRATEGIES همان فایل اضافه کن؛
این ماژول خودش پیدایش می‌کند.
"""
from __future__ import annotations
from typing import List, Type
from strategies.base import Strategy

from strategies.classic.strategies import STRATEGIES as CLASSIC
from strategies.price_action.strategies import STRATEGIES as PRICE_ACTION
from strategies.smc.strategies import STRATEGIES as SMC
from strategies.ict.strategies import STRATEGIES as ICT
from strategies.range_rtm.strategies import STRATEGIES as RANGE_RTM


ALL_STRATEGIES: List[Type[Strategy]] = [
    *CLASSIC, *PRICE_ACTION, *SMC, *ICT, *RANGE_RTM,
]


def get_all_strategies() -> List[Strategy]:
    return [cls() for cls in ALL_STRATEGIES]


def get_by_category(category: str) -> List[Strategy]:
    return [s for s in get_all_strategies() if s.category == category]


def get_by_timeframe(timeframe: str) -> List[Strategy]:
    return [s for s in get_all_strategies() if s.timeframe == timeframe]


def get_timeframes() -> List[str]:
    """همه تایم‌فریم‌های یکتایی که حداقل یک استراتژی از آن‌ها استفاده می‌کند."""
    order = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
    used = {s.timeframe for s in get_all_strategies()}
    return [tf for tf in order if tf in used] + sorted(used - set(order))


def get_by_name(name: str) -> Strategy | None:
    for cls in ALL_STRATEGIES:
        if cls.name == name:
            return cls()
    return None


if __name__ == "__main__":
    strategies = get_all_strategies()
    print(f"تعداد کل استراتژی‌های ثبت‌شده: {len(strategies)}")
    by_cat = {}
    for s in strategies:
        by_cat.setdefault(s.category, []).append((s.name, s.timeframe))
    for cat, items in by_cat.items():
        print(f"\n[{cat}] ({len(items)})")
        for n, tf in items:
            print(f"  - {n}  [{tf}]")

    print("\nتایم‌فریم‌های مورد نیاز:", get_timeframes())
