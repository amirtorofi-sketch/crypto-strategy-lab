"""
موتور بک‌تست ساده - بار به بار (bar-by-bar) روی دیتای تاریخی حرکت می‌کند،
به استراتژی اجازه می‌دهد سیگنال بدهد، پوزیشن فرضی باز می‌کند و با برخورد
به TP یا SL آن را می‌بندد (فرض: هر لحظه فقط یک پوزیشن باز به ازای هر استراتژی).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List
import pandas as pd

from strategies.base import Strategy, Signal


@dataclass
class Trade:
    strategy: str
    category: str
    symbol: str
    timeframe: str
    side: str
    entry_time: str
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: str | None = None
    exit_price: float | None = None
    result: str | None = None       # "win" | "loss" | "open"
    pnl_pct: float | None = None
    reason: str = ""


class Backtester:
    def __init__(self, fee_pct: float = 0.04, slippage_pct: float = 0.02):
        self.fee_pct = fee_pct / 100
        self.slippage_pct = slippage_pct / 100

    def run(self, strategy: Strategy, df: pd.DataFrame, symbol: str, max_lookback: int = 300) -> List[Trade]:
        """
        max_lookback: به‌جای پاس‌دادن کل دیتای گذشته (که باعث کندی O(n^2) در
        استراتژی‌هایی مثل تشخیص سوئینگ/ساختار بازار می‌شود)، فقط این تعداد
        کندل آخر به strategy.generate_signal داده می‌شود - دقیقاً مثل رفتار
        واقعیِ اجرای زنده که فقط ohlcv_limit کندل آخر را می‌بیند.
        دیتای df باید از قبل روی strategy.timeframe واکشی شده باشد.
        """
        trades: List[Trade] = []
        open_trade: Trade | None = None
        min_bars = max(strategy.min_bars, 60)
        window_size = max(max_lookback, min_bars)

        for i in range(min_bars, len(df)):
            start = max(0, i + 1 - window_size)
            window = df.iloc[start: i + 1]
            bar = window.iloc[-1]

            # مدیریت پوزیشن باز: بررسی برخورد به SL/TP در همین کندل
            if open_trade is not None:
                hit_tp = (bar["high"] >= open_trade.take_profit) if open_trade.side == "long" \
                    else (bar["low"] <= open_trade.take_profit)
                hit_sl = (bar["low"] <= open_trade.stop_loss) if open_trade.side == "long" \
                    else (bar["high"] >= open_trade.stop_loss)

                if hit_sl and hit_tp:
                    # نمی‌دانیم کدام اول رخ داد؛ محافظه‌کارانه SL را می‌زنیم
                    self._close(open_trade, bar.name, open_trade.stop_loss, "loss")
                    trades.append(open_trade)
                    open_trade = None
                elif hit_sl:
                    self._close(open_trade, bar.name, open_trade.stop_loss, "loss")
                    trades.append(open_trade)
                    open_trade = None
                elif hit_tp:
                    self._close(open_trade, bar.name, open_trade.take_profit, "win")
                    trades.append(open_trade)
                    open_trade = None
                continue  # وقتی پوزیشن باز داریم سیگنال جدید نمی‌گیریم

            try:
                signal: Signal | None = strategy.generate_signal(window)
            except Exception:
                signal = None

            if signal is not None:
                open_trade = Trade(
                    strategy=strategy.name,
                    category=strategy.category,
                    symbol=symbol,
                    timeframe=strategy.timeframe,
                    side=signal.side,
                    entry_time=str(bar.name),
                    entry_price=signal.entry,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    reason=signal.reason,
                )

        if open_trade is not None:
            open_trade.result = "open"
            trades.append(open_trade)

        return trades

    def _close(self, trade: Trade, exit_time, exit_price: float, result: str):
        trade.exit_time = str(exit_time)
        trade.exit_price = exit_price
        trade.result = result
        direction = 1 if trade.side == "long" else -1
        raw_pnl_pct = direction * (exit_price - trade.entry_price) / trade.entry_price * 100
        costs = (self.fee_pct + self.slippage_pct) * 2 * 100
        trade.pnl_pct = round(raw_pnl_pct - costs, 4)


def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in trades])
