"""
موتور بک‌تست - نسخه‌ی ۲

تغییرات نسبت به نسخه‌ی اول:
- شبیه‌سازی درون کندلی: وقتی در یک کندل هم SL و هم TP لمس می‌شود، دو حالت قابل
  انتخاب است (intrabar_mode):
      "conservative" -> همیشه SL زده می‌شود (بدترین حالت؛ مناسب ارزیابی محافظه‌کارانه)
      "distance"     -> فرض می‌کنیم قیمت از open به «نزدیک‌ترین» سطح می‌رسد؛
                        تقریب متداول صنعت وقتی دیتای تیک نداریم
- دو حالت قیمت ورود (entry_price_mode):
      "close"     -> ورود با close کندل سیگنال (رفتار قبلی، پیش‌فرض)
      "next_open" -> ورود با open کندل بعد (واقع‌گرایانه‌تر؛ اسلیپیج واقعی را
                     بهتر مدل می‌کند)
- اعتبارسنجی سیگنال با strategies.base.validate_signal: قبل از باز کردن هر
  پوزیشن، جهت TP/SL و (اختیاری) حداقل فاصله‌ی سودآوری چک می‌شود - دقیقاً
  همان قفلی که در اجرای زنده (live/paper_trader.py) هم استفاده می‌شود، تا
  رفتار بک‌تست و زنده یکسان بماند.

توجه: همان max_lookback قبلی حفظ شده تا رفتار بک‌تست شبیه اجرای زنده بماند
و از کندی O(n^2) جلوگیری شود.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Literal, Optional
import pandas as pd

from strategies.base import Strategy, Signal, validate_signal


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
    def __init__(self, fee_pct: float = 0.04, slippage_pct: float = 0.02,
                 entry_price_mode: Literal["close", "next_open"] = "close",
                 intrabar_mode: Literal["conservative", "distance"] = "conservative",
                 min_reward_pct: float = 0.0):
        if entry_price_mode not in ("close", "next_open"):
            raise ValueError(f"entry_price_mode نامعتبر: {entry_price_mode}")
        if intrabar_mode not in ("conservative", "distance"):
            raise ValueError(f"intrabar_mode نامعتبر: {intrabar_mode}")
        self.fee_pct = fee_pct / 100
        self.slippage_pct = slippage_pct / 100
        self.entry_price_mode = entry_price_mode
        self.intrabar_mode = intrabar_mode
        self.min_reward_pct = min_reward_pct

    def run(self, strategy: Strategy, df: pd.DataFrame, symbol: str,
            max_lookback: int = 300) -> List[Trade]:
        """
        دیتای df باید از قبل روی strategy.timeframe واکشی شده باشد.
        فرض: هر لحظه فقط یک پوزیشن باز به ازای هر استراتژی.
        """
        trades: List[Trade] = []
        open_trade: Trade | None = None
        pending_signal: Signal | None = None   # برای حالت next_open
        min_bars = max(strategy.min_bars, 60)
        window_size = max(max_lookback, min_bars)

        for i in range(min_bars, len(df)):
            bar = df.iloc[i]

            # --- ورود معوق: سیگنالِ کندل قبل، در open همین کندل اجرا می‌شود
            if pending_signal is not None:
                open_trade = Trade(
                    strategy=strategy.name,
                    category=strategy.category,
                    symbol=symbol,
                    timeframe=strategy.timeframe,
                    side=pending_signal.side,
                    entry_time=str(df.index[i]),
                    entry_price=float(bar["open"]),
                    stop_loss=pending_signal.stop_loss,
                    take_profit=pending_signal.take_profit,
                    reason=pending_signal.reason,
                )
                pending_signal = None
                # در حالت next_open باید خروج در همان کندل ورود هم بررسی شود
                exit_kind = self._check_exit(open_trade, bar)
                if exit_kind is not None:
                    exit_price = open_trade.stop_loss if exit_kind == "loss" else open_trade.take_profit
                    self._close(open_trade, df.index[i], exit_price, exit_kind)
                    trades.append(open_trade)
                    open_trade = None
                    continue

            # --- مدیریت پوزیشن باز
            if open_trade is not None:
                exit_kind = self._check_exit(open_trade, bar)
                if exit_kind == "loss":
                    self._close(open_trade, df.index[i], open_trade.stop_loss, "loss")
                    trades.append(open_trade)
                    open_trade = None
                elif exit_kind == "win":
                    self._close(open_trade, df.index[i], open_trade.take_profit, "win")
                    trades.append(open_trade)
                    open_trade = None
                continue  # وقتی پوزیشن باز داریم سیگنال جدید نمی‌گیریم

            # --- تولید سیگنال روی پنجره‌ی محدود (مثل اجرای زنده)
            start = max(0, i + 1 - window_size)
            window = df.iloc[start: i + 1]
            try:
                signal: Signal | None = strategy.generate_signal(window)
            except Exception:
                signal = None

            if signal is not None:
                is_valid, _reason = validate_signal(signal, min_reward_pct=self.min_reward_pct)
                if not is_valid:
                    continue

                if self.entry_price_mode == "next_open":
                    if i + 1 < len(df):
                        pending_signal = signal
                else:
                    open_trade = Trade(
                        strategy=strategy.name,
                        category=strategy.category,
                        symbol=symbol,
                        timeframe=strategy.timeframe,
                        side=signal.side,
                        entry_time=str(df.index[i]),
                        entry_price=signal.entry,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        reason=signal.reason,
                    )

        if open_trade is not None:
            open_trade.result = "open"
            trades.append(open_trade)

        return trades

    def _check_exit(self, trade: Trade, bar: pd.Series) -> Optional[Literal["win", "loss"]]:
        """بررسی برخورد به TP/SL در یک کندل. خروجی: "win" | "loss" | None"""
        if trade.side == "long":
            hit_tp = bar["high"] >= trade.take_profit
            hit_sl = bar["low"] <= trade.stop_loss
        else:
            hit_tp = bar["low"] <= trade.take_profit
            hit_sl = bar["high"] >= trade.stop_loss

        if hit_sl and hit_tp:
            if self.intrabar_mode == "conservative":
                # دیتای تیک نداریم؛ محافظه‌کارانه SL را زده فرض می‌کنیم
                return "loss"
            # حالت distance: قیمت از open ابتدا به نزدیک‌ترین سطح می‌رسد
            d_sl = abs(float(bar["open"]) - trade.stop_loss)
            d_tp = abs(float(bar["open"]) - trade.take_profit)
            return "loss" if d_sl <= d_tp else "win"
        if hit_sl:
            return "loss"
        if hit_tp:
            return "win"
        return None

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
