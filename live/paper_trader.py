"""
دفتر پوزیشن‌های فرضی (paper trading ledger).
روی دیسک به صورت JSON ذخیره می‌شود تا در GitHub Actions بین اجراها
(با git commit) باقی بماند. هر ردیف یک پوزیشن فرضی است، مستقل برای
هر استراتژی + نماد + تایم‌فریم + سشن معاملاتی، تا بعداً بتوان عملکرد
هرکدام را جدا و دقیق تحلیل کرد.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import List, Optional


LEDGER_PATH = "results/live/positions.json"


def get_trading_session(dt_utc: datetime) -> str:
    """
    سشن معاملاتی را بر اساس ساعت UTC زمان ورود تعیین می‌کند.
    بازه‌ها بر اساس ساعات رایج فعالیت هر بازار (تقریبی) هستند:
      - Asia            00:00–07:00 UTC  (توکیو/سیدنی)
      - London           07:00–12:00 UTC
      - London-NY Overlap 12:00–16:00 UTC  (پرنوسان‌ترین بازه)
      - New York          16:00–21:00 UTC
      - Off-hours          21:00–24:00 UTC
    """
    hour = dt_utc.astimezone(timezone.utc).hour
    if 0 <= hour < 7:
        return "Asia"
    if 7 <= hour < 12:
        return "London"
    if 12 <= hour < 16:
        return "London-NY Overlap"
    if 16 <= hour < 21:
        return "New York"
    return "Off-hours"


@dataclass
class PaperPosition:
    id: str
    strategy: str
    category: str
    symbol: str
    timeframe: str
    side: str
    entry_time: str
    entry_price: float
    stop_loss: float
    take_profit: float
    size_usdt: float
    session: str = "Unknown"      # Asia | London | London-NY Overlap | New York | Off-hours
    status: str = "open"          # open | win | loss
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    pnl_usdt: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration_minutes: Optional[float] = None
    reason: str = ""


class PaperLedger:
    def __init__(self, path: str = LEDGER_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.positions: List[PaperPosition] = self._load()

    def _load(self) -> List[PaperPosition]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # سازگاری با دفترهای قدیمی که فیلد timeframe یا session نداشتند
        for p in raw:
            p.setdefault("timeframe", "1h")
            p.setdefault("session", "Unknown")
            p.setdefault("duration_minutes", None)
        return [PaperPosition(**p) for p in raw]

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in self.positions], f, ensure_ascii=False, indent=2)

    def has_open_position(self, strategy: str, symbol: str) -> bool:
        return any(p.strategy == strategy and p.symbol == symbol and p.status == "open" for p in self.positions)

    def open_position(self, strategy: str, category: str, symbol: str, timeframe: str, signal, size_usdt: float) -> PaperPosition:
        entry_dt = datetime.now(timezone.utc)
        pos_id = f"{strategy}__{symbol.replace('/', '-')}__{entry_dt.strftime('%Y%m%dT%H%M%S')}"
        pos = PaperPosition(
            id=pos_id,
            strategy=strategy,
            category=category,
            symbol=symbol,
            timeframe=timeframe,
            side=signal.side,
            entry_time=entry_dt.isoformat(),
            entry_price=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            size_usdt=size_usdt,
            session=get_trading_session(entry_dt),
            reason=signal.reason,
        )
        self.positions.append(pos)
        return pos

    def update_open_positions(self, symbol: str, timeframe: str, high: float, low: float, close: float, fee_pct: float = 0.04):
        """
        با آخرین کندل بسته‌شده‌ی یک (symbol, timeframe) مشخص، فقط پوزیشن‌های
        بازِ همان ترکیب symbol+timeframe را بررسی می‌کند - چون هر استراتژی
        روی تایم‌فریم خودش معامله می‌کند و نباید با کندل تایم‌فریم دیگری چک شود.
        """
        closed_now = []
        for p in self.positions:
            if p.symbol != symbol or p.timeframe != timeframe or p.status != "open":
                continue
            hit_tp = high >= p.take_profit if p.side == "long" else low <= p.take_profit
            hit_sl = low <= p.stop_loss if p.side == "long" else high >= p.stop_loss

            exit_price = None
            status = None
            if hit_sl:
                exit_price, status = p.stop_loss, "loss"
            elif hit_tp:
                exit_price, status = p.take_profit, "win"

            if exit_price is not None:
                direction = 1 if p.side == "long" else -1
                raw_pct = direction * (exit_price - p.entry_price) / p.entry_price * 100
                net_pct = raw_pct - fee_pct * 2
                exit_dt = datetime.now(timezone.utc)
                p.status = status
                p.exit_time = exit_dt.isoformat()
                p.exit_price = exit_price
                p.pnl_pct = round(net_pct, 4)
                p.pnl_usdt = round(p.size_usdt * net_pct / 100, 4)
                p.duration_minutes = round((exit_dt - datetime.fromisoformat(p.entry_time)).total_seconds() / 60, 1)
                closed_now.append(p)
        return closed_now

    def month_stats(self, year: int, month: int):
        rows = [p for p in self.positions
                if p.status in ("win", "loss")
                and datetime.fromisoformat(p.exit_time).year == year
                and datetime.fromisoformat(p.exit_time).month == month]
        by_strategy = {}
        for p in rows:
            by_strategy.setdefault(p.strategy, []).append(p)
        return by_strategy
