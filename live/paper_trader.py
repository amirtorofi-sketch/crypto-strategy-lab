"""
دفتر پوزیشن‌های فرضی (paper trading ledger).
روی دیسک به صورت JSON ذخیره می‌شود تا در GitHub Actions بین اجراها
(با git commit) باقی بماند. هر ردیف یک پوزیشن فرضی است، مستقل برای
هر استراتژی + نماد، تا در پایان ماه بتوان عملکرد هرکدام را جدا دید.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import List, Optional


LEDGER_PATH = "results/live/positions.json"


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
    status: str = "open"          # open | win | loss
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    pnl_usdt: Optional[float] = None
    pnl_pct: Optional[float] = None
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
        # سازگاری با دفترهای قدیمی که فیلد timeframe نداشتند
        for p in raw:
            p.setdefault("timeframe", "1h")
        return [PaperPosition(**p) for p in raw]

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in self.positions], f, ensure_ascii=False, indent=2)

    def has_open_position(self, strategy: str, symbol: str) -> bool:
        return any(p.strategy == strategy and p.symbol == symbol and p.status == "open" for p in self.positions)

    def open_position(self, strategy: str, category: str, symbol: str, timeframe: str, signal, size_usdt: float) -> PaperPosition:
        pos_id = f"{strategy}__{symbol.replace('/', '-')}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        pos = PaperPosition(
            id=pos_id,
            strategy=strategy,
            category=category,
            symbol=symbol,
            timeframe=timeframe,
            side=signal.side,
            entry_time=datetime.now(timezone.utc).isoformat(),
            entry_price=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            size_usdt=size_usdt,
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
                p.status = status
                p.exit_time = datetime.now(timezone.utc).isoformat()
                p.exit_price = exit_price
                p.pnl_pct = round(net_pct, 4)
                p.pnl_usdt = round(p.size_usdt * net_pct / 100, 4)
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
