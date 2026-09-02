"""
اجرای زنده (live signal runner) - نسخه چندتایم‌فریمی:

هر استراتژی روی تایم‌فریم مخصوص به خودش (strategy.timeframe) اجرا می‌شود -
مثلاً استراتژی‌های ICT مثل Silver Bullet روی 5m، و Golden Cross روی 1d.
این اسکریپت:

۱. استراتژی‌ها را بر اساس تایم‌فریم گروه‌بندی می‌کند.
۲. برای هر نماد + هر تایم‌فریمِ موردنیاز، فقط یک‌بار دیتا از بایننس می‌گیرد
   (نه یک‌بار به‌ازای هر استراتژی) تا تعداد درخواست‌ها کم بماند.
۳. پوزیشن‌های باز همان (نماد، تایم‌فریم) را با آخرین کندل چک می‌کند؛ اگر به
   TP/SL خورده باشند می‌بندد و پیام می‌فرستد.
۴. استراتژی‌های همان تایم‌فریم را روی دیتا اجرا می‌کند و در صورت سیگنال
   جدید (و نبودِ پوزیشن باز برای آن استراتژی/نماد) پوزیشن فرضی باز می‌کند.
۵. هر پیام (باز شدن/بسته شدن) به تاپیک تلگرامِ مخصوص همان استراتژی می‌رود
   (اگر گروه Forum باشد)، وگرنه به‌صورت معمولی با تگ استراتژی.

چون هر اجرا فقط وقتی پوزیشن جدید باز می‌کند که پوزیشن باز دیگری برای همان
استراتژی/نماد نباشد، اجرای این اسکریپت هر چند دقیقه یک‌بار (حتی بیشتر از
تعداد کندل‌های واقعی) کاملاً بی‌خطر است و سیگنال تکراری نمی‌فرستد.

اجرا:
    python -m live.runner
"""
from __future__ import annotations
import os
import yaml
from collections import defaultdict

from data.fetcher import get_exchange, fetch_ohlcv_df
from strategies.registry import get_all_strategies
from live.telegram_bot import (
    send_telegram_message, format_signal_message, format_close_message, get_or_create_topic,
)
from live.paper_trader import PaperLedger


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        path = "config.example.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def group_strategies_by_timeframe():
    groups = defaultdict(list)
    for s in get_all_strategies():
        groups[s.timeframe].append(s)
    return groups


def notify(tg_cfg: dict, strategy_name: str, text: str):
    thread_id = get_or_create_topic(
        tg_cfg["bot_token"], tg_cfg["chat_id"], strategy_name,
        use_topics=tg_cfg.get("use_topics", True),
    )
    send_telegram_message(tg_cfg["bot_token"], tg_cfg["chat_id"], text, message_thread_id=thread_id)


def main():
    cfg = load_config()
    tg = cfg["telegram"]
    ex_cfg = cfg["exchange"]
    pt_cfg = cfg["paper_trading"]
    buffer_bars = ex_cfg.get("extra_bars_buffer", 50)
    max_fetch = ex_cfg.get("max_fetch_bars", 1000)

    ex = get_exchange(ex_cfg["id"])
    groups = group_strategies_by_timeframe()
    ledger = PaperLedger()

    print(f"تایم‌فریم‌های فعال: {list(groups.keys())}")

    for symbol in ex_cfg["symbols"]:
        for timeframe, strategies in groups.items():
            needed_bars = min(max(s.min_bars for s in strategies) + buffer_bars, max_fetch)
            print(f"\n=== {symbol} [{timeframe}] ({len(strategies)} استراتژی، {needed_bars} کندل) ===")

            try:
                df = fetch_ohlcv_df(ex, symbol, timeframe, needed_bars)
            except Exception as e:
                print(f"  خطا در واکشی دیتا: {e}")
                continue

            if df.empty or len(df) < 60:
                print("  دیتای کافی نیست، رد شد.")
                continue

            last = df.iloc[-1]

            # ۱) بررسی بسته‌شدن پوزیشن‌های باز روی این (symbol, timeframe)
            closed = ledger.update_open_positions(symbol, timeframe, last["high"], last["low"], last["close"])
            for p in closed:
                text = format_close_message(p)
                notify(tg, p.strategy, text)
                print(f"  [{p.strategy}] بسته شد: {p.status} ({p.pnl_pct}%)")

            # ۲) بررسی سیگنال جدید برای هر استراتژی این تایم‌فریم
            for strat in strategies:
                if len(df) < strat.min_bars:
                    continue
                if ledger.has_open_position(strat.name, symbol):
                    continue
                try:
                    signal = strat.generate_signal(df)
                except Exception as e:
                    print(f"  [{strat.name}] خطا: {e}")
                    continue
                if signal is None:
                    continue

                risk_amount = pt_cfg["initial_balance_usdt"] * (pt_cfg["risk_per_trade_pct"] / 100)
                stop_dist_pct = abs(signal.entry - signal.stop_loss) / signal.entry * 100
                size_usdt = risk_amount / (stop_dist_pct / 100) if stop_dist_pct > 0 else 0

                ledger.open_position(strat.name, strat.category, symbol, timeframe, signal, size_usdt)
                text = format_signal_message(strat.name, strat.category, timeframe, symbol, signal)
                notify(tg, strat.name, text)
                print(f"  [{strat.name}] سیگنال جدید: {signal.side} @ {signal.entry}")

    ledger.save()
    print("\nدفتر پوزیشن‌ها ذخیره شد:", ledger.path)


if __name__ == "__main__":
    main()
