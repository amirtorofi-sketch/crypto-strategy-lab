"""
اجرای بک‌تست برای همه (یا برخی) استراتژی‌ها روی نمادهای تعریف‌شده در config.yaml.

هر استراتژی روی تایم‌فریم مخصوص به خودش (strategy.timeframe) بک‌تست می‌شود -
دقیقاً همان تایم‌فریمی که در اجرای زنده هم استفاده می‌کند. برای هر (نماد،
تایم‌فریم) دیتا فقط یک‌بار واکشی می‌شود، بعد بین همه استراتژی‌های آن
تایم‌فریم به اشتراک گذاشته می‌شود.

نتیجه در results/backtest_summary.csv و results/trades/<strategy>.csv

اجرا:
    python -m backtest.run_backtest
    python -m backtest.run_backtest --category ict --symbol BTC/USDT
    python -m backtest.run_backtest --timeframe 15m
"""
from __future__ import annotations
import argparse
import os
from collections import defaultdict
import yaml
import pandas as pd

from data.fetcher import get_exchange, fetch_ohlcv_history
from strategies.registry import get_all_strategies
from backtest.engine import Backtester, trades_to_df
from backtest.metrics import compute_metrics


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        path = "config.example.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="فقط یک دسته: ict|smc|price_action|classic|range_rtm")
    parser.add_argument("--timeframe", default=None, help="فقط یک تایم‌فریم: 5m|15m|1h|4h|1d")
    parser.add_argument("--symbol", default=None, help="فقط یک نماد، مثلا BTC/USDT")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ex = get_exchange(cfg["exchange"]["id"])
    symbols = [args.symbol] if args.symbol else cfg["exchange"]["symbols"]
    bt_cfg = cfg["backtest"]
    pt_cfg = cfg["paper_trading"]

    bt = Backtester(fee_pct=bt_cfg["fee_pct"], slippage_pct=bt_cfg["slippage_pct"])
    strategies = get_all_strategies()
    if args.category:
        strategies = [s for s in strategies if s.category == args.category]
    if args.timeframe:
        strategies = [s for s in strategies if s.timeframe == args.timeframe]

    groups = defaultdict(list)
    for s in strategies:
        groups[s.timeframe].append(s)

    os.makedirs("results/trades", exist_ok=True)
    summary_rows = []

    for symbol in symbols:
        for timeframe, tf_strategies in groups.items():
            print(f"\n=== واکشی دیتای {symbol} [{timeframe}] از {bt_cfg['start_date']} تا {bt_cfg['end_date']} "
                  f"برای {len(tf_strategies)} استراتژی ===")
            df = fetch_ohlcv_history(ex, symbol, timeframe, bt_cfg["start_date"], bt_cfg["end_date"])
            print(f"تعداد کندل: {len(df)}")
            if df.empty:
                print("دیتای کافی نیست، این ترکیب رد شد.")
                continue

            for strat in tf_strategies:
                if len(df) < strat.min_bars:
                    print(f"  [{strat.name}] دیتای کافی برای min_bars={strat.min_bars} نیست، رد شد.")
                    continue

                trades = bt.run(strat, df, symbol)
                tdf = trades_to_df(trades)
                metrics = compute_metrics(tdf, pt_cfg["initial_balance_usdt"], pt_cfg["risk_per_trade_pct"])

                safe_symbol = symbol.replace("/", "-")
                if not tdf.empty:
                    tdf.to_csv(f"results/trades/{strat.name}__{safe_symbol}.csv", index=False)

                summary_rows.append({
                    "strategy": strat.name,
                    "category": strat.category,
                    "timeframe": timeframe,
                    "symbol": symbol,
                    **metrics,
                })
                print(f"  [{strat.category}] {strat.name} [{timeframe}]: trades={metrics['total_trades']} "
                      f"winrate={metrics['win_rate']}% PF={metrics['profit_factor']} "
                      f"net={metrics['net_return_pct']}%")

    if not summary_rows:
        print("\nهیچ نتیجه‌ای تولید نشد.")
        return

    summary_df = pd.DataFrame(summary_rows).sort_values("net_return_pct", ascending=False)
    summary_df.to_csv("results/backtest_summary.csv", index=False)
    print("\n>>> نتیجه در results/backtest_summary.csv ذخیره شد.")
    print(summary_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
