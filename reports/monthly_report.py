"""
گزارش ماهانه: عملکرد هر استراتژی را از دفتر پوزیشن‌های فرضی (paper_trader)
خلاصه می‌کند. دو نوع پیام تلگرام می‌فرستد:
۱. یک جدول رتبه‌بندی‌شده کلی (به چت اصلی، بدون تاپیک خاص)
۲. برای هر استراتژی که در این ماه معامله داشته، یک خلاصه کوچک داخل
   همان تاپیک مخصوص آن استراتژی (همان‌جایی که سیگنال‌ها/بسته‌شدن‌ها
   می‌رفتند) - تا با اسکرول همان تاپیک، کل تاریخچه + جمع‌بندی ماه را
   کنار هم ببینی.

همچنین یک فایل CSV کامل در results/reports/ ذخیره می‌کند.

اجرا (مثلا روز اول هر ماه با GitHub Actions):
    python -m reports.monthly_report
    python -m reports.monthly_report --year 2026 --month 7   # ماه دلخواه
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime, timezone
import pandas as pd
import yaml

from live.paper_trader import PaperLedger
from live.telegram_bot import send_telegram_message, get_or_create_topic


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        path = "config.example.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_report(year: int, month: int) -> pd.DataFrame:
    ledger = PaperLedger()
    by_strategy = ledger.month_stats(year, month)

    rows = []
    for strat_name, positions in by_strategy.items():
        wins = [p for p in positions if p.status == "win"]
        losses = [p for p in positions if p.status == "loss"]
        total = len(positions)
        pnl_sum = sum(p.pnl_usdt or 0 for p in positions)
        pnl_pct_sum = sum(p.pnl_pct or 0 for p in positions)
        gross_profit = sum(p.pnl_pct for p in wins) if wins else 0
        gross_loss = abs(sum(p.pnl_pct for p in losses)) if losses else 0
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")
        rows.append({
            "strategy": strat_name,
            "category": positions[0].category,
            "timeframe": positions[0].timeframe,
            "symbol": positions[0].symbol,
            "trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / total * 100, 1) if total else 0,
            "profit_factor": pf,
            "total_pnl_usdt": round(pnl_sum, 2),
            "total_pnl_pct_sum": round(pnl_pct_sum, 2),
        })

    df = pd.DataFrame(rows).sort_values("total_pnl_usdt", ascending=False)
    return df


def send_per_strategy_summaries(tg_cfg: dict, df: pd.DataFrame, year: int, month: int):
    for _, r in df.iterrows():
        text = (
            f"📅 <b>خلاصه ماه {year}-{month:02d}</b> برای این استراتژی\n"
            f"تایم‌فریم: {r['timeframe']} | نماد: {r['symbol']}\n"
            f"تعداد معامله: {int(r['trades'])} | برد: {int(r['wins'])} | باخت: {int(r['losses'])}\n"
            f"وین‌ریت: {r['win_rate_pct']}% | پروفیت‌فکتور: {r['profit_factor']}\n"
            f"سود/زیان خالص: <b>{r['total_pnl_usdt']} USDT</b>"
        )
        thread_id = get_or_create_topic(
            tg_cfg["bot_token"], tg_cfg["chat_id"], r["strategy"],
            use_topics=tg_cfg.get("use_topics", True),
        )
        send_telegram_message(tg_cfg["bot_token"], tg_cfg["chat_id"], text, message_thread_id=thread_id)


def main():
    parser = argparse.ArgumentParser()
    now = datetime.now(timezone.utc)
    parser.add_argument("--year", type=int, default=now.year)
    parser.add_argument("--month", type=int, default=now.month)
    args = parser.parse_args()

    cfg = load_config()
    tg = cfg["telegram"]
    df = build_report(args.year, args.month)

    os.makedirs("results/reports", exist_ok=True)
    out_path = f"results/reports/report_{args.year}_{args.month:02d}.csv"
    df.to_csv(out_path, index=False)

    if df.empty:
        summary_text = f"📊 گزارش {args.year}-{args.month:02d}: هیچ معامله بسته‌شده‌ای ثبت نشد."
    else:
        best = df.iloc[0]
        lines = [f"📊 <b>گزارش ماهانه {args.year}-{args.month:02d} - رتبه‌بندی همه استراتژی‌ها</b>", ""]
        for _, r in df.head(30).iterrows():
            lines.append(
                f"• <code>{r['strategy']}</code> [{r['category']}/{r['timeframe']}]: "
                f"{int(r['trades'])} معامله | وین‌ریت {r['win_rate_pct']}% | "
                f"PF {r['profit_factor']} | سود خالص {r['total_pnl_usdt']} USDT"
            )
        lines.append("")
        lines.append(f"🏆 بهترین استراتژی این ماه: <b>{best['strategy']}</b> با سود {best['total_pnl_usdt']} USDT")
        summary_text = "\n".join(lines)

        # پیام جدا برای هر استراتژی داخل تاپیک خودش
        send_per_strategy_summaries(tg, df, args.year, args.month)

    print(summary_text)
    # خلاصه کلی به چت اصلی (بدون تاپیک خاص) تا رتبه‌بندی کلی همیشه یک‌جا قابل دیدن باشد
    send_telegram_message(tg["bot_token"], tg["chat_id"], summary_text)
    print(f"\nگزارش کامل در {out_path} ذخیره شد.")


if __name__ == "__main__":
    main()
