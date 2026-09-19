"""
بررسی (audit) دفتر پوزیشن‌های فرضی (results/live/positions.json) برای پیدا
کردن معاملاتی که با منطق فعلی validate_signal() سازگار نیستند - یعنی از
نسخه‌ی قدیمی‌تر کد (قبل از اضافه‌شدن اعتبارسنجی مرکزی) باقی مانده‌اند.

این معاملات را حذف نمی‌کند (تاریخچه دست‌نخورده می‌ماند)، بلکه:
۱. یک فیلد "legacy_invalid": true/false به هر پوزیشن اضافه می‌کند و در یک
   فایل جدید ذخیره می‌کند (results/live/positions_audited.json) - ledger
   اصلی دست‌نخورده باقی می‌ماند.
۲. یک گزارش خلاصه چاپ می‌کند: چند معامله نامعتبرند، به تفکیک استراتژی، و
   مقایسه‌ی win_rate/PnL قبل و بعد از حذف آن‌ها.

اجرا:
    python -m scripts.audit_ledger
    python -m scripts.audit_ledger --write   # فایل positions_audited.json را هم بنویسد
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict

LEDGER_PATH = "results/live/positions.json"
AUDITED_PATH = "results/live/positions_audited.json"


def is_valid_direction(p: dict) -> bool:
    """همان قانونی که strategies/base.py:validate_signal() چک می‌کند."""
    entry, sl, tp, side = p["entry_price"], p["stop_loss"], p["take_profit"], p["side"]
    if side == "long":
        return sl < entry < tp
    if side == "short":
        return tp < entry < sl
    return False


def summarize(positions: list, label: str):
    closed = [p for p in positions if p.get("status") in ("win", "loss")]
    if not closed:
        print(f"  [{label}] معامله‌ی بسته‌ای نیست.")
        return
    total = len(closed)
    wins = sum(1 for p in closed if (p.get("pnl_usdt") or 0) >= 0)
    total_pnl = sum(p.get("pnl_usdt") or 0 for p in closed)
    print(f"  [{label}] معاملات: {total} | وین‌ریت واقعی: {round(wins/total*100, 2)}% | "
          f"PnL کل: {round(total_pnl, 2)} USDT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                         help="فایل results/live/positions_audited.json را هم بنویسد")
    parser.add_argument("--path", default=LEDGER_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"فایل پیدا نشد: {args.path}")
        return

    with open(args.path, "r", encoding="utf-8") as f:
        positions = json.load(f)

    by_strategy_invalid = defaultdict(int)
    audited = []
    for p in positions:
        p = dict(p)
        # فقط معاملات بسته‌شده قابل بررسی جهت هستند - پوزیشن‌های باز فعلاً TP/SL
        # نهایی‌شده ندارند تا نامعتبر بودنشان به همین شکل بررسی شود.
        if p.get("status") in ("win", "loss"):
            legacy_invalid = not is_valid_direction(p)
        else:
            legacy_invalid = False
        p["legacy_invalid"] = legacy_invalid
        if legacy_invalid:
            by_strategy_invalid[p["strategy"]] += 1
        audited.append(p)

    total_invalid = sum(by_strategy_invalid.values())
    print(f"کل پوزیشن‌ها: {len(positions)}")
    print(f"معاملات نامعتبر (legacy، قبل از validate_signal): {total_invalid}")
    for strat, count in sorted(by_strategy_invalid.items(), key=lambda x: -x[1]):
        print(f"  - {strat}: {count}")

    print("\nمقایسه‌ی آماری قبل/بعد از حذف معاملات legacy:")
    summarize(positions, "همه‌ی معاملات (شامل legacy)")
    clean = [p for p in audited if not p["legacy_invalid"]]
    summarize(clean, "فقط معاملات معتبر (بدون legacy)")

    if args.write:
        with open(AUDITED_PATH, "w", encoding="utf-8") as f:
            json.dump(audited, f, ensure_ascii=False, indent=2)
        print(f"\nفایل با فیلد legacy_invalid ذخیره شد: {AUDITED_PATH}")


if __name__ == "__main__":
    main()
