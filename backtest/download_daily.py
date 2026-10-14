"""
دانلود کندل‌های «روزانه»ی اسپات بایننس برای آزمون روندپیروی و مومنتوم (رایگان، بدون API Key).

از REST عمومی data-api.binance.vision استفاده می‌کند: هر درخواست تا ۱۰۰۰ کندل روزانه برمی‌گرداند، پس ۵ سال
تاریخچه‌ی هر نماد فقط ۲ درخواست است (چند ثانیه برای همه‌ی نمادها).
- خروجی: backtest_data_daily/<SYMBOL>_1d.csv  با ستون‌های date (UTC) ، open ، high ، low ، close ، volume
- کندل روزِ جاری ناقص است و ذخیره نمی‌شود.
- نمادی که تاریخچه‌اش کوتاه‌تر است (تازه لیست شده) فقط از روز لیست‌شدنش داده دارد؛ تاریخ اولین کندل در گزارش می‌آید.

اجرا (از ریشه‌ی ریپو):
    python -m backtest.download_daily --years 5
    python -m backtest.download_daily --years 5 --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

REST = "https://data-api.binance.vision/api/v3/klines"
DAY_MS = 86_400_000
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_vol", "trades",
        "taker_base", "taker_quote", "ignore"]


def fetch_daily(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """کندل‌های روزانه‌ی کامل بین start و end (UTC)؛ خالی اگر نماد وجود نداشته باشد."""
    out, cur = [], int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cur < end_ms:
        data = None
        for attempt in range(5):
            try:
                r = requests.get(REST, params={"symbol": symbol, "interval": "1d", "startTime": cur,
                                               "endTime": end_ms, "limit": 1000}, timeout=30)
                if r.status_code == 400:               # نماد وجود ندارد
                    return pd.DataFrame()
                r.raise_for_status()
                data = r.json()
                break
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
        if not data:
            break
        out.extend(data)
        cur = data[-1][0] + DAY_MS
        time.sleep(0.15)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=COLS)
    res = pd.DataFrame({"date": pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")})
    for c in ["open", "high", "low", "close", "volume"]:
        res[c] = df[c].astype(float).values
    # فقط روزهای کامل (کندلی که تا ۲۴ ساعت بعد از بازشدنش بسته شده)
    done = (pd.to_datetime(df["open_time"], unit="ms", utc=True) + pd.Timedelta(days=1)) <= pd.Timestamp.now(tz="UTC")
    return res[done.values].drop_duplicates("date").reset_index(drop=True)


def gap_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "بدون داده"
    d = pd.to_datetime(df["date"])
    full = pd.date_range(d.iloc[0], d.iloc[-1], freq="D")
    miss = len(full) - len(df)
    return f"{len(df)} روز از {df['date'].iloc[0]} تا {df['date'].iloc[-1]}" + (f"، {miss} روز گم‌شده" if miss else "، بدون حفره")


def _job(symbol: str, start: datetime, end: datetime, out_dir: str):
    try:
        df = fetch_daily(symbol, start, end)
    except Exception as e:  # noqa: BLE001
        return symbol, f"❌ {e}"
    if df.empty:
        return symbol, "⚠️ داده‌ای در بایننس اسپات پیدا نشد (رد شد)"
    df.to_csv(os.path.join(out_dir, f"{symbol}_1d.csv"), index=False)
    return symbol, gap_report(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--symbols", nargs="*", default=None, help="مثلاً BTCUSDT ETHUSDT (خالی = مجموعه‌ی --symbol-set)")
    ap.add_argument("--symbol-set", choices=["liquid", "all"], default="liquid")
    ap.add_argument("--out", default="backtest_data_daily")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    from backtest.combos import pick_symbols
    symbols = pick_symbols(a.symbol_set, None, a.symbols)
    os.makedirs(a.out, exist_ok=True)
    end = datetime.now(timezone.utc)
    start = (end - timedelta(days=int(a.years * 365.25))).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"دانلود کندل روزانه‌ی {len(symbols)} نماد از {start:%Y-%m-%d} تا {end:%Y-%m-%d}")
    ok = 0
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        futs = [ex.submit(_job, s, start, end, a.out) for s in symbols]
        for f in as_completed(futs):
            sym, msg = f.result()
            ok += 0 if msg.startswith(("❌", "⚠️")) else 1
            print(f"[{sym}] {msg}", flush=True)
    print(f"\n{ok} از {len(symbols)} نماد با موفقیت دانلود شد.")
    if ok == 0:
        raise SystemExit("هیچ نمادی دانلود نشد.")


if __name__ == "__main__":
    main()
