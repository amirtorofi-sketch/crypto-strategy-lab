"""
دانلود کندل‌های ۱۵ دقیقه‌ای اسپات بایننس از آرشیو عمومی data.binance.vision (رایگان، بدون API Key)
برای بک‌تستر ترکیب‌های «استراتژی × سشن».

- ماه‌های کامل  -> فایل ZIP ماهانه
- ماه جاری      -> فایل‌های ZIP روزانه
- روز/ماهی که فایلش نبود -> REST عمومی data-api.binance.vision (صفحه‌بندی ۱۰۰۰تایی)
- از ۱ ژانویه ۲۰۲۵ تایم‌استمپ فایل‌های اسپات «میکروثانیه» است، قبلش میلی‌ثانیه؛ هر دو پشتیبانی می‌شود.
- خروجی: backtest_data/<SYMBOL>_15m.csv  (open_time[UTC], open, high, low, close, volume)
- کندل ۱ ساعته از روی همین ۱۵ دقیقه‌ای‌ها ساخته می‌شود (OHLCV یکسان با کندل ۱h خود بایننس).

اجرا (از ریشه‌ی ریپو):
    python -m backtest.download_klines --days 180
    python -m backtest.download_klines --days 180 --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

ARCHIVE = "https://data.binance.vision/data/spot"
REST = "https://data-api.binance.vision/api/v3/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_vol", "trades", "taker_base", "taker_quote", "ignore"]
INTERVAL = "15m"
STEP = pd.Timedelta(minutes=15)
EMPTY = ["open_time", "open", "high", "low", "close", "volume"]


def _to_dt(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series)
    unit = "us" if s.iloc[0] > 1e14 else "ms"   # میکروثانیه ~1.7e15 ، میلی‌ثانیه ~1.7e12
    return pd.to_datetime(s, unit=unit)


def parse_zip_bytes(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open(z.namelist()[0]) as f:
            raw = pd.read_csv(f, header=None, names=COLS, dtype=str)
    if not raw.empty and not str(raw["open_time"].iloc[0]).strip().isdigit():
        raw = raw.iloc[1:]   # بعضی فایل‌ها هدر دارند
    if raw.empty:
        return pd.DataFrame(columns=EMPTY)
    df = pd.DataFrame({"open_time": _to_dt(raw["open_time"])})
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = raw[c].astype(float).values
    return df


def fetch_zip(url: str):
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return parse_zip_bytes(r.content)
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_rest(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    out, cur = [], int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cur < end_ms:
        data = None
        for attempt in range(4):
            try:
                r = requests.get(REST, params={"symbol": symbol, "interval": INTERVAL,
                                               "startTime": cur, "endTime": end_ms, "limit": 1000}, timeout=30)
                if r.status_code == 400:      # نماد وجود ندارد
                    return pd.DataFrame(columns=EMPTY)
                r.raise_for_status()
                data = r.json()
                break
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
        if not data:
            break
        out.extend(data)
        cur = data[-1][0] + 1
        time.sleep(0.2)
    if not out:
        return pd.DataFrame(columns=EMPTY)
    df = pd.DataFrame(out, columns=COLS)
    res = pd.DataFrame({"open_time": pd.to_datetime(df["open_time"], unit="ms")})
    for c in ["open", "high", "low", "close", "volume"]:
        res[c] = df[c].astype(float).values
    return res


def month_starts(start: datetime, end: datetime):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def download_symbol(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    parts = []
    first_of_this_month = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    for y, m in month_starts(start, end):
        mstart = datetime(y, m, 1, tzinfo=timezone.utc)
        if mstart >= first_of_this_month:
            d = mstart
            while d.date() < end.date():          # امروز ناقص است، از REST می‌آید
                url = f"{ARCHIVE}/daily/klines/{symbol}/{INTERVAL}/{symbol}-{INTERVAL}-{d:%Y-%m-%d}.zip"
                df = fetch_zip(url)
                if df is None:
                    df = fetch_rest(symbol, d, d + timedelta(days=1))
                parts.append(df)
                d += timedelta(days=1)
        else:
            url = f"{ARCHIVE}/monthly/klines/{symbol}/{INTERVAL}/{symbol}-{INTERVAL}-{y}-{m:02d}.zip"
            df = fetch_zip(url)
            if df is None:
                nxt = datetime(y + (m == 12), (m % 12) + 1, 1, tzinfo=timezone.utc)
                df = fetch_rest(symbol, mstart, nxt)
            parts.append(df)
    last = max((p["open_time"].max() for p in parts if len(p)), default=None)
    tail_from = (last + STEP).to_pydatetime().replace(tzinfo=timezone.utc) if last is not None else start
    parts.append(fetch_rest(symbol, tail_from, end))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=EMPTY)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    lo = pd.Timestamp(start.replace(tzinfo=None))
    return df[df["open_time"] >= lo].reset_index(drop=True)


def gap_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "بدون داده"
    full = pd.date_range(df["open_time"].iloc[0], df["open_time"].iloc[-1], freq="15min")
    missing = full.difference(pd.DatetimeIndex(df["open_time"]))
    if len(missing) == 0:
        return f"{len(df)} کندل، بدون حفره"
    s = pd.Series(missing)
    grp = (s.diff() != pd.Timedelta(minutes=15)).cumsum()
    big = s.groupby(grp).agg(["first", "count"]).sort_values("count", ascending=False).iloc[0]
    return (f"{len(df)} کندل، {len(missing)} کندل گم‌شده ({len(missing)/len(full)*100:.2f}٪)، "
            f"بزرگ‌ترین حفره: {int(big['count'])} کندل از {big['first']}")


def _job(symbol: str, start: datetime, end: datetime, out_dir: str):
    try:
        df = download_symbol(symbol, start, end)
    except Exception as e:  # noqa: BLE001
        return symbol, f"❌ {e}"
    if df.empty:
        return symbol, "⚠️ داده‌ای در بایننس اسپات پیدا نشد (رد شد)"
    df.to_csv(os.path.join(out_dir, f"{symbol}_15m.csv"), index=False)
    return symbol, gap_report(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--symbols", nargs="*", default=None, help="مثلاً BTCUSDT ETHUSDT (خالی = همه‌ی نمادهای دست‌ترید)")
    ap.add_argument("--out", default="backtest_data")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    from backtest.combos import BOT_SYMBOLS, to_binance
    symbols = [to_binance(s) for s in (a.symbols or BOT_SYMBOLS)]
    os.makedirs(a.out, exist_ok=True)
    end = datetime.now(timezone.utc)
    start = (end - timedelta(days=a.days)).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"دانلود {len(symbols)} نماد از {start:%Y-%m-%d} تا {end:%Y-%m-%d %H:%M} UTC")
    ok = 0
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        futs = [ex.submit(_job, s, start, end, a.out) for s in symbols]
        for f in as_completed(futs):
            sym, msg = f.result()
            ok += 0 if msg.startswith(("❌", "⚠️")) else 1
            print(f"[{sym}] {msg}", flush=True)
    print(f"\n{ok} از {len(symbols)} نماد با موفقیت دانلود شد.")


if __name__ == "__main__":
    main()
