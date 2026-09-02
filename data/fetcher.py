"""
واکشی دادهٔ کندل (OHLCV) از بایننس با استفاده از ccxt.
برای دیتای عمومی (public market data) نیازی به API key نیست.
"""
from __future__ import annotations
import time
import pandas as pd
import ccxt


def get_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    klass = getattr(ccxt, exchange_id)
    ex = klass({
        "enableRateLimit": True,
    })
    return ex


def fetch_ohlcv_df(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 500,
    since_ms: int | None = None,
) -> pd.DataFrame:
    """یک بچ از کندل‌ها را می‌گیرد و به DataFrame تبدیل می‌کند."""
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def fetch_ohlcv_history(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    برای بک‌تست: از start_date تا end_date با صفحه‌بندی (pagination) دیتا می‌گیرد.
    """
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end_date}T00:00:00Z") if end_date else exchange.milliseconds()

    all_rows = []
    while since < end_ts:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts == since:
            break
        since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.drop_duplicates(subset="timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[df.index <= pd.to_datetime(end_date, utc=True)] if end_date else df
    return df
