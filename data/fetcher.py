"""
واکشی دادهٔ کندل (OHLCV) از بایننس با استفاده از ccxt.
برای دیتای عمومی (public market data) نیازی به API key نیست.
"""
from __future__ import annotations
import time
import pandas as pd
import ccxt

# طول هر تایم‌فریم بر حسب ثانیه - برای تشخیص اینکه آخرین کندل واقعاً بسته
# شده یا هنوز درحال تشکیل است.
TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
    "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800,
}


def drop_unclosed_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    اگر آخرین ردیف df هنوز کاملاً بسته نشده باشد (یعنی زمان فعلی هنوز به
    timestamp_شروع + طول_تایم‌فریم نرسیده)، آن ردیف را حذف می‌کند.

    این رفع همان باگی است که در آن exchange.fetch_ohlcv همیشه آخرین کندلِ
    درحال‌شکل‌گیری بازار را هم برمی‌گرداند، ولی کد قبلی فرض می‌کرد
    df.iloc[-1] همیشه یک کندل بسته‌شده است - هم برای تولید سیگنال و هم
    برای چک کردن TP/SL پوزیشن‌های باز. تایم‌فریم ناشناخته را دست‌نخورده
    برمی‌گرداند (fail-safe، به‌جای فرض غلط).
    """
    if df.empty:
        return df
    tf_seconds = TIMEFRAME_SECONDS.get(timeframe)
    if tf_seconds is None:
        return df
    last_open_time = df.index[-1]
    close_time = last_open_time + pd.Timedelta(seconds=tf_seconds)
    now_utc = pd.Timestamp.now(tz="UTC")
    if now_utc < close_time:
        return df.iloc[:-1]
    return df


def get_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    klass = getattr(ccxt, exchange_id)
    ex = klass({
        "enableRateLimit": True,
    })
    if exchange_id == "binance":
        # این پروژه فقط بازار اسپات (BTC/USDT, ETH/USDT) استفاده می‌کند. بدون
        # این خط، ccxt هنگام load_markets علاوه بر اسپات، به fapi.binance.com
        # (فیوچرز) و dapi.binance.com (inverse) هم سر می‌زند که همان بلاک
        # 451 روی سرورهای GitHub Actions را دارند.
        ex.options["fetchMarkets"] = {"types": ["spot"], "loadAllOptions": False}
        # اندپوینت عمومی اسپات را به mirror رسمی و بدون محدودیت جغرافیایی
        # بایننس هدایت کن (مخصوص همین سناریوی سرورهای ابری/CI ساخته شده).
        ex.urls["api"]["public"] = "https://data-api.binance.vision/api/v3"
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
