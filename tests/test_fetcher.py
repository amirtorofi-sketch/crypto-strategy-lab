"""تست‌های واحد برای data.fetcher.drop_unclosed_candle - رفع باگ کندل درحال‌تشکیل."""
import pandas as pd
import pytest

from data.fetcher import drop_unclosed_candle


def make_df(last_open_time):
    idx = pd.date_range(end=last_open_time, periods=5, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": range(5), "high": range(5), "low": range(5), "close": range(5), "volume": range(5)},
        index=idx,
    )


def test_drops_last_candle_when_still_forming():
    # کندل ۱۵ دقیقه‌ای که ۲ دقیقه پیش شروع شده - هنوز بسته نشده
    now = pd.Timestamp.now(tz="UTC")
    last_open = now - pd.Timedelta(minutes=2)
    df = make_df(last_open)
    result = drop_unclosed_candle(df, "15m")
    assert len(result) == 4
    assert result.index[-1] < df.index[-1]


def test_keeps_last_candle_when_fully_closed():
    # کندل ۱۵ دقیقه‌ای که ۲۰ دقیقه پیش شروع شده - قطعاً بسته شده
    now = pd.Timestamp.now(tz="UTC")
    last_open = now - pd.Timedelta(minutes=20)
    df = make_df(last_open)
    result = drop_unclosed_candle(df, "15m")
    assert len(result) == 5


def test_empty_df_returns_empty():
    df = pd.DataFrame()
    result = drop_unclosed_candle(df, "15m")
    assert result.empty


def test_unknown_timeframe_is_noop():
    now = pd.Timestamp.now(tz="UTC")
    df = make_df(now - pd.Timedelta(minutes=1))
    result = drop_unclosed_candle(df, "weird_tf")
    assert len(result) == len(df)
