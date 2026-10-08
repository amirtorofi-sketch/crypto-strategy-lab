"""
ترکیب‌های «استراتژی × سشن» انتخاب‌شده و فهرست نمادهای ربات‌های دست‌ترید.

- SELECTED_COMBOS: فقط همین ترکیب‌ها بک‌تست می‌شوند (سشن‌ها دقیقاً همان تعریف
  live/paper_trader.get_trading_session هستند و بر اساس ساعت UTC «بسته‌شدن کندل سیگنال»
  تعیین می‌شوند، همان‌طور که در اجرای زنده بر اساس لحظه‌ی ورود تعیین می‌شود).
- BOT_SYMBOLS: اجتماع نمادهای دست‌ترید۱ (signal_bot.py، ۱۳ نماد) و دست‌ترید۲
  (signal_bot_v2.py، ۱۸۰ نماد) = 181 نماد. اگر فهرست آن ربات‌ها عوض شد، اینجا هم عوض کن.
- LIQUID_SYMBOLS: ۳۶ نماد پرنقدینگی (زیرمجموعه‌ی BOT_SYMBOLS) که بک‌تست به‌طور پیش‌فرض روی آن‌ها اجرا می‌شود.
  انتخابش بر اساس شناخت از حجم معاملات معمول جفت‌های USDT بایننس است (نه داده‌ی زنده)؛ هر وقت خواستی همین‌جا عوض کن.
- EXCLUDED_STRATEGIES: سه استراتژی که در بک‌تست ۳۶۵ روزه (بازه‌ی 2025-10-08 تا 2026-10-08)
  در هر پنج سشن Expectancy منفی داشتند و بدترین میانگین Expectancy هر معامله را داشتند؛ در حالت --all-strategies کنار گذاشته می‌شوند.
"""
from __future__ import annotations

SESSIONS = ["Asia", "London", "London-NY Overlap", "New York", "Off-hours"]

SELECTED_COMBOS: dict[str, list[str]] = {
    "pa_inside_bar_breakout": ["London-NY Overlap", "Asia"],
    "classic_macd_signal_cross": ["Asia"],
    "classic_stochastic_cross": ["New York"],
    "ict_london_killzone_breakout": ["London"],
    "smc_bos_continuation": ["Asia"],
    "smc_liquidity_sweep_reversal": ["London-NY Overlap", "London"],
    "smc_equal_highs_lows_grab": ["Asia", "Off-hours", "London"],
    "smc_fvg_fill_entry": ["London-NY Overlap"],
    "range_fade_extremes": ["London-NY Overlap"],
}

BOT_SYMBOLS: list[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "1000SATSUSDT",
    "1INCHUSDT", "AAVEUSDT", "ADAUSDT", "AEROUSDT", "AGLDUSDT", "AIUSDT",
    "ALGOUSDT", "ALICEUSDT", "ALTUSDT", "AMPUSDT", "APEUSDT", "API3USDT",
    "APTUSDT", "ARBUSDT", "ARKMUSDT", "ARKUSDT", "ARPAUSDT", "ARUSDT",
    "ASTERUSDT", "ATOMUSDT", "AVAXUSDT", "AVNTUSDT", "AXLUSDT", "AXSUSDT",
    "BANANAUSDT", "BANDUSDT", "BBUSDT", "BCHUSDT", "BEAMXUSDT", "BERAUSDT",
    "BICOUSDT", "BIGTIMEUSDT", "BLURUSDT", "BOMEUSDT", "BONKUSDT", "CAKEUSDT",
    "CATIUSDT", "CELOUSDT", "CELRUSDT", "CETUSUSDT", "CFXUSDT", "CHRUSDT",
    "CHZUSDT", "CKBUSDT", "COMPUSDT", "CRVUSDT", "CTSIUSDT", "CYBERUSDT",
    "DIAUSDT", "DOGSUSDT", "DOTUSDT", "DUSKUSDT", "DYDXUSDT", "DYMUSDT",
    "EDUUSDT", "EGLDUSDT", "EIGENUSDT", "ENAUSDT", "ENSUSDT", "ERAUSDT",
    "ETCUSDT", "ETHFIUSDT", "FETUSDT", "FIDAUSDT", "FILUSDT", "FLOKIUSDT",
    "FLOWUSDT", "FTTUSDT", "GALAUSDT", "GMTUSDT", "GMXUSDT", "GRAMUSDT",
    "GRTUSDT", "GTCUSDT", "HBARUSDT", "HMSTRUSDT", "ICPUSDT", "IDUSDT",
    "ILVUSDT", "IMXUSDT", "INITUSDT", "INJUSDT", "IOTAUSDT", "IOUSDT",
    "JASMYUSDT", "JOEUSDT", "JTOUSDT", "JUPUSDT", "KERNELUSDT", "KNCUSDT",
    "KSMUSDT", "LAYERUSDT", "LDOUSDT", "LINKUSDT", "LPTUSDT", "LQTYUSDT",
    "LTCUSDT", "LUNAUSDT", "LUNCUSDT", "MAGICUSDT", "MANAUSDT", "MANTAUSDT",
    "MASKUSDT", "MEMEUSDT", "METISUSDT", "MINAUSDT", "MOVRUSDT", "MTLUSDT",
    "NEARUSDT", "NEOUSDT", "NEWTUSDT", "NMRUSDT", "NOTUSDT", "OGNUSDT",
    "ONDOUSDT", "OPUSDT", "ORDIUSDT", "PENDLEUSDT", "PEOPLEUSDT", "PEPEUSDT",
    "PIXELUSDT", "POLUSDT", "POLYXUSDT", "PORTALUSDT", "PROMUSDT", "PUMPUSDT",
    "PYTHUSDT", "RAREUSDT", "RAYUSDT", "RENDERUSDT", "RLCUSDT", "ROSEUSDT",
    "RPLUSDT", "RSRUSDT", "RUNEUSDT", "RVNUSDT", "SAGAUSDT", "SANDUSDT",
    "SEIUSDT", "SHIBUSDT", "SKLUSDT", "SLPUSDT", "SSVUSDT", "STGUSDT",
    "STRKUSDT", "STXUSDT", "SUIUSDT", "SUNUSDT", "SUPERUSDT", "SUSDT",
    "SUSHIUSDT", "SYNUSDT", "TAOUSDT", "THETAUSDT", "TIAUSDT", "TRBUSDT",
    "TRXUSDT", "TURBOUSDT", "UMAUSDT", "UNIUSDT", "VETUSDT", "WIFUSDT",
    "WLDUSDT", "WOOUSDT", "WUSDT", "XLMUSDT", "XRPUSDT", "YGGUSDT",
    "ZECUSDT", "ZKUSDT", "ZROUSDT", "ZRXUSDT", "PAXGUSDT", "EURUSDT",
    "ONEUSDT",
]


def to_ccxt(symbol: str) -> str:
    """BTCUSDT -> BTC/USDT (فرمت نمادهای این ریپو)."""
    if "/" in symbol:
        return symbol
    return f"{symbol[:-4]}/USDT" if symbol.endswith("USDT") else symbol


def to_binance(symbol: str) -> str:
    """BTC/USDT -> BTCUSDT (فرمت فایل‌های دانلودشده)."""
    return symbol.replace("/", "")


# ----------------------------------------------------------------------------- نمادهای پرنقدینگی
LIQUID_SYMBOLS: list[str] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "UNIUSDT", "ATOMUSDT",
    "SUIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "FILUSDT", "ETCUSDT", "AAVEUSDT",
    "HBARUSDT", "ICPUSDT", "SHIBUSDT", "PEPEUSDT", "TAOUSDT", "ENAUSDT", "WLDUSDT", "RENDERUSDT",
    "ONDOUSDT", "SEIUSDT", "TIAUSDT", "FETUSDT",
]

# بدترین‌های بک‌تست ۳۶۵ روزه (میانگین وزنی Expectancy هر معامله روی هر پنج سشن؛ هر سه در هر پنج سشن منفی بودند):
#   smc_bos_continuation -0.0167$ | pa_inside_bar_breakout -0.0161$ | classic_macd_signal_cross -0.0135$
EXCLUDED_STRATEGIES: set[str] = {"smc_bos_continuation", "pa_inside_bar_breakout", "classic_macd_signal_cross"}


def parse_shard(shard: str | None) -> tuple[int, int] | None:
    """'2/6' -> (2, 6) ؛ شماره از ۱ شروع می‌شود."""
    if not shard:
        return None
    try:
        i, n = (int(x) for x in shard.split("/"))
    except ValueError as e:                       # noqa: PERF203
        raise ValueError(f"فرمت shard باید مثل 2/6 باشد، نه {shard!r}") from e
    if n < 1 or not 1 <= i <= n:
        raise ValueError(f"shard نامعتبر: {shard!r}")
    return i, n


def pick_symbols(symbol_set: str = "liquid", shard: str | None = None, explicit: list[str] | None = None) -> list[str]:
    """
    فهرست نمادهای یک اجرا (فرمت BTCUSDT).
      explicit  : اگر داده شود همان‌ها (به‌جای مجموعه)
      symbol_set: 'liquid' (پیش‌فرض، ۳۶ نماد) یا 'all' (هر ۱۸۱ نماد دست‌ترید)
      shard     : 'i/n' -> نمادهای شماره‌ی i از n تکه (تقسیم یک‌درمیان تا بار تکه‌ها برابر بماند)
    """
    if explicit:
        syms = [to_binance(s) for s in explicit]
    elif symbol_set == "all":
        syms = list(BOT_SYMBOLS)
    elif symbol_set == "liquid":
        syms = list(LIQUID_SYMBOLS)
    else:
        raise ValueError(f"symbol_set نامعتبر: {symbol_set!r}")
    sh = parse_shard(shard)
    if sh:
        i, n = sh
        syms = syms[i - 1::n]
    return syms


if __name__ == "__main__":      # python -m backtest.combos 2/6  -> نمادهای تکه‌ی ۲ از ۶ (برای ورک‌فلو)
    import sys
    print(" ".join(pick_symbols("liquid", sys.argv[1] if len(sys.argv) > 1 else None)))
