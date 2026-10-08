"""
ترکیب‌های «استراتژی × سشن» انتخاب‌شده و فهرست نمادهای ربات‌های دست‌ترید.

- SELECTED_COMBOS: فقط همین ترکیب‌ها بک‌تست می‌شوند (سشن‌ها دقیقاً همان تعریف
  live/paper_trader.get_trading_session هستند و بر اساس ساعت UTC «بسته‌شدن کندل سیگنال»
  تعیین می‌شوند، همان‌طور که در اجرای زنده بر اساس لحظه‌ی ورود تعیین می‌شود).
- BOT_SYMBOLS: اجتماع نمادهای دست‌ترید۱ (signal_bot.py، ۱۳ نماد) و دست‌ترید۲
  (signal_bot_v2.py، ۱۸۰ نماد) = 181 نماد. اگر فهرست آن ربات‌ها عوض شد، اینجا هم عوض کن.
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
