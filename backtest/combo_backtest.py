"""
بک‌تستر ترکیب‌های «استراتژی × سشن» روی همان نمادهای ربات‌های دست‌ترید.

اصل طراحی: *هیچ منطق استراتژی‌ای بازنویسی نشده*. همان کلاس‌های strategies/ و همان
validate_signal و همان تعریف سشن live/paper_trader.get_trading_session استفاده می‌شوند.
جزئیات اجرای زنده که عیناً رعایت شده:
  - پنجره‌ی داده = آخرین `needed_bars` کندلِ «بسته‌شده»؛ needed_bars مثل live/runner از بیشینه‌ی
    min_bars همه‌ی استراتژی‌های ثبت‌شده‌ی همان تایم‌فریم + extra_bars_buffer (سقف max_fetch_bars)
  - سیگنال فقط وقتی بررسی می‌شود که آن استراتژی روی آن نماد پوزیشن باز نداشته باشد
  - ورود با signal.entry، حجم ثابت trade_value_usdt (۱۰ دلار)، اعتبارسنجی با min_reward_pct کانفیگ
  - ترتیب خروج: اگر در یک کندل هم SL و هم TP لمس شود، SL (مثل paper_trader)
  - کارمزد: fee_pct هر طرف (پیش‌فرض ۰.۰۴٪ مثل live)، بدون اسلیپیج (قابل تنظیم)
  - سشن بر اساس ساعت UTC «لحظه‌ی بسته‌شدن کندل سیگنال» (ران‌ر زنده چند دقیقه بعد از بسته‌شدن کندل
    اجرا می‌شود، پس همان ساعت را می‌بیند)

تفاوت آگاهانه با اجرای زنده: در زنده، پوزیشنِ تازه‌باز‌شده در اجراهای بعدیِ همان ساعت (هر ۱۵ دقیقه) با
«همان کندل سیگنال» چک می‌شود و گاهی بلافاصله استاپ می‌خورد و دوباره باز می‌شود. اینجا خروج فقط از
کندل بعد از ورود چک می‌شود (رفتار درست)، پس هر سیگنال یک‌بار شمرده می‌شود.

در حالت --all-sessions هر سشن مستقل شبیه‌سازی می‌شود (پوزیشن یک سشن جلوی سشن دیگر را نمی‌گیرد)؛
پس اعداد کلیِ آن حالت ممکن است کمی با حالت پیش‌فرض فرق کند. حالت پیش‌فرض مرجع است.

اجرا (از ریشه‌ی ریپو، بعد از python -m backtest.download_klines):
    python -m backtest.combo_backtest
    python -m backtest.combo_backtest --all-sessions        # همین استراتژی‌ها در همه‌ی سشن‌ها (چک سوگیری انتخاب)
    python -m backtest.combo_backtest --symbols BTCUSDT ETHUSDT --only smc_fvg_fill_entry
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

from data.fetcher import TIMEFRAME_SECONDS
from live.paper_trader import get_trading_session
from strategies.base import validate_signal
from strategies.registry import get_all_strategies, get_by_name

from backtest.combos import BOT_SYMBOLS, SELECTED_COMBOS, SESSIONS, to_binance, to_ccxt


# ----------------------------------------------------------------------------- داده
def load_klines(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["open_time"])
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")[["open", "high", "low", "close", "volume"]]


def resample_tf(df15: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """۱۵ دقیقه‌ای -> تایم‌فریم بالاتر؛ فقط کندل‌های کامل (همه‌ی ۱۵ دقیقه‌ای‌هایش موجود) نگه داشته می‌شود."""
    if timeframe == "15m":
        return df15
    rule = {"1h": "1h", "4h": "4h"}[timeframe]
    per = TIMEFRAME_SECONDS[timeframe] // 900
    g = df15.resample(rule)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(), "cnt": g["close"].count(),
    })
    return out[out["cnt"] == per].drop(columns="cnt")


def live_window_size(timeframe: str, buffer_bars: int, max_fetch: int) -> int:
    """همان فرمول live/runner: بیشینه‌ی min_bars همه‌ی استراتژی‌های ثبت‌شده‌ی این تایم‌فریم + بافر."""
    mins = [s.min_bars for s in get_all_strategies() if s.timeframe == timeframe]
    return min(max(mins) + buffer_bars, max_fetch)


# ----------------------------------------------------------------------------- شبیه‌ساز
def simulate(strategy, df: pd.DataFrame, symbol: str, allowed_sessions: set[str], selected_sessions: set[str],
             window_size: int, min_reward_pct: float, trade_value: float,
             fee_pct: float, slippage_pct: float) -> list[dict]:
    """یک استراتژی روی یک نماد؛ یک پوزیشن باز در هر لحظه. کندل i بسته‌شده است و سیگنال روی close آن."""
    tf = strategy.timeframe
    tf_delta = pd.Timedelta(seconds=TIMEFRAME_SECONDS[tf])
    idx = df.index
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    trades: list[dict] = []
    pos = None
    cost_pct = (fee_pct + slippage_pct) * 2
    i = max(strategy.min_bars, 60) - 1       # live: len(df) >= 60 و >= min_bars

    while i < n:
        close_time = idx[i] + tf_delta
        if pos is not None:
            # خروج فقط از کندلِ بعد از ورود (pos["i"] < i)؛ SL قبل از TP
            if pos["side"] == "long":
                hit_tp, hit_sl = hi[i] >= pos["tp"], lo[i] <= pos["sl"]
            else:
                hit_tp, hit_sl = lo[i] <= pos["tp"], hi[i] >= pos["sl"]
            if hit_sl or hit_tp:
                exit_price = pos["sl"] if hit_sl else pos["tp"]
                direction = 1 if pos["side"] == "long" else -1
                raw_pct = direction * (exit_price - pos["entry"]) / pos["entry"] * 100
                net_pct = raw_pct - cost_pct
                pnl = trade_value * net_pct / 100
                trades.append({
                    "strategy": strategy.name, "category": strategy.category, "symbol": symbol,
                    "timeframe": tf, "session": pos["session"],
                    "selected": pos["session"] in selected_sessions,
                    "side": pos["side"], "entry_time": pos["entry_time"].isoformat(),
                    "entry_price": pos["entry"], "stop_loss": pos["sl"], "take_profit": pos["tp"],
                    "exit_time": close_time.isoformat(), "exit_price": exit_price,
                    "exit_reason": "sl" if hit_sl else "tp",
                    "pnl_pct": round(net_pct, 4), "pnl_usdt": round(pnl, 4), "win": bool(pnl > 0),
                    "duration_minutes": round((close_time - pos["entry_time"]).total_seconds() / 60, 1),
                    "reason": pos["reason"],
                })
                pos = None
            i += 1
            continue

        session = get_trading_session(close_time.to_pydatetime())
        if session in allowed_sessions:
            window = df.iloc[max(0, i + 1 - window_size): i + 1]
            try:
                sig = strategy.generate_signal(window)
            except Exception:       # noqa: BLE001  (همان رفتار live: خطا = بدون سیگنال)
                sig = None
            if sig is not None:
                ok, _ = validate_signal(sig, min_reward_pct=min_reward_pct)
                if ok:
                    pos = {"i": i, "side": sig.side, "entry": float(sig.entry), "sl": float(sig.stop_loss),
                           "tp": float(sig.take_profit), "entry_time": close_time, "session": session,
                           "reason": sig.reason}
        i += 1
    return trades


def run_symbol(job: dict):
    """کار یک نماد (در پروسس جدا). خروجی: (نماد, معاملات, متادیتا)."""
    sym = job["symbol"]
    df15 = load_klines(job["path"])
    meta = {"candles_15m": int(len(df15)),
            "first": str(df15.index[0]) if len(df15) else None,
            "last": str(df15.index[-1]) if len(df15) else None}
    if len(df15) < 500:
        meta["error"] = "داده‌ی کافی نیست"
        return sym, [], meta
    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    for sname, allowed in job["allowed"].items():
        strat = get_by_name(sname)
        if strat is None:
            continue
        tf = strat.timeframe
        if tf not in frames:
            frames[tf] = resample_tf(df15, tf)
        # حالت عادی: همه‌ی سشن‌های مجاز یک استراتژی با هم (یک پوزیشن باز در هر لحظه، مثل ربات واقعی).
        # حالت --all-sessions: هر سشن مستقل شبیه‌سازی می‌شود تا آمار هر سشن قابل مقایسه با بقیه باشد
        # (پوزیشن بازِ یک سشن جلوی سیگنال سشن دیگر را نگیرد).
        groups = [[x] for x in allowed] if job["independent"] else [list(allowed)]
        for grp in groups:
            rows += simulate(strat, frames[tf], to_ccxt(sym), set(grp), set(job["selected"][sname]),
                             job["windows"][tf], job["min_reward_pct"], job["trade_value"],
                             job["fee_pct"], job["slippage_pct"])
    meta["trades"] = len(rows)
    return sym, rows, meta


# ----------------------------------------------------------------------------- CLI
def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        path = "config.example.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="backtest_data")
    ap.add_argument("--out", default="results/backtest/combos")
    ap.add_argument("--symbols", nargs="*", default=None, help="مثلاً BTCUSDT ETHUSDT (خالی = همه‌ی نمادهای دست‌ترید)")
    ap.add_argument("--only", default=None, help="فقط این استراتژی‌ها (جداشده با ویرگول)")
    ap.add_argument("--all-sessions", action="store_true",
                    help="استراتژی‌های انتخاب‌شده را در همه‌ی سشن‌ها اجرا کن؛ ترکیب‌های انتخاب‌شده علامت می‌خورند")
    ap.add_argument("--fee-pct", type=float, default=None, help="کارمزد هر طرف (٪)؛ پیش‌فرض backtest.fee_pct کانفیگ (۰.۰۴)")
    ap.add_argument("--slippage-pct", type=float, default=0.0, help="اسلیپیج هر طرف (٪)؛ پیش‌فرض ۰ مثل اجرای زنده")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()

    cfg = load_config(a.config)
    ex_cfg, pt_cfg, bt_cfg = cfg["exchange"], cfg["paper_trading"], cfg["backtest"]
    fee_pct = a.fee_pct if a.fee_pct is not None else float(bt_cfg["fee_pct"])
    trade_value = float(pt_cfg["trade_value_usdt"])
    min_reward = float(pt_cfg.get("min_reward_pct", 0.0))
    buffer_bars = int(ex_cfg.get("extra_bars_buffer", 50))
    max_fetch = int(ex_cfg.get("max_fetch_bars", 1000))

    combos = dict(SELECTED_COMBOS)
    if a.only:
        want = {x.strip() for x in a.only.split(",") if x.strip()}
        bad = want - set(combos)
        if bad:
            raise SystemExit(f"استراتژی خارج از SELECTED_COMBOS: {sorted(bad)}")
        combos = {k: v for k, v in combos.items() if k in want}
    allowed = {k: (list(SESSIONS) if a.all_sessions else list(v)) for k, v in combos.items()}
    windows = {tf: live_window_size(tf, buffer_bars, max_fetch)
               for tf in {get_by_name(s).timeframe for s in combos}}

    syms = [to_binance(s) for s in (a.symbols or BOT_SYMBOLS)]
    jobs, missing = [], []
    for s in syms:
        p = os.path.join(a.data, f"{s}_15m.csv")
        if not os.path.exists(p):
            missing.append(s)
            continue
        jobs.append({"symbol": s, "path": p, "allowed": allowed, "selected": combos, "windows": windows,
                     "min_reward_pct": min_reward, "trade_value": trade_value,
                     "fee_pct": fee_pct, "slippage_pct": a.slippage_pct, "independent": bool(a.all_sessions)})
    if not jobs:
        raise SystemExit("هیچ فایل داده‌ای نیست؛ اول python -m backtest.download_klines را اجرا کن.")
    if missing:
        print(f"⚠️ داده‌ی {len(missing)} نماد موجود نبود و رد شد: {' '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")

    print(f"{len(jobs)} نماد × {len(combos)} استراتژی | سشن‌ها: "
          f"{'همه' if a.all_sessions else 'فقط انتخاب‌شده'} | کارمزد هر طرف {fee_pct}% + اسلیپیج {a.slippage_pct}% | "
          f"پنجره‌ی داده: {windows} | min_reward_pct={min_reward}")
    all_rows: list[dict] = []
    metas: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_symbol, j): j["symbol"] for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            sym, rows, meta = f.result()
            metas[sym] = meta
            all_rows += rows
            print(f"[{k}/{len(jobs)}] {sym}: {meta.get('trades', 0)} معامله {meta.get('error', '')}", flush=True)
    if not all_rows:
        raise SystemExit("هیچ معامله‌ای ثبت نشد.")

    from backtest.combo_report import write_outputs
    trades = pd.DataFrame(all_rows)
    params = {"fee_pct": fee_pct, "slippage_pct": a.slippage_pct, "trade_value": trade_value,
              "min_reward_pct": min_reward, "windows": windows, "all_sessions": bool(a.all_sessions),
              "symbols_requested": len(syms), "symbols_with_data": len(jobs), "symbols_missing": missing,
              "combos": combos}
    write_outputs(trades, metas, params, a.out)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
