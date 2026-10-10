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

حالت «همه‌ی استراتژی‌ها روی ۱۵ دقیقه» (۳۶ نماد پرنقدینگی پیش‌فرض؛ --symbol-set all = هر ۱۸۱ نماد):
    python -m backtest.combo_backtest --all-strategies --timeframe 15m
  - همه‌ی استراتژی‌های ثبت‌شده (به‌جز EXCLUDED_STRATEGIES) در هر پنج سشن، هر کدام روی تایم‌فریم داده‌شده
    (کپی استراتژی با timeframe عوض‌شده؛ خود منطق استراتژی دست‌نخورده است ولی پارامترهایش بر حسب «کندل»
    است، پس روی ۱۵ دقیقه عملاً استراتژی دیگری است).
  - پنجره‌ی داده برای هر استراتژی = min_bars + extra_bars_buffer (سقف max_fetch_bars).
  - ترکیب‌های SELECTED_COMBOS دیگر علامت «انتخاب‌شده» نمی‌گیرند (آن انتخاب برای تایم‌فریم اصلی بود).

اجرای موازی در چند job گیت‌هاب (هر job یک تکه از نمادها) و ادغام در پایان:
    python -m backtest.combo_backtest --all-strategies --timeframe 15m --shard 1/6 --trades-out shards/s1.pkl.gz
    python -m backtest.combo_backtest --merge shards/*.pkl.gz --out results/backtest/combos_15m_all
"""
from __future__ import annotations

import argparse
import copy
import glob
import gzip
import os
import pickle
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

from backtest.combos import EXCLUDED_STRATEGIES, SELECTED_COMBOS, SESSIONS, pick_symbols, to_binance, to_ccxt


# ----------------------------------------------------------------------------- داده
def load_klines(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["open_time"])
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")[["open", "high", "low", "close", "volume"]]


RESAMPLE_RULES = {"1h": "1h", "4h": "4h"}      # 15m خودش بدون resample؛ 5m/1d از داده‌ی ۱۵ دقیقه‌ای ساخته نمی‌شود
SUPPORTED_TIMEFRAMES = ("15m", "1h", "4h")


def resample_tf(df15: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """۱۵ دقیقه‌ای -> تایم‌فریم بالاتر؛ فقط کندل‌های کامل (همه‌ی ۱۵ دقیقه‌ای‌هایش موجود) نگه داشته می‌شود."""
    if timeframe == "15m":
        return df15
    if timeframe not in RESAMPLE_RULES:
        raise ValueError(f"تایم‌فریم {timeframe!r} از داده‌ی ۱۵ دقیقه‌ای قابل ساخت نیست؛ مجاز: {sorted(RESAMPLE_RULES)}")
    rule = RESAMPLE_RULES[timeframe]
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


def split_buildable(names: list[str]) -> tuple[list[str], list[str]]:
    """استراتژی‌هایی که تایم‌فریم خودشان از داده‌ی ۱۵ دقیقه‌ای قابل ساخت نیست (۵m، 1d) را جدا می‌کند."""
    ok = [n for n in names if get_by_name(n).timeframe in SUPPORTED_TIMEFRAMES]
    return ok, [n for n in names if n not in ok]


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
    tf_override = job.get("tf_override")
    for sname, allowed in job["allowed"].items():
        strat = get_by_name(sname)
        if strat is None:
            continue
        if tf_override:
            # کپی سطحی: همان کلاس/پارامترها، فقط تایم‌فریم عوض می‌شود (رجیستری دست نمی‌خورد)
            strat = copy.copy(strat)
            strat.timeframe = tf_override
            window = min(strat.min_bars + job["buffer_bars"], job["max_fetch"])
        else:
            window = job["windows"][strat.timeframe]
        tf = strat.timeframe
        if tf not in frames:
            frames[tf] = resample_tf(df15, tf)
        # حالت عادی: همه‌ی سشن‌های مجاز یک استراتژی با هم (یک پوزیشن باز در هر لحظه، مثل ربات واقعی).
        # حالت --all-sessions: هر سشن مستقل شبیه‌سازی می‌شود تا آمار هر سشن قابل مقایسه با بقیه باشد
        # (پوزیشن بازِ یک سشن جلوی سیگنال سشن دیگر را نگیرد).
        groups = [[x] for x in allowed] if job["independent"] else [list(allowed)]
        for grp in groups:
            rows += simulate(strat, frames[tf], to_ccxt(sym), set(grp), set(job["selected"].get(sname, [])),
                             window, job["min_reward_pct"], job["trade_value"],
                             job["fee_pct"], job["slippage_pct"])
    meta["trades"] = len(rows)
    return sym, rows, meta


# ----------------------------------------------------------------------------- CLI
def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        path = "config.example.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_shard(path: str, trades: pd.DataFrame, metas: dict, params: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump({"trades": trades, "metas": metas, "params": params}, f, protocol=4)


def merge_shards(paths: list[str], out_dir: str) -> None:
    """معاملات چند تکه (خروجی --trades-out) را ادغام می‌کند و گزارش کامل را می‌نویسد."""
    files: list[str] = []
    for p in paths:
        files += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    files = sorted(dict.fromkeys(files))
    if not files:
        raise SystemExit("هیچ فایل تکه‌ای برای ادغام پیدا نشد.")
    parts = []
    for fp in files:
        with gzip.open(fp, "rb") as f:
            parts.append(pickle.load(f))
    base = dict(parts[0]["params"])
    for k in ("fee_pct", "slippage_pct", "trade_value", "min_reward_pct", "all_sessions", "timeframe_override"):
        vals = {repr(p["params"].get(k)) for p in parts}
        if len(vals) > 1:
            raise SystemExit(f"تکه‌ها با پارامتر متفاوت اجرا شده‌اند ({k}: {sorted(vals)}) و قابل ادغام نیستند.")
    metas: dict[str, dict] = {}
    frames = []
    for p in parts:
        metas.update(p["metas"])
        if len(p["trades"]):
            frames.append(p["trades"])
    base["symbols_requested"] = sum(p["params"]["symbols_requested"] for p in parts)
    base["symbols_with_data"] = sum(p["params"]["symbols_with_data"] for p in parts)
    base["symbols_missing"] = [m for p in parts for m in p["params"]["symbols_missing"]]
    base["shards"] = len(parts)
    if not frames:
        raise SystemExit("هیچ معامله‌ای در هیچ تکه‌ای ثبت نشد.")
    trades = pd.concat(frames, ignore_index=True)
    print(f"ادغام {len(parts)} تکه: {len(trades):,} معامله از {base['symbols_with_data']} نماد")
    from backtest.combo_report import write_outputs
    write_outputs(trades, metas, base, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="backtest_data")
    ap.add_argument("--out", default="results/backtest/combos")
    ap.add_argument("--symbols", nargs="*", default=None, help="مثلاً BTCUSDT ETHUSDT (خالی = مجموعه‌ی --symbol-set)")
    ap.add_argument("--symbol-set", choices=["liquid", "all"], default="liquid",
                    help="liquid = ۳۶ نماد پرنقدینگی (پیش‌فرض) ، all = هر ۱۸۱ نماد دست‌ترید")
    ap.add_argument("--shard", default=None, help="i/n: فقط تکه‌ی i از n تکه‌ی نمادها (برای اجرای موازی)")
    ap.add_argument("--only", default=None, help="فقط این استراتژی‌ها (جداشده با ویرگول)")
    ap.add_argument("--all-strategies", action="store_true",
                    help="همه‌ی استراتژی‌های ثبت‌شده (به‌جز --exclude) در هر پنج سشن؛ به‌جای SELECTED_COMBOS")
    ap.add_argument("--exclude", default=None,
                    help="استراتژی‌های حذف‌شده در --all-strategies (جداشده با ویرگول؛ پیش‌فرض EXCLUDED_STRATEGIES؛ '' = هیچ‌کدام)")
    ap.add_argument("--timeframe", choices=SUPPORTED_TIMEFRAMES, default=None,
                    help="همه‌ی استراتژی‌ها را روی این تایم‌فریم اجرا کن (به‌جای تایم‌فریم خودشان)")
    ap.add_argument("--all-sessions", action="store_true",
                    help="استراتژی‌های انتخاب‌شده را در همه‌ی سشن‌ها اجرا کن؛ ترکیب‌های انتخاب‌شده علامت می‌خورند")
    ap.add_argument("--fee-pct", type=float, default=None, help="کارمزد هر طرف (٪)؛ پیش‌فرض backtest.fee_pct کانفیگ (۰.۰۴)")
    ap.add_argument("--slippage-pct", type=float, default=0.0, help="اسلیپیج هر طرف (٪)؛ پیش‌فرض ۰ مثل اجرای زنده")
    ap.add_argument("--trades-out", default=None, help="به‌جای نوشتن گزارش، معاملات این اجرا را در این فایل ذخیره کن (برای تکه‌ها)")
    ap.add_argument("--merge", nargs="+", default=None, help="فایل‌های --trades-out را ادغام کن و گزارش بنویس (شبیه‌سازی انجام نمی‌شود)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()

    if a.merge:
        merge_shards(a.merge, a.out)
        return

    cfg = load_config(a.config)
    ex_cfg, pt_cfg, bt_cfg = cfg["exchange"], cfg["paper_trading"], cfg["backtest"]
    fee_pct = a.fee_pct if a.fee_pct is not None else float(bt_cfg["fee_pct"])
    trade_value = float(pt_cfg["trade_value_usdt"])
    min_reward = float(pt_cfg.get("min_reward_pct", 0.0))
    buffer_bars = int(ex_cfg.get("extra_bars_buffer", 50))
    max_fetch = int(ex_cfg.get("max_fetch_bars", 1000))

    only = {x.strip() for x in a.only.split(",") if x.strip()} if a.only else None
    if a.all_strategies:
        excl = EXCLUDED_STRATEGIES if a.exclude is None else {x.strip() for x in a.exclude.split(",") if x.strip()}
        names = [s.name for s in get_all_strategies() if s.name not in excl]
        if only:
            bad = only - set(names)
            if bad:
                raise SystemExit(f"استراتژی ناشناخته یا حذف‌شده: {sorted(bad)}")
            names = [n for n in names if n in only]
        combos: dict[str, list[str]] = {}                  # هیچ ترکیبی «انتخاب‌شده» علامت نمی‌خورد
        allowed = {n: list(SESSIONS) for n in names}
        a.all_sessions = True
    else:
        combos = dict(SELECTED_COMBOS)
        if only:
            bad = only - set(combos)
            if bad:
                raise SystemExit(f"استراتژی خارج از SELECTED_COMBOS: {sorted(bad)} (برای بقیه از --all-strategies استفاده کن)")
            combos = {k: v for k, v in combos.items() if k in only}
        allowed = {k: (list(SESSIONS) if a.all_sessions else list(v)) for k, v in combos.items()}
        names = list(combos)
    skipped: list[str] = []
    if not a.timeframe:                      # تایم‌فریم خودِ استراتژی: ۵m و 1d از داده‌ی ۱۵ دقیقه‌ای ساخته نمی‌شود
        names, skipped = split_buildable(names)
        allowed = {n: allowed[n] for n in names}
        if skipped:
            print(f"⚠️ {len(skipped)} استراتژی رد شد چون تایم‌فریمشان از داده‌ی ۱۵ دقیقه‌ای ساخته نمی‌شود: "
                  f"{', '.join(f'{n} ({get_by_name(n).timeframe})' for n in skipped)}")
    if not names:
        raise SystemExit("هیچ استراتژی‌ای برای اجرا نمانده.")
    windows = {} if a.timeframe else {tf: live_window_size(tf, buffer_bars, max_fetch)
                                      for tf in {get_by_name(s).timeframe for s in names}}

    syms = pick_symbols(a.symbol_set, a.shard, a.symbols)
    jobs, missing = [], []
    for s in syms:
        p = os.path.join(a.data, f"{s}_15m.csv")
        if not os.path.exists(p):
            missing.append(s)
            continue
        jobs.append({"symbol": s, "path": p, "allowed": allowed, "selected": combos, "windows": windows,
                     "tf_override": a.timeframe, "buffer_bars": buffer_bars, "max_fetch": max_fetch,
                     "min_reward_pct": min_reward, "trade_value": trade_value,
                     "fee_pct": fee_pct, "slippage_pct": a.slippage_pct, "independent": bool(a.all_sessions)})
    if not jobs:
        raise SystemExit("هیچ فایل داده‌ای نیست؛ اول python -m backtest.download_klines را اجرا کن.")
    if missing:
        print(f"⚠️ داده‌ی {len(missing)} نماد موجود نبود و رد شد: {' '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")

    print(f"{len(jobs)} نماد × {len(names)} استراتژی | تایم‌فریم: {a.timeframe or 'خود استراتژی'} | سشن‌ها: "
          f"{'همه' if a.all_sessions else 'فقط انتخاب‌شده'} | کارمزد هر طرف {fee_pct}% + اسلیپیج {a.slippage_pct}% | "
          f"پنجره‌ی داده: {windows or 'min_bars+buffer هر استراتژی'} | min_reward_pct={min_reward}")
    all_rows: list[dict] = []
    metas: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_symbol, j): j["symbol"] for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            sym, rows, meta = f.result()
            metas[sym] = meta
            all_rows += rows
            print(f"[{k}/{len(jobs)}] {sym}: {meta.get('trades', 0)} معامله {meta.get('error', '')}", flush=True)
    params = {"fee_pct": fee_pct, "slippage_pct": a.slippage_pct, "trade_value": trade_value,
              "min_reward_pct": min_reward, "windows": windows, "all_sessions": bool(a.all_sessions),
              "timeframe_override": a.timeframe, "strategies": names, "skipped_unbuildable": skipped,
              "symbols_requested": len(syms), "symbols_with_data": len(jobs), "symbols_missing": missing,
              "combos": combos}
    trades = pd.DataFrame(all_rows)
    if a.trades_out:
        _dump_shard(a.trades_out, trades, metas, params)
        print(f"\n✅ {len(trades):,} معامله‌ی این تکه در {a.trades_out} ذخیره شد.")
        return
    if not all_rows:
        raise SystemExit("هیچ معامله‌ای ثبت نشد.")
    from backtest.combo_report import write_outputs
    write_outputs(trades, metas, params, a.out)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
