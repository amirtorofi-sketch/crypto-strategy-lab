"""
آزمون سیگنال (Event Study): «بعد از سیگنال هر استراتژی، قیمت واقعاً در جهت سیگنال می‌رود یا نه؟»

این آزمون عمداً SL و TP را کنار می‌گذارد. برای هر سیگنال، بازده قیمت در جهت سیگنال را در افق‌های ثابت
(۱، ۴، ۱۲، ۲۴ و ۷۲ ساعت بعد) اندازه می‌گیرد و با «میانگین بازار» همان نماد و همان سشن مقایسه می‌کند
تا اثر روند کلی بازار حذف شود (excess). اگر سیگنال‌ها جهت قیمت را پیش‌بینی نکنند، هیچ طراحی SL/TP‌ای نجاتشان
نمی‌دهد؛ اگر کنند، همین آزمون می‌گوید برتری روی چه افقی است و چقدر است (به نقطه‌ی پایه، قبل از کارمزد).

جزئیات (همه مثل اجرای زنده یا صریحاً توضیح داده‌شده):
  - همان کلاس‌های استراتژی، همان validate_signal و همان تعریف سشن؛ پنجره‌ی داده = آخرین needed_bars کندلِ بسته‌شده.
  - تایم‌فریم هر استراتژی = تایم‌فریم خودش (native)؛ ۵ دقیقه‌ای و ۱ روزه از داده‌ی ۱۵ دقیقه‌ای ساخته نمی‌شود و رد می‌شود.
  - قیمت مبدأ = close کندل سیگنال (لحظه‌ی بسته‌شدنش)؛ قیمت آینده از کندل‌های ۱۵ دقیقه‌ای خوانده می‌شود.
  - سیگنال‌های پشت‌سرهمِ هم‌جهت (کندل‌های متوالی) یک رویداد حساب می‌شوند (فقط اولین؛ بدون این‌کار یک ایده
    چند بار شمرده می‌شود).
  - baseline = میانگین بازده آینده‌ی همه‌ی کندل‌های همان نماد، همان «ماه» و همان سشن (بدون شرط سیگنال).
    excess = جهت × (بازده − baseline). baseline ماهانه روند/رژیم همان ماه بازار را حذف می‌کند؛ با baseline یک‌ساله
    رژیم‌های نیمه‌ی اول و دوم بازه (مثلاً ریزش در نیمه‌ی اول و صعود در دوم) به‌اشتباه به حساب مهارت سیگنال می‌افتاد.
    اگر یک سلول (ماه×سشن) کمتر از ۱۰۰ کندل داشته باشد، میانگین همان ماه (همه‌ی سشن‌ها) و در نهایت میانگین کل بازه‌ی
    همان سشن استفاده می‌شود. گزینه‌ی --baseline = month (پیش‌فرض) | week | year (رفتار قدیمی).
    باز هم کمی نگاه‌به‌آینده دارد (میانگین ماه با داده‌ی همان ماه ساخته می‌شود) ولی فقط برای حذف رژیم بازار است.
  - MFE/MAE: بیشترین حرکت موافق و مخالف قیمت (٪) در ۴ و ۲۴ ساعت بعد، برای انتخاب منطقی SL/TP.

اجرا (از ریشه‌ی ریپو، بعد از python -m backtest.download_klines):
    python -m backtest.signal_test                              # ۳۶ نماد پرنقدینگی
    python -m backtest.signal_test --shard 2/6 --events-out shards/ev_2.pkl.gz
    python -m backtest.signal_test --merge "shards/ev_*.pkl.gz" --out results/backtest/signal_test
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.combo_backtest import SUPPORTED_TIMEFRAMES, live_window_size, load_config, load_klines, resample_tf
from backtest.combos import pick_symbols, to_ccxt
from data.fetcher import TIMEFRAME_SECONDS
from live.paper_trader import get_trading_session
from strategies.base import validate_signal
from strategies.registry import get_all_strategies, get_by_name

STEP15 = 900
HORIZONS = {"1h": 4, "4h": 16, "12h": 48, "24h": 96, "72h": 288}     # بر حسب تعداد کندل ۱۵ دقیقه‌ای
H_LABELS = list(HORIZONS)
MFE_WINDOWS = {"4h": 16, "24h": 96}
SESSIONS = ["Asia", "London", "London-NY Overlap", "New York", "Off-hours"]
SESSION_CODE = {s: i for i, s in enumerate(SESSIONS)}


def _session_of_hour(hour: int) -> int:
    from datetime import datetime, timezone
    return SESSION_CODE[get_trading_session(datetime(2026, 1, 1, hour, 0, tzinfo=timezone.utc))]


HOUR_TO_SESSION = np.array([_session_of_hour(h) for h in range(24)], dtype=np.int8)


# ----------------------------------------------------------------------------- آماده‌سازی هر نماد
@dataclass
class SymbolArrays:
    idx_ns: np.ndarray            # زمان باز شدن کندل‌های ۱۵ دقیقه‌ای (ns)
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    ret: dict                     # ret[h] : بازده آینده (bps) از close کندل k تا close کندل k+h ؛ NaN اگر حفره باشد
    base: dict                    # base[h] : برای هر کندل k ، میانگین بازده آینده‌ی «همتا»ها (همان نماد، دوره‌ی baseline و سشن)
    mfe: dict                     # mfe[w] , mae[w] : حرکت موافق/مخالف (٪) در w کندل بعد از کندل k (بر اساس جهت لانگ)
    mae: dict


def _roll_fwd(a: np.ndarray, w: int, fn: str) -> np.ndarray:
    """out[k] = fn(a[k+1 .. k+w])"""
    rev = pd.Series(a[::-1])
    r = getattr(rev.rolling(w), fn)().to_numpy()[::-1]       # r[j] = fn(a[j .. j+w-1])
    out = np.full(len(a), np.nan)
    out[:-1] = r[1:]
    return out


BASELINES = ("month", "week", "year")
MIN_CELL = 100


def _period_ids(t_close_ns: np.ndarray, baseline: str) -> np.ndarray:
    if baseline == "year":
        return np.zeros(len(t_close_ns), dtype=np.int64)
    if baseline == "week":
        return (t_close_ns // (7 * 86400 * 10**9)).astype(np.int64)
    if baseline == "month":
        ts = pd.DatetimeIndex(t_close_ns.astype("datetime64[ns]"))
        return (ts.year * 12 + ts.month).to_numpy(dtype=np.int64)
    raise ValueError(f"baseline نامعتبر: {baseline!r} (مجاز: {BASELINES})")


def _local_baseline(r: np.ndarray, period: np.ndarray, sess: np.ndarray) -> np.ndarray:
    """برای هر کندل: میانگین r در سلول (دوره × سشن) ← اگر کم‌نمونه بود: میانگین دوره ← میانگین کل بازه‌ی همان سشن."""
    df = pd.DataFrame({"p": period, "s": sess, "r": r})
    cell_m = df.groupby(["p", "s"])["r"].transform("mean").to_numpy()
    cell_c = df.groupby(["p", "s"])["r"].transform("count").to_numpy()
    per_m = df.groupby("p")["r"].transform("mean").to_numpy()
    per_c = df.groupby("p")["r"].transform("count").to_numpy()
    sess_m = df.groupby("s")["r"].transform("mean").to_numpy()
    out = np.where(cell_c >= MIN_CELL, cell_m, np.where(per_c >= MIN_CELL, per_m, sess_m))
    return out


def prepare_symbol(df15: pd.DataFrame, baseline: str = "month") -> SymbolArrays:
    idx_ns = df15.index.as_unit("ns").asi8.astype(np.int64)      # pandas ۳ پیش‌فرض µs است؛ همیشه به ns تبدیل می‌کنیم
    close = df15["close"].to_numpy(float)
    high = df15["high"].to_numpy(float)
    low = df15["low"].to_numpy(float)
    n = len(df15)
    hours = ((idx_ns + STEP15 * 10**9) // (3600 * 10**9)) % 24          # ساعت UTC لحظه‌ی بسته‌شدن کندل k
    sess = HOUR_TO_SESSION[hours.astype(int)]
    period = _period_ids(idx_ns + STEP15 * 10**9, baseline)
    ret, base = {}, {}
    for lab, h in HORIZONS.items():
        r = np.full(n, np.nan)
        if n > h:
            ok = (idx_ns[h:] - idx_ns[:-h]) == h * STEP15 * 10**9
            r[:-h] = np.where(ok, (close[h:] / close[:-h] - 1) * 1e4, np.nan)
        ret[lab] = r
        base[lab] = _local_baseline(r, period, sess)
    mfe, mae = {}, {}
    for lab, w in MFE_WINDOWS.items():
        mh = _roll_fwd(high, w, "max")
        ml = _roll_fwd(low, w, "min")
        mfe[lab] = (mh / close - 1) * 100            # برای لانگ: بیشترین صعود
        mae[lab] = (ml / close - 1) * 100            # برای لانگ: بیشترین افت (منفی)
    return SymbolArrays(idx_ns, close, high, low, ret, base, mfe, mae)


# ----------------------------------------------------------------------------- جمع‌آوری رویدادها
def collect_events(strategy, df_tf: pd.DataFrame, arr: SymbolArrays, symbol: str,
                   window_size: int, min_reward_pct: float) -> list[dict]:
    """همه‌ی کندل‌های بسته‌شده را می‌گذراند و برای هر «شروع یک دسته سیگنال هم‌جهت» یک رویداد می‌سازد."""
    tf = strategy.timeframe
    tf_ns = TIMEFRAME_SECONDS[tf] * 10**9
    idx = df_tf.index
    n = len(df_tf)
    events: list[dict] = []
    prev_side = None
    i = max(strategy.min_bars, 60) - 1
    while i < n:
        window = df_tf.iloc[max(0, i + 1 - window_size): i + 1]
        try:
            sig = strategy.generate_signal(window)
        except Exception:                    # noqa: BLE001  (مثل زنده: خطا = بدون سیگنال)
            sig = None
        side = None
        if sig is not None:
            ok, _ = validate_signal(sig, min_reward_pct=min_reward_pct)
            if ok:
                side = 1 if sig.side == "long" else -1
        if side is not None and side != prev_side:
            t_close = idx[i].value + tf_ns
            k = int(np.searchsorted(arr.idx_ns, t_close - STEP15 * 10**9))
            if k < len(arr.idx_ns) and arr.idx_ns[k] == t_close - STEP15 * 10**9:
                sess = int(HOUR_TO_SESSION[(t_close // (3600 * 10**9)) % 24])
                ev = {"strategy": strategy.name, "symbol": symbol, "tf": tf, "side": side,
                      "t": t_close, "session": sess}
                for lab in HORIZONS:
                    ev["r_" + lab] = arr.ret[lab][k]
                    ev["b_" + lab] = arr.base[lab][k]
                for lab in MFE_WINDOWS:
                    up, dn = arr.mfe[lab][k], arr.mae[lab][k]
                    ev["mfe_" + lab] = up if side == 1 else -dn          # موافق (٪)
                    ev["mae_" + lab] = dn if side == 1 else -up          # مخالف (٪، منفی)
                events.append(ev)
        prev_side = side
        i += 1
    return events


def run_symbol(job: dict):
    sym = job["symbol"]
    df15 = load_klines(job["path"])
    meta = {"candles_15m": int(len(df15)), "first": str(df15.index[0]) if len(df15) else None,
            "last": str(df15.index[-1]) if len(df15) else None}
    if len(df15) < 1000:
        meta["error"] = "داده‌ی کافی نیست"
        return sym, [], meta
    arr = prepare_symbol(df15, job.get("baseline", "month"))
    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    for sname in job["strategies"]:
        strat = get_by_name(sname)
        tf = strat.timeframe
        if tf not in frames:
            frames[tf] = resample_tf(df15, tf)
        rows += collect_events(strat, frames[tf], arr, to_ccxt(sym), job["windows"][tf], job["min_reward_pct"])
    meta["events"] = len(rows)
    return sym, rows, meta


# ----------------------------------------------------------------------------- CLI
def _dump(path: str, events: pd.DataFrame, metas: dict, params: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump({"events": events, "metas": metas, "params": params}, f, protocol=4)


def merge(paths: list[str], out_dir: str) -> None:
    files: list[str] = []
    for p in paths:
        files += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    files = sorted(dict.fromkeys(files))
    if not files:
        raise SystemExit("هیچ فایل رویدادی برای ادغام پیدا نشد.")
    parts = []
    for fp in files:
        with gzip.open(fp, "rb") as f:
            parts.append(pickle.load(f))
    base = dict(parts[0]["params"])
    for k in ("min_reward_pct", "fee_pct_a", "fee_pct_b", "baseline"):
        vals = {repr(p["params"].get(k)) for p in parts}
        if len(vals) > 1:
            raise SystemExit(f"تکه‌ها با پارامتر متفاوت اجرا شده‌اند ({k}: {sorted(vals)}).")
    metas: dict = {}
    frames = []
    for p in parts:
        metas.update(p["metas"])
        if len(p["events"]):
            frames.append(p["events"])
    base["symbols_requested"] = sum(p["params"]["symbols_requested"] for p in parts)
    base["symbols_with_data"] = sum(p["params"]["symbols_with_data"] for p in parts)
    base["symbols_missing"] = [m for p in parts for m in p["params"]["symbols_missing"]]
    if not frames:
        raise SystemExit("هیچ رویدادی در هیچ تکه‌ای ثبت نشد.")
    events = pd.concat(frames, ignore_index=True)
    print(f"ادغام {len(parts)} تکه: {len(events):,} رویداد از {base['symbols_with_data']} نماد")
    from backtest.signal_report import write_outputs
    write_outputs(events, metas, base, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="backtest_data")
    ap.add_argument("--out", default="results/backtest/signal_test")
    ap.add_argument("--symbols", nargs="*", default=None, help="مثلاً BTCUSDT ETHUSDT (خالی = مجموعه‌ی --symbol-set)")
    ap.add_argument("--symbol-set", choices=["liquid", "all"], default="liquid", help="liquid = ۳۶ نماد پرنقدینگی (پیش‌فرض)")
    ap.add_argument("--shard", default=None, help="مثل 2/6 : فقط نمادهای تکه‌ی ۲ از ۶")
    ap.add_argument("--only", default=None, help="فقط این استراتژی‌ها (جداشده با ویرگول)")
    ap.add_argument("--events-out", default=None, help="فقط رویدادها را ذخیره کن (برای حالت تکه‌تکه)؛ گزارش را --merge می‌سازد")
    ap.add_argument("--merge", nargs="+", default=None, help="فایل‌های --events-out را ادغام و گزارش را بساز")
    ap.add_argument("--baseline", choices=list(BASELINES), default="month",
                    help="دوره‌ی baseline بازار: month (پیش‌فرض) | week | year (رفتار قبلی)")
    ap.add_argument("--fee-pct-a", type=float, default=0.04, help="کارمزد هر طرف برای مقایسه‌ی اول (٪)")
    ap.add_argument("--fee-pct-b", type=float, default=0.35, help="کارمزد هر طرف برای مقایسه‌ی دوم (٪)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()

    if a.merge:
        merge(a.merge, a.out)
        return

    cfg = load_config(a.config)
    ex_cfg, pt_cfg = cfg["exchange"], cfg["paper_trading"]
    min_reward = float(pt_cfg.get("min_reward_pct", 0.0))
    buffer_bars = int(ex_cfg.get("extra_bars_buffer", 50))
    max_fetch = int(ex_cfg.get("max_fetch_bars", 1000))

    strats = [s for s in get_all_strategies() if s.timeframe in SUPPORTED_TIMEFRAMES]
    skipped = [s.name for s in get_all_strategies() if s.timeframe not in SUPPORTED_TIMEFRAMES]
    names = [s.name for s in strats]
    if a.only:
        want = {x.strip() for x in a.only.split(",") if x.strip()}
        bad = want - set(names)
        if bad:
            raise SystemExit(f"استراتژی ناشناخته یا غیرقابل‌ساخت از داده‌ی ۱۵ دقیقه‌ای: {sorted(bad)}")
        names = [n for n in names if n in want]
    windows = {tf: live_window_size(tf, buffer_bars, max_fetch) for tf in {get_by_name(n).timeframe for n in names}}

    syms = pick_symbols(a.symbol_set, a.shard, a.symbols)
    jobs, missing = [], []
    for s in syms:
        p = os.path.join(a.data, f"{s}_15m.csv")
        if not os.path.exists(p):
            missing.append(s)
            continue
        jobs.append({"symbol": s, "path": p, "strategies": names, "windows": windows, "min_reward_pct": min_reward,
                     "baseline": a.baseline})
    if not jobs:
        raise SystemExit("هیچ فایل داده‌ای نیست؛ اول python -m backtest.download_klines را اجرا کن.")
    if missing:
        print(f"⚠️ داده‌ی {len(missing)} نماد نبود و رد شد: {' '.join(missing[:20])}")
    print(f"{len(jobs)} نماد × {len(names)} استراتژی (تایم‌فریم اصلی) | رد شده (۵m/۱d): {', '.join(skipped) or '—'} | "
          f"افق‌ها: {', '.join(H_LABELS)} | baseline: {a.baseline} | پنجره‌ی داده: {windows}")

    rows: list[dict] = []
    metas: dict = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_symbol, j): j["symbol"] for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            sym, r, meta = f.result()
            metas[sym] = meta
            rows += r
            print(f"[{k}/{len(jobs)}] {sym}: {meta.get('events', 0)} رویداد {meta.get('error', '')}", flush=True)
    if not rows:
        raise SystemExit("هیچ رویدادی ثبت نشد.")
    events = pd.DataFrame(rows)
    for c in events.columns:
        if events[c].dtype == np.float64:
            events[c] = events[c].astype(np.float32)
    params = {"min_reward_pct": min_reward, "windows": windows, "strategies": names, "skipped": skipped,
              "symbols_requested": len(syms), "symbols_with_data": len(jobs), "symbols_missing": missing,
              "fee_pct_a": a.fee_pct_a, "fee_pct_b": a.fee_pct_b, "symbol_set": a.symbol_set, "shard": a.shard,
              "baseline": a.baseline}
    if a.events_out:
        _dump(a.events_out, events, metas, params)
        print(f"رویدادها در {a.events_out} ذخیره شد ({len(events):,} رویداد).")
        return
    from backtest.signal_report import write_outputs
    write_outputs(events, metas, params, a.out)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
