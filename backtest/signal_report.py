"""
آمار و خروجی‌های «آزمون سیگنال» (backtest/signal_test.py) — همه داخل ریپو در results/backtest/signal_test/ :

    summary.md              گزارش خوانا
    dashboard.html          داشبورد تعاملی (فایل را دانلود کن و در مرورگر باز کن)
    by_strategy.csv         هر استراتژی × هر افق
    by_strategy_session.csv استراتژی × سشن × افق
    by_strategy_symbol.csv  استراتژی × نماد × افق
    by_strategy_side.csv    استراتژی × جهت (لانگ/شورت) × افق
    meta.json               پارامترها و وضعیت داده‌ی هر نماد

معنی ستون‌ها (همه به «نقطه‌ی پایه» bps = ۰.۰۱٪ ، قبل از SL و TP):
    mean_dir   میانگین بازده در جهت سیگنال
    mean_exc   mean_dir منهای بازده‌ی «بازار» (baseline همان نماد و سشن) = برتری واقعی سیگنال
    t, p       آزمون t با خطای استاندارد خوشه‌بندی‌شده (بلوک ۱ روزه تا ۱۲ ساعت، ۲ روزه برای ۲۴ ساعت، ۳ روزه برای ۷۲ ساعت)؛
               رویدادهای هم‌زمان روی نمادهای هم‌بسته و افق‌های هم‌پوشان مستقل نیستند. p افق ۷۲ ساعته هنوز کمی خوش‌بینانه است.
    q          اصلاح Benjamini–Hochberg درون همان جدول (برای مقایسه‌ی چندگانه)
    net_a/net_b میانگین mean_dir منهای کارمزد رفت‌وبرگشت برای دو سطح کارمزد
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from backtest.combo_stats import bh_adjust
from backtest.signal_test import H_LABELS, MFE_WINDOWS, SESSIONS

MIN_N_LOW = 30
MIN_N_RELIABLE = 100
# طول بلوک خوشه‌بندی (روز) برای هر افق: افق‌های بلند روی هم می‌افتند، پس رویدادهای چند روز متوالی هم‌بسته‌اند.
# با شبیه‌سازی روی راه‌رفتن تصادفی کالیبره شد (جهت پایدار روندپیرو): نرخ مثبت کاذب ~۵–۷٪ تا ۲۴ ساعت و حدود ۱۰٪ در ۷۲ ساعت.
BLOCK_DAYS = {"1h": 1, "4h": 1, "12h": 1, "24h": 2, "72h": 3}

VERDICT_FA = {
    "tradable": "برتری معنادار و بالاتر از کارمزد",
    "below_cost": "برتری معنادار ولی کمتر از کارمزد",
    "inverse": "معنادار ولی ضد جهت سیگنال",
    "hypothesis": "فرضیه",
    "low_n": "نمونه ناکافی",
}
HALF_FA = {"pos": "هر دو نیمه مثبت", "neg": "هر دو نیمه منفی", "mixed": "ناپایدار", "na": "نامشخص"}

COLS = ["n", "hit", "mean_dir", "mean_exc", "med_dir", "se_exc", "t", "p", "q", "days", "n1", "exc1", "n2", "exc2",
        "stable", "net_a", "net_b", "mfe4", "mae4", "mfe24", "mae24", "verdict"]

LEVELS = [
    ("strategy", ["strategy"]),
    ("strategy_session", ["strategy", "session"]),
    ("strategy_symbol", ["strategy", "symbol"]),
    ("strategy_side", ["strategy", "side"]),
]


def fmt(x, d=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


# ----------------------------------------------------------------------------- آمار
def cluster_mean_test(x: np.ndarray, day: np.ndarray) -> tuple[float, float, float, float, int]:
    """
    میانگین، خطای استاندارد خوشه‌بندی‌شده (روزانه)، t، p (نرمال) و تعداد خوشه.
    var(mean) = G/(G-1) * Σ_روز (Σ باقی‌مانده‌های آن روز)² / n²
    """
    n = len(x)
    mean = float(x.mean())
    _, inv = np.unique(day, return_inverse=True)
    G = int(inv.max()) + 1
    if G < 5 or n < 5:
        return mean, float("nan"), float("nan"), float("nan"), G
    s = np.bincount(inv, weights=x - mean)
    var = (G / (G - 1)) * float((s ** 2).sum()) / n ** 2
    se = math.sqrt(var)
    if se <= 0:
        return mean, se, float("nan"), float("nan"), G
    t = mean / se
    return mean, se, t, math.erfc(abs(t) / math.sqrt(2)), G


def _verdict(n: int, mean_exc: float, q, stable: str, net_a: float) -> str:
    if n < MIN_N_LOW:
        return "low_n"
    if n >= MIN_N_RELIABLE and q is not None and q < 0.05:
        if mean_exc > 0 and stable == "pos":
            return "tradable" if net_a > 0 else "below_cost"
        if mean_exc < 0 and stable == "neg":
            return "inverse"
    return "hypothesis"


def build_table(ev: pd.DataFrame, keys: list[str], fee_a: float, fee_b: float, mid_t: int) -> list[dict]:
    cost_a, cost_b = 2 * fee_a * 100, 2 * fee_b * 100        # کارمزد رفت‌وبرگشت به bps
    rows: list[dict] = []
    for k, g in ev.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        day = (g["t"].to_numpy() // (86400 * 10**9)).astype(np.int64)
        first = (g["t"].to_numpy() < mid_t)
        mf4 = float(np.nanmedian(g["mfe_4h"])) if len(g) else float("nan")
        ma4 = float(np.nanmedian(g["mae_4h"])) if len(g) else float("nan")
        mf24 = float(np.nanmedian(g["mfe_24h"])) if len(g) else float("nan")
        ma24 = float(np.nanmedian(g["mae_24h"])) if len(g) else float("nan")
        for lab in H_LABELS:
            exc = g["exc_" + lab].to_numpy(float)
            dr = g["dir_" + lab].to_numpy(float)
            ok = ~np.isnan(exc)
            n = int(ok.sum())
            if n == 0:
                continue
            e, d, dy, f1 = exc[ok], dr[ok], day[ok], first[ok]
            mean_exc, se, t, p, G = cluster_mean_test(e, dy // BLOCK_DAYS[lab])
            n1, n2 = int(f1.sum()), int((~f1).sum())
            exc1 = float(e[f1].mean()) if n1 else float("nan")
            exc2 = float(e[~f1].mean()) if n2 else float("nan")
            if n1 >= 5 and n2 >= 5:
                stable = "pos" if (exc1 > 0 and exc2 > 0) else ("neg" if (exc1 < 0 and exc2 < 0) else "mixed")
            else:
                stable = "na"
            mean_dir = float(d.mean())
            rows.append({
                "key": list(k) + [lab], "n": n, "hit": float((d > 0).mean()), "mean_dir": mean_dir,
                "mean_exc": mean_exc, "med_dir": float(np.median(d)), "se_exc": se, "t": t,
                "p": None if math.isnan(p) else p, "days": G, "n1": n1, "exc1": exc1, "n2": n2, "exc2": exc2,
                "stable": stable, "net_a": mean_dir - cost_a, "net_b": mean_dir - cost_b,
                "mfe4": mf4, "mae4": ma4, "mfe24": mf24, "mae24": ma24,
            })
    bh_adjust(rows, "p")
    for r in rows:
        r["verdict"] = _verdict(r["n"], r["mean_exc"], r["q"], r["stable"], r["net_a"])
    return rows


def prepare_events(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["session"] = ev["session"].map(lambda c: SESSIONS[int(c)])
    ev["side_name"] = np.where(ev["side"] > 0, "long", "short")
    for lab in H_LABELS:
        r = ev["r_" + lab].to_numpy(float)
        b = ev["b_" + lab].to_numpy(float)
        s = ev["side"].to_numpy(float)
        ev["dir_" + lab] = s * r
        ev["exc_" + lab] = s * (r - b)
    return ev


# ----------------------------------------------------------------------------- گزارش
def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(v) or math.isinf(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _flat(rows: list[dict], key_names: list[str]) -> pd.DataFrame:
    out = []
    for r in rows:
        d = {c: _clean(r.get(c)) for c in COLS}
        for name, val in zip(key_names + ["horizon"], r["key"]):
            d[name] = val
        out.append(d)
    df = pd.DataFrame(out)
    first = key_names + ["horizon"]
    return df[first + [c for c in df.columns if c not in first]].round(4)


def make_findings(tabs: dict, pooled: list[dict], n_events: int, params: dict) -> dict:
    F = {"reliable": [], "hypo": [], "notes": []}
    strat = tabs["strategy"]
    m = len(strat)
    cnt = pd.Series([r["verdict"] for r in strat]).value_counts().to_dict()
    pooled_txt = "، ".join(f"{r['key'][1]}: {r['mean_exc']:+.1f} bps (t={fmt(r['t'])})" for r in pooled)
    F["reliable"].append(f"<b>کل رویدادها:</b> {n_events:,} سیگنال (پس از یک‌دسته‌کردن سیگنال‌های متوالی هم‌جهت) از "
                         f"{len(params['strategies'])} استراتژی و {params['symbols_with_data']} نماد. میانگین برتری نسبت به بازار "
                         f"(excess) به تفکیک افق: {pooled_txt}.")
    pos_sig = [r for r in strat if r["verdict"] in ("tradable", "below_cost")]
    neg_sig = [r for r in strat if r["verdict"] == "inverse"]
    F["reliable"].append(f"<b>استراتژی × افق ({m} آزمون):</b> {len(pos_sig)} مورد با برتری معنادار و پایدار، "
                         f"{cnt.get('tradable', 0)} مورد بالاتر از کارمزد، {len(neg_sig)} مورد معنادار ولی ضد جهت سیگنال، "
                         f"{cnt.get('hypothesis', 0)} فرضیه، {cnt.get('low_n', 0)} نمونه ناکافی.")
    if pos_sig:
        F["reliable"].append("<b>برتری معنادار و پایدار:</b> " + "؛ ".join(
            f"{r['key'][0]} @ {r['key'][1]} (n={r['n']:,}, excess={r['mean_exc']:+.1f} bps, t={fmt(r['t'])}, q={fmt(r['q'], 3)}, "
            f"net بعد از کارمزد کم={r['net_a']:+.1f} bps)" for r in sorted(pos_sig, key=lambda r: -r["mean_exc"])[:6]))
    else:
        F["reliable"].append("<b>برتری معنادار و پایدار:</b> در سطح استراتژی هیچ‌کدام پیدا نشد.")
    if neg_sig:
        F["reliable"].append("<b>ضد جهت سیگنال (معنادار و پایدار):</b> " + "؛ ".join(
            f"{r['key'][0]} @ {r['key'][1]} (n={r['n']:,}, excess={r['mean_exc']:+.1f} bps, t={fmt(r['t'])})"
            for r in sorted(neg_sig, key=lambda r: r["mean_exc"])[:5]) +
            ". یعنی بعد از این سیگنال، قیمت به‌طور میانگین برعکس رفته؛ اما این هم تا آزمون پیش‌رو فقط فرضیه‌ی «معکوس‌کردن» است.")
    cost_a = 2 * params["fee_pct_a"] * 100
    cost_b = 2 * params["fee_pct_b"] * 100
    F["reliable"].append(f"<b>مقیاس هزینه:</b> کارمزد رفت‌وبرگشت {cost_a:.0f} bps (کارمزد {params['fee_pct_a']}٪ هر طرف) و "
                         f"{cost_b:.0f} bps (کارمزد {params['fee_pct_b']}٪ هر طرف). برتری سیگنال باید از این بزرگ‌تر باشد.")
    # لایه‌های پایین‌تر
    for nm, lab in (("strategy_session", "استراتژی × سشن"), ("strategy_symbol", "استراتژی × نماد"), ("strategy_side", "استراتژی × جهت")):
        rows = tabs[nm]
        mm = len(rows)
        pos = [r for r in rows if r["verdict"] in ("tradable", "below_cost")]
        trad = [r for r in rows if r["verdict"] == "tradable"]
        F["hypo"].append(f"<b>{lab}:</b> از {mm:,} آزمون، {len(pos)} با برتری معنادار و پایدار ({len(trad)} بالاتر از کارمزد). "
                         f"با سطح ۵٪ حدود {mm * 0.05:.0f} مورد شانسی انتظار می‌رود؛ q این را اصلاح می‌کند.")
        for r in sorted(pos, key=lambda r: -r["mean_exc"])[:4]:
            F["hypo"].append(f"&nbsp;&nbsp;↳ {' | '.join(map(str, r['key']))}: n={r['n']:,} ، excess={r['mean_exc']:+.1f} bps ، "
                             f"t={fmt(r['t'])} ، q={fmt(r['q'], 3)} ، net={r['net_a']:+.1f} bps ، {VERDICT_FA[r['verdict']]}")
    allrows = [r for rows in tabs.values() for r in rows]
    cand = sorted([r for r in allrows if r["n"] >= MIN_N_RELIABLE and r["mean_exc"] > 0 and r["net_a"] > 0 and r["stable"] == "pos"],
                  key=lambda r: -r["mean_exc"])[:6]
    if cand:
        F["hypo"].append("<b>کاندیداهای آزمون پیش‌رو</b> (n≥۱۰۰ ، excess مثبت ، هر دو نیمه مثبت ، بعد از کارمزد کم مثبت؛ ممکن است q بزرگ باشد): " +
                         "؛ ".join(f"{' | '.join(map(str, r['key']))} (excess {r['mean_exc']:+.1f} bps, q={fmt(r['q'], 2)})" for r in cand))
    F["notes"].append("تفسیر: اگر هیچ استراتژی‌ای در هیچ افقی برتری معنادار و پایدار نشان نداد، یعنی ورودهایش جهت قیمت را "
                      "پیش‌بینی نمی‌کنند و پهن‌تر کردن SL و TP یا تغییر خروج کمکی نمی‌کند. اگر برتری هست، افق و اندازه‌ی آن "
                      "طراحی خروج را تعیین می‌کند.")
    return F


def write_outputs(events: pd.DataFrame, metas: dict, params: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ev = prepare_events(events)
    mid_t = int(ev["t"].quantile(0.5))
    fee_a, fee_b = params["fee_pct_a"], params["fee_pct_b"]
    tabs = {name: build_table(ev, keys, fee_a, fee_b, mid_t) for name, keys in LEVELS}
    ev["ALL"] = "ALL"
    pooled = build_table(ev, ["ALL"], fee_a, fee_b, mid_t)
    span_start, span_end = pd.Timestamp(ev["t"].min(), tz="UTC"), pd.Timestamp(ev["t"].max(), tz="UTC")
    findings = make_findings(tabs, pooled, len(ev), params)

    for name, keys in LEVELS:
        _flat(tabs[name], keys).to_csv(os.path.join(out_dir, f"by_{name}.csv"), index=False)
    meta = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "period_start": str(span_start), "period_end": str(span_end), "half_boundary": str(pd.Timestamp(mid_t, tz="UTC")),
            "events": int(len(ev)), "params": params, "symbols": metas}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)
    _write_md(os.path.join(out_dir, "summary.md"), tabs, pooled, findings, meta, params)
    _write_dashboard(os.path.join(out_dir, "dashboard.html"), tabs, pooled, findings, meta, params)
    print(f"\n✅ نتایج در {out_dir}/ ذخیره شد (summary.md ، dashboard.html ، CSVها)\n")
    print(_md_matrix(tabs["strategy"], "excess"))


def _md_matrix(rows: list[dict], metric: str) -> str:
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["key"][0], {})[r["key"][1]] = r
    head = "| استراتژی | " + " | ".join(H_LABELS) + " |\n|---|" + "--:|" * len(H_LABELS) + "\n"
    lines = []
    for s in sorted(by, key=lambda s: -max((by[s][h]["mean_exc"] for h in H_LABELS if h in by[s]), default=0)):
        cells = []
        for h in H_LABELS:
            r = by[s].get(h)
            if r is None:
                cells.append("—")
                continue
            star = "★" if (r["q"] is not None and r["q"] < 0.05) else ""
            cells.append(f"{r['mean_exc']:+.1f} (t={fmt(r['t'], 1)}){star}")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def _write_md(path: str, tabs: dict, pooled: list[dict], F: dict, meta: dict, params: dict) -> None:
    L = ["# نتیجه‌ی آزمون سیگنال (Event Study)\n"]
    L.append(f"- بازه‌ی رویدادها: {meta['period_start'][:10]} تا {meta['period_end'][:10]} | مرز نیمه‌ها: {meta['half_boundary'][:10]}")
    L.append(f"- نمادها: {params['symbols_with_data']} از {params['symbols_requested']} | استراتژی‌ها: {len(params['strategies'])} "
             f"(رد شده چون از داده‌ی ۱۵ دقیقه‌ای ساخته نمی‌شوند: {', '.join(params['skipped']) or '—'}) | رویدادها: {meta['events']:,}")
    L.append("- افق‌ها: " + "، ".join(H_LABELS) + " بعد از بسته‌شدن کندل سیگنال؛ بازده بر حسب bps (۰.۰۱٪) در جهت سیگنال، منهای میانگین بازار همان نماد و سشن.\n")
    L.append("## یافته‌های کلیدی\n")
    L.append("**قابل اتکا**\n")
    L += [f"- {x}" for x in F["reliable"]]
    L.append("\n**فقط فرضیه / هنوز زوده**\n")
    L += [f"- {x}" for x in F["hypo"]]
    L.append("\n" + "\n".join(F["notes"]) + "\n")
    L.append("## برتری هر استراتژی نسبت به بازار (excess، bps)؛ ★ = q<0.05 (پس از اصلاح مقایسه‌ی چندگانه)\n")
    L.append(_md_matrix(tabs["strategy"], "excess"))
    L.append("\n## نکات\n")
    L.append("- هر رویداد یک ورود است؛ سیگنال‌های پشت‌سرهمِ هم‌جهت یک‌بار شمرده شده‌اند. بازه‌ی افق‌های بلند (۲۴ و ۷۲ ساعته) روی هم می‌افتند؛ "
             "برای همین t با خطای استاندارد خوشه‌بندی‌شده‌ی روزانه حساب شده است.")
    L.append("- baseline (میانگین بازار همان نماد و سشن) روی کل بازه حساب شده و فقط برای حذف روند کلی بازار است.")
    L.append("- کالیبراسیون آزمون روی راه‌رفتن تصادفی: نرخ مثبت کاذب ~۵–۷٪ تا افق ۲۴ ساعت و حدود ۱۰٪ در افق ۷۲ ساعت؛ به نتایج ۷۲ ساعته با احتیاط‌تر نگاه کن.")
    L.append("- «بالاتر از کارمزد» یعنی میانگین بازده در جهت سیگنال بعد از کسر کارمزد رفت‌وبرگشت هنوز مثبت است (بدون SL و TP، با خروج در پایان افق).")
    L.append("- داده‌ی بایننس اسپات است؛ اسپرد و لغزش واقعی Tabdeal جدا باید لحاظ شود.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _rnd(x):
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else round(x, 4)
    if isinstance(x, dict):
        return {k: _rnd(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rnd(v) for v in x]
    return x


def _write_dashboard(path: str, tabs: dict, pooled: list[dict], F: dict, meta: dict, params: dict) -> None:
    def pack(rows):
        return [[r["key"]] + [r.get(c) for c in COLS] for r in rows]

    payload = {
        "cols": COLS, "horizons": H_LABELS,
        "tables": {n: {"keys": dict(LEVELS)[n], "rows": pack(r)} for n, r in tabs.items()},
        "pooled": pack(pooled), "findings": F, "verdict_fa": VERDICT_FA, "half_fa": HALF_FA,
        "meta": {"start": meta["period_start"][:10], "end": meta["period_end"][:10], "mid": meta["half_boundary"][:10],
                 "events": meta["events"], "symbols": params["symbols_with_data"], "strategies": len(params["strategies"]),
                 "skipped": params["skipped"], "cost_a": 2 * params["fee_pct_a"] * 100, "cost_b": 2 * params["fee_pct_b"] * 100,
                 "fee_a": params["fee_pct_a"], "fee_b": params["fee_pct_b"]},
        "names": {"strategy": "استراتژی", "strategy_session": "استراتژی × سشن", "strategy_symbol": "استراتژی × نماد",
                  "strategy_side": "استراتژی × جهت"},
    }
    html = _TEMPLATE.replace("__DATA__", json.dumps(_rnd(payload), ensure_ascii=False, separators=(",", ":")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


_TEMPLATE = r"""<!DOCTYPE html><html lang="fa" dir="rtl"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>آزمون سیگنال</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#f4f5f7;--card:#fff;--t:#1d2330;--s:#667085;--b:#e3e6eb;--pos:#0a8f5b;--neg:#c93b3b;--acc:#3454d1;--soft:#eef0f4}
@media(prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b202b;--t:#e8ebf1;--s:#98a2b3;--b:#2c3342;--pos:#3ecf8e;--neg:#ff6b6b;--acc:#7c93ff;--soft:#242a38}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--t);font-family:Tahoma,system-ui,sans-serif;font-size:14px;line-height:1.75}
.w{max-width:1500px;margin:auto;padding:16px}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 10px}.sub{color:var(--s);font-size:12px}
.card{background:var(--card);border:1px solid var(--b);border-radius:10px;padding:14px;margin-bottom:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}.k{background:var(--card);border:1px solid var(--b);border-radius:10px;padding:10px 14px}
.k b{display:block;font-size:20px;direction:ltr;text-align:right}.k span{color:var(--s);font-size:12px}.pos{color:var(--pos)}.neg{color:var(--neg)}
.tabs,.flt{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}.tabs button,select,input{font:inherit;border:1px solid var(--b);background:var(--card);color:var(--t);border-radius:6px;padding:5px 10px}
.tabs button.on,.seg button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.tw{overflow-x:auto;max-height:600px;overflow-y:auto}table{border-collapse:collapse;width:100%;font-size:12.5px;white-space:nowrap}th,td{padding:6px 8px;border-bottom:1px solid var(--b);text-align:right}
th{position:sticky;top:0;background:var(--card);cursor:pointer;color:var(--s);font-weight:600}td.n{direction:ltr}
.tag{padding:1px 8px;border-radius:10px;font-size:11px}.t1{background:#d8f3e6;color:#065f3d}.t2{background:#fde0e0;color:#8b1f1f}.t3{background:#fff0c9;color:#6a4a00}.t4{background:#e6e8ee;color:#505868}
.cw{position:relative;height:340px}.sum{background:var(--soft);border-radius:8px;padding:6px 10px;margin:6px 0 10px;font-size:13px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:900px){.g2{grid-template-columns:1fr}}
.fl{margin:0;padding:0;list-style:none}.fl li{padding:6px 0;border-bottom:1px dashed var(--b)}.fl li:last-child{border-bottom:0}
.hd-r{border-top:3px solid var(--pos)}.hd-h{border-top:3px solid #d99a00}
.hm td{text-align:center;min-width:112px}.hm td.sg{outline:2px solid var(--acc);outline-offset:-2px}
.seg{display:inline-flex;border:1px solid var(--b);border-radius:8px;overflow:hidden}.seg button{border:0;border-radius:0;background:var(--card);color:var(--t);font:inherit;padding:6px 12px;cursor:pointer}
</style></head><body><div class="w">
<h1>آزمون سیگنال: بعد از سیگنال، قیمت در جهت درست می‌رود؟</h1><div class="sub" id="meta"></div><br>
<div class="kpis" id="kpis"></div>
<div class="g2"><div class="card hd-r"><h2>✅ قابل اتکا</h2><ul class="fl" id="f_rel"></ul></div>
<div class="card hd-h"><h2>⏳ فقط فرضیه — هنوز زوده</h2><ul class="fl" id="f_hyp"></ul><div class="sub" id="f_note"></div></div></div>
<div class="card"><h2>نقشه‌ی برتری هر استراتژی در هر افق</h2>
<div class="flt"><div class="seg" id="hmsel"><button data-m="mean_exc" class="on">برتری نسبت به بازار (excess)</button><button data-m="mean_dir">بازده در جهت سیگنال</button><button data-m="net_a">بعد از کارمزد کم</button><button data-m="net_b">بعد از کارمزد زیاد</button></div></div>
<div class="sub">عدد = میانگین به bps (۰.۰۱٪). سبز = مثبت ، قرمز = منفی ، کادر آبی = q&lt;۰.۰۵ ، کم‌رنگ = n&lt;۳۰. زیر هر عدد: وین‌ریت جهت و n.</div>
<div class="tw" style="max-height:none"><table class="hm" id="hm"></table></div></div>
<div class="card"><h2>منحنی افق: برای یک استراتژی</h2><div class="flt"><select id="cs"></select><span class="sub" id="csn"></span></div><div class="cw"><canvas id="ch"></canvas></div>
<div class="sub">آبی = برتری نسبت به بازار با بازه‌ی ۹۵٪ (خط‌چین) ، نارنجی = بازده در جهت سیگنال ، قرمز نقطه‌چین = کارمزد رفت‌وبرگشت (کم و زیاد). برای سودآور بودن، منحنی نارنجی باید بالاتر از خط کارمزد برود.</div></div>
<div class="card"><h2>جدول‌ها</h2><div class="tabs" id="tabs"></div>
<div class="flt"><span id="fs_w">استراتژی: <select id="fs"></select></span><span id="fy_w">نماد: <select id="fy"></select></span><span id="fe_w">سشن: <select id="fe"></select></span><span id="fd_w">جهت: <select id="fd"></select></span>
افق: <select id="fh"></select> حداقل n: <input type="number" id="minn" value="30" style="width:70px">
وضعیت: <select id="vf"><option value="">همه</option></select>
<input id="q" placeholder="جستجو..."><span class="sub" id="cnt"></span></div>
<div class="sum" id="fsum"></div>
<div class="tw"><table><thead id="th"></thead><tbody id="tb"></tbody></table></div>
<div class="sub" style="margin-top:8px">t و p با خطای استاندارد خوشه‌بندی‌شده‌ی روزانه (معاملات هم‌زمان مستقل نیستند). q: اصلاح Benjamini–Hochberg درون هر جدول. «برتری معنادار» = n≥۱۰۰ و q&lt;۰.۰۵ و excess هر دو نیمه هم‌علامت. ردیف‌های کم‌رنگ n&lt;۳۰ دارند.</div></div>
</div><script>
const D=__DATA__;const $=i=>document.getElementById(i);const M=D.meta,H=D.horizons;
const f=(x,d=2)=>x==null?'—':(+x).toFixed(d),pc=x=>x==null?'—':(x*100).toFixed(1)+'%';
const VC={tradable:'t1',below_cost:'t3',inverse:'t2',hypothesis:'t3',low_n:'t4'};
const T={};Object.keys(D.tables).forEach(k=>{T[k]={keys:D.tables[k].keys,rows:D.tables[k].rows.map(a=>{const o={key:a[0],hz:a[0][a[0].length-1]};D.cols.forEach((c,i)=>o[c]=a[i+1]);return o})}});
const P=D.pooled.map(a=>{const o={key:a[0],hz:a[0][1]};D.cols.forEach((c,i)=>o[c]=a[i+1]);return o});
let lv='strategy',sk='rank',sd=-1,hmm='mean_exc',ec=null;
function stratList(){return[...new Set(T.strategy.rows.map(r=>r.key[0]))].sort()}
$('meta').textContent=`از ${M.start} تا ${M.end} | ${M.events.toLocaleString()} سیگنال | ${M.symbols} نماد | ${M.strategies} استراتژی (تایم‌فریم اصلی) | رد شده: ${M.skipped.join('، ')||'—'} | کارمزد رفت‌وبرگشت ${M.cost_a} و ${M.cost_b} bps | مرز نیمه‌ها: ${M.mid}`;
const kp=P.map(r=>[`برتری کل @ ${r.hz}`,(r.mean_exc>=0?'+':'')+f(r.mean_exc,1)+' bps',r.mean_exc>0?'pos':'neg',`t=${f(r.t,1)} | n=${r.n.toLocaleString()}`]);
$('kpis').innerHTML=kp.map(k=>`<div class="k"><span>${k[0]}</span><b class="${k[2]}">${k[1]}</b><span>${k[3]}</span></div>`).join('');
$('f_rel').innerHTML=D.findings.reliable.map(x=>`<li>${x}</li>`).join('');$('f_hyp').innerHTML=D.findings.hypo.map(x=>`<li>${x}</li>`).join('');$('f_note').innerHTML=D.findings.notes.join('<br>');
function heat(){document.querySelectorAll('#hmsel button').forEach(b=>b.classList.toggle('on',b.dataset.m===hmm));
const rows=T.strategy.rows,S=stratList();let h='<thead><tr><th>استراتژی</th>'+H.map(x=>`<th>${x}</th>`).join('')+'</tr></thead><tbody>';
S.forEach(s=>{h+=`<tr><td style="text-align:right">${s}</td>`;H.forEach(hz=>{const r=rows.find(x=>x.key[0]===s&&x.hz===hz);if(!r){h+='<td>—</td>';return}
const v=r[hmm],a=Math.min(Math.abs(v)/25,1)*0.6+0.05,op=r.n<30?0.45:1,bg=v>=0?`rgba(10,143,91,${a*op})`:`rgba(201,59,59,${a*op})`,sg=(r.q!=null&&r.q<0.05)?'sg':'';
h+=`<td class="${sg}" style="background:${bg}"><b>${v>=0?'+':''}${f(v,1)}</b><div class="sub">${pc(r.hit)} | n=${r.n}</div></td>`});h+='</tr>'});$('hm').innerHTML=h+'</tbody>'}
function chart(){const s=$('cs').value,rows=H.map(hz=>T.strategy.rows.find(x=>x.key[0]===s&&x.hz===hz));
const exc=rows.map(r=>r?r.mean_exc:null),hi=rows.map(r=>r&&r.se_exc!=null?r.mean_exc+1.96*r.se_exc:null),lo=rows.map(r=>r&&r.se_exc!=null?r.mean_exc-1.96*r.se_exc:null),dr=rows.map(r=>r?r.mean_dir:null);
$('csn').textContent=rows.map(r=>r?`${r.hz}: n=${r.n}`:'').join(' | ');if(ec)ec.destroy();const tc=getComputedStyle(document.documentElement).getPropertyValue('--s');
const ds=[{label:'برتری نسبت به بازار',data:exc,borderColor:'#3454d1',borderWidth:2,pointRadius:3},{label:'حد بالا ۹۵٪',data:hi,borderColor:'#3454d1',borderDash:[4,4],borderWidth:1,pointRadius:0},{label:'حد پایین ۹۵٪',data:lo,borderColor:'#3454d1',borderDash:[4,4],borderWidth:1,pointRadius:0},{label:'بازده در جهت سیگنال',data:dr,borderColor:'#e08a00',borderWidth:2,pointRadius:3},{label:'کارمزد کم',data:H.map(()=>M.cost_a),borderColor:'#c93b3b',borderDash:[2,3],borderWidth:1,pointRadius:0},{label:'کارمزد زیاد',data:H.map(()=>M.cost_b),borderColor:'#8b1f1f',borderDash:[2,3],borderWidth:1,pointRadius:0}];
ec=new Chart($('ch'),{type:'line',data:{labels:H,datasets:ds},options:{animation:false,maintainAspectRatio:false,scales:{x:{ticks:{color:tc}},y:{title:{display:true,text:'bps',color:tc},ticks:{color:tc}}},plugins:{legend:{labels:{color:tc}}}}})}
$('cs').innerHTML=stratList().map(s=>`<option>${s}</option>`).join('');$('cs').onchange=chart;
document.querySelectorAll('#hmsel button').forEach(b=>b.onclick=()=>{hmm=b.dataset.m;heat()});
const COLSV=[['key','گروه'],['hz','افق'],['n','n'],['hit','وین‌ریت جهت'],['mean_dir','بازده جهتی bps'],['mean_exc','برتری (excess) bps'],['med_dir','میانه bps'],['t','t'],['p','p'],['q','q'],['exc1','excess نیمه ۱'],['exc2','excess نیمه ۲'],['net_a','بعد از کارمزد کم'],['net_b','بعد از کارمزد زیاد'],['mfe4','MFE 4h ٪'],['mae4','MAE 4h ٪'],['mfe24','MFE 24h ٪'],['mae24','MAE 24h ٪'],['verdict','وضعیت']];
function val(r,k){if(k==='key')return r.key.slice(0,-1).join(' | ');if(k==='hz')return H.indexOf(r.hz);if(k==='verdict')return D.verdict_fa[r.verdict];return r[k]}
function uniq(key){const i=T[lv].keys.indexOf(key);return[...new Set(T[lv].rows.map(r=>r.key[i]))].sort()}
function fillSel(){const t=T[lv];[['fs','strategy','fs_w'],['fy','symbol','fy_w'],['fe','session','fe_w'],['fd','side','fd_w']].forEach(([id,key,w])=>{const has=t.keys.includes(key);$(w).style.display=has?'':'none';if(has){const old=$(id).value;$(id).innerHTML='<option value="">همه</option>'+uniq(key).map(x=>`<option>${x}</option>`).join('');$(id).value=old}else $(id).value=''})}
function render(){const t=T[lv];let R=t.rows.slice();const mn=+$('minn').value||0,vf=$('vf').value,qq=$('q').value.trim(),fh=$('fh').value;
const fl=[['fs','strategy'],['fy','symbol'],['fe','session'],['fd','side']].map(([id,k])=>[t.keys.indexOf(k),$(id).value]);
R=R.filter(r=>r.n>=mn&&(!vf||r.verdict===vf)&&(!fh||r.hz===fh)&&(!qq||r.key.join(' ').includes(qq))&&fl.every(([i,v])=>i<0||!v||r.key[i]===v));
if(sk==='rank')R.sort((a,b)=>(b.mean_exc??-1e9)-(a.mean_exc??-1e9));else R.sort((a,b)=>{const x=val(a,sk),y=val(b,sk);if(x==null)return 1;if(y==null)return -1;return(x>y?1:x<y?-1:0)*sd});
const sn=R.reduce((a,r)=>a+r.n,0);$('cnt').textContent=R.length+' ردیف';$('fsum').innerHTML=`ردیف‌های نمایش‌داده‌شده: ${R.length} | مجموع رویدادها (با هم‌پوشانی افق‌ها) ${sn.toLocaleString()}`;
$('th').innerHTML='<tr>'+COLSV.map(c=>`<th data-k="${c[0]}">${c[1]}</th>`).join('')+'</tr>';
$('tb').innerHTML=R.slice(0,700).map(r=>`<tr${r.n<30?' style="opacity:.55"':''}><td>${r.key.slice(0,-1).join(' | ')}</td><td>${r.hz}</td><td class="n">${r.n}</td><td class="n">${pc(r.hit)}</td><td class="n">${f(r.mean_dir,1)}</td><td class="n ${r.mean_exc>0?'pos':'neg'}">${f(r.mean_exc,1)}</td><td class="n">${f(r.med_dir,1)}</td><td class="n">${f(r.t,2)}</td><td class="n">${r.p==null?'—':r.p<.001?'<.001':f(r.p,3)}</td><td class="n">${r.q==null?'—':r.q<.001?'<.001':f(r.q,3)}</td><td class="n">${f(r.exc1,1)}</td><td class="n">${f(r.exc2,1)}</td><td class="n ${r.net_a>0?'pos':'neg'}">${f(r.net_a,1)}</td><td class="n ${r.net_b>0?'pos':'neg'}">${f(r.net_b,1)}</td><td class="n">${f(r.mfe4,2)}</td><td class="n">${f(r.mae4,2)}</td><td class="n">${f(r.mfe24,2)}</td><td class="n">${f(r.mae24,2)}</td><td><span class="tag ${VC[r.verdict]}">${D.verdict_fa[r.verdict]}</span></td></tr>`).join('');
document.querySelectorAll('th').forEach(h=>h.onclick=()=>{const k=h.dataset.k;if(sk===k)sd=-sd;else{sk=k;sd=-1}render()})}
function setLevel(k){lv=k;sk='rank';document.querySelectorAll('#tabs button').forEach(x=>x.classList.toggle('on',x.dataset.l===k));fillSel();render()}
$('tabs').innerHTML=Object.keys(T).map(k=>`<button data-l="${k}" class="${k===lv?'on':''}">${D.names[k]}</button>`).join('');
document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>setLevel(b.dataset.l));
$('fh').innerHTML='<option value="">همه</option>'+H.map(x=>`<option>${x}</option>`).join('');
$('vf').innerHTML+=Object.keys(D.verdict_fa).map(k=>`<option value="${k}">${D.verdict_fa[k]}</option>`).join('');
['minn','vf','q','fs','fy','fe','fd','fh'].forEach(i=>$(i).oninput=render);
heat();chart();fillSel();render();
</script></body></html>
"""
