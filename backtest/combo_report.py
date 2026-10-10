"""
خروجی‌های بک‌تستر ترکیب‌ها؛ همه داخل خود ریپو (پیش‌فرض results/backtest/combos/) ذخیره می‌شود:

    summary.md                       گزارش خوانا (در گیت‌هاب مستقیم دیده می‌شود)
    dashboard.html                   داشبورد تعاملی (فایل را دانلود کن و در مرورگر باز کن)
    by_session.csv                   هر سشن (جمع همه‌ی استراتژی‌ها)
    by_symbol.csv                    هر نماد (جمع همه‌ی استراتژی‌ها و سشن‌ها)
    by_session_symbol.csv            سشن × نماد
    by_strategy.csv                  هر استراتژی
    by_strategy_session.csv          استراتژی × سشن
    by_strategy_symbol.csv           استراتژی × نماد
    by_strategy_session_symbol.csv.gz  استراتژی × سشن × نماد (سه‌بعدی؛ فقط گروه‌های n>=10)
    by_combo.csv                     ترکیب‌های انتخاب‌شده‌ی استراتژی × سشن (در حالت --all-strategies: همه‌ی ترکیب‌ها)
    trades.csv.gz                    همه‌ی معاملات (حجیم؛ workflow آن را فقط به‌عنوان Artifact نگه می‌دارد)
    meta.json                        پارامترها و وضعیت داده‌ی هر نماد

نکته: بدون --all-sessions فقط ترکیب‌های انتخاب‌شده شبیه‌سازی می‌شوند و جدول‌های سشن/نماد فقط همان‌ها را شامل می‌شود.
برای تحلیل کامل سشن و نماد، بک‌تست را با --all-sessions (در workflow: all_sessions = true) اجرا کن.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from backtest.combo_stats import VERDICT_FA, build_table, group_stats

HALF_FA = {"pos": "هر دو نیمه مثبت", "neg": "هر دو نیمه منفی", "mixed": "ناپایدار", "na": "—"}
SLIP_STRESS_PCT = 0.03          # لغزش فرضی هر طرف برای ستون exp_slip / pf_slip
BIG_MIN_N = 10                  # در جدول‌های بزرگ گروه‌های کوچک‌تر از این حذف می‌شوند

LEVELS = [
    ("session", ["session"]),
    ("symbol", ["symbol"]),
    ("session_symbol", ["session", "symbol"]),
    ("strategy", ["strategy"]),
    ("strategy_session", ["strategy", "session"]),
    ("strategy_symbol", ["strategy", "symbol"]),
    ("strategy_session_symbol", ["strategy", "session", "symbol"]),
]
BIG_LEVELS = {"session_symbol", "strategy_symbol", "strategy_session_symbol"}

COLS = ["n", "wins", "wr", "wr_lo", "wr_hi", "pf", "exp", "exp_gross", "exp_slip", "pf_slip", "med", "net",
        "payoff", "be", "mdd", "streak", "tpd", "days", "t_day", "p", "p_day", "q",
        "n1", "exp1", "pf1", "n2", "exp2", "pf2", "stable", "conf", "verdict"]


def _fmt(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def _flat(rows: list[dict], key_names: list[str]) -> pd.DataFrame:
    out = []
    for r in rows:
        d = {k: v for k, v in r.items() if k != "key"}
        for name, val in zip(key_names, r["key"]):
            d[name] = val
        out.append(d)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    first = [c for c in key_names + ["selected"] if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]].round(4)


def _equity(days: list[str], base: pd.DataFrame, sel: pd.DataFrame, all_sessions: bool, label: str) -> dict:
    def curve(g):
        s = g.groupby(g["exit_dt"].dt.strftime("%Y-%m-%d"))["pnl_usdt"].sum()
        return [round(float(v), 4) for v in s.reindex(days, fill_value=0).cumsum()]

    eq = {f"ALL ({label})": curve(sel)}
    if all_sessions:
        eq["ALL (همه‌ی سشن‌ها)"] = curve(base)
    for ses, g in base.groupby("session"):
        eq[f"سشن: {ses}"] = curve(g)
    for (s, ses), g in base.groupby(["strategy", "session"]):
        eq[f"{s} | {ses}"] = curve(g)
    return eq


def _md_combo_table(rows: list[dict]) -> str:
    head = ("| ترکیب | n | وین‌ریت | بازه ۹۵٪ | PF | Expectancy $ | Exp قبل از کارمزد $ | سود خالص $ | "
            "PF نیمه ۱ | PF نیمه ۲ | q | وضعیت |\n")
    head += "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|\n"
    body = []
    for r in rows:
        name = " \\| ".join(r["key"])
        body.append(
            f"| {name} | {r['n']} | {r['wr']*100:.1f}% | {r['wr_lo']*100:.0f}–{r['wr_hi']*100:.0f}% | "
            f"{_fmt(r['pf'])} | {r['exp']:.4f} | {r['exp_gross']:.4f} | {r['net']:.2f} | {_fmt(r['pf1'])} | "
            f"{_fmt(r['pf2'])} | {_fmt(r['q'], 3)} | {VERDICT_FA[r['verdict']]} |")
    return head + "\n".join(body) + "\n"


def _md_simple_table(rows: list[dict], label: str) -> str:
    head = (f"| {label} | n | وین‌ریت | PF | Expectancy $ | Exp قبل از کارمزد $ | Exp با لغزش $ | سود خالص $ | "
            "PF نیمه ۱ | PF نیمه ۲ | q | وضعیت |\n")
    head += "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|\n"
    body = []
    for r in rows:
        name = " \\| ".join(r["key"])
        body.append(
            f"| {name} | {r['n']} | {r['wr']*100:.1f}% | {_fmt(r['pf'])} | {r['exp']:.4f} | {r['exp_gross']:.4f} | "
            f"{r['exp_slip']:.4f} | {r['net']:.2f} | {_fmt(r['pf1'])} | {_fmt(r['pf2'])} | {_fmt(r['q'], 3)} | "
            f"{VERDICT_FA[r['verdict']]} |")
    return head + "\n".join(body) + "\n"


def write_outputs(trades: pd.DataFrame, metas: dict, params: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    t = trades.copy()
    t["entry_dt"] = pd.to_datetime(t["entry_time"], utc=True)
    t["exit_dt"] = pd.to_datetime(t["exit_time"], utc=True)
    t = t.sort_values("exit_dt").reset_index(drop=True)
    mid = t["entry_dt"].quantile(0.5)
    t["half"] = np.where(t["entry_dt"] < mid, 1, 2)
    span = max((t["exit_dt"].max() - t["entry_dt"].min()).total_seconds() / 86400, 1e-9)

    sel = t[t["selected"]].copy()
    all_sessions = bool(params["all_sessions"])
    sel_map = {(s, ses) for s, ses_list in params["combos"].items() for ses in ses_list}
    has_selected = not sel.empty
    if not has_selected:                       # حالت --all-strategies: هیچ ترکیبی «انتخاب‌شده» نیست؛ همه را بگیر
        sel = t.copy()
        all_sessions = True
    label = f"{len(sel_map)} ترکیب انتخاب‌شده" if has_selected else "همه‌ی ترکیب‌ها"
    params = {**params, "sel_label": label, "has_selected": has_selected,
              "strategy_tf": t.groupby("strategy")["timeframe"].first().to_dict()}
    base = t if all_sessions else sel

    cost_per_trade = params["trade_value"] * 2 * (params["fee_pct"] + params["slippage_pct"]) / 100
    slip_cost = params["trade_value"] * 2 * SLIP_STRESS_PCT / 100

    tables: dict[str, tuple[list[str], list[dict]]] = {}
    for name, keys in LEVELS:
        rows = build_table(base, keys, span, min_n=BIG_MIN_N if name in BIG_LEVELS else 1, slip_cost=slip_cost)
        for r in rows:
            r["exp_gross"] = r["exp"] + cost_per_trade
            if "strategy" in keys and "session" in keys:
                r["selected"] = (r["key"][0], r["key"][1]) in sel_map
            elif keys == ["strategy"]:
                r["selected"] = r["key"][0] in params["combos"]
            else:
                r["selected"] = False
        tables[name] = (keys, rows)

    sel_rows = build_table(sel, ["strategy", "session"], span, slip_cost=slip_cost)
    for r in sel_rows:
        r["exp_gross"] = r["exp"] + cost_per_trade
        r["selected"] = True
    overall = group_stats(sel, span, slip_cost)
    overall["exp_gross"] = overall["exp"] + cost_per_trade
    overall_all = group_stats(base, span, slip_cost)
    overall_all["exp_gross"] = overall_all["exp"] + cost_per_trade

    days = sorted(base["exit_dt"].dt.strftime("%Y-%m-%d").unique())
    equity = _equity(days, base, sel, all_sessions, label)

    # ---- CSV و معاملات
    for name, (keys, rows) in tables.items():
        df = _flat(rows, keys)
        path = os.path.join(out_dir, f"by_{name}.csv")
        if name == "strategy_session_symbol":
            df.to_csv(path + ".gz", index=False, compression="gzip")
        else:
            df.to_csv(path, index=False)
    _flat(sel_rows, ["strategy", "session"]).to_csv(os.path.join(out_dir, "by_combo.csv"), index=False)
    t.drop(columns=["entry_dt", "exit_dt"]).to_csv(os.path.join(out_dir, "trades.csv.gz"), index=False,
                                                    compression="gzip")

    meta = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "period_start": str(t["entry_dt"].min()), "period_end": str(t["exit_dt"].max()),
            "span_days": round(span, 2), "half_boundary": str(mid), "trades_all": int(len(t)),
            "trades_selected": int(len(sel)), "params": params, "symbols": metas}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)

    _write_md(os.path.join(out_dir, "summary.md"), tables, sel_rows, overall, overall_all, meta, params)
    _write_dashboard(os.path.join(out_dir, "dashboard.html"), tables, overall, overall_all, meta, equity, days, label)
    print(f"\n✅ نتایج در {out_dir}/ ذخیره شد (summary.md ، dashboard.html ، CSVها ، trades.csv.gz)\n")
    print(_md_combo_table(sel_rows))


def _write_md(path: str, tables: dict, sel_rows: list[dict], o: dict, oa: dict, meta: dict, params: dict) -> None:
    all_sessions = bool(params["all_sessions"])
    L = []
    L.append("# نتیجه‌ی بک‌تست ترکیب‌های استراتژی × سشن\n")
    L.append(f"- بازه: {meta['period_start'][:10]} تا {meta['period_end'][:10]} ({meta['span_days']:.0f} روز) | "
             f"نیمه‌ی اول/دوم: مرز {meta['half_boundary'][:10]}")
    miss = params["symbols_missing"]
    L.append(f"- نمادها: {params['symbols_with_data']} از {params['symbols_requested']} نماد دست‌ترید داده داشتند"
             + (f" (بدون داده: {', '.join(miss[:10])}{'…' if len(miss) > 10 else ''})" if miss else ""))
    L.append(f"- حجم ثابت {params['trade_value']:g}$ ، کارمزد هر طرف {params['fee_pct']}% ، اسلیپیج هر طرف "
             f"{params['slippage_pct']}% ، min_reward_pct={params['min_reward_pct']}")
    label = params["sel_label"]
    L.append(f"- {label}: **{o['n']}** معامله | وین‌ریت {o['wr']*100:.1f}% "
             f"(سربه‌سر {_fmt(o['be'] and o['be']*100, 1)}%) | PF {_fmt(o['pf'], 3)} | Expectancy {o['exp']:.4f}$ | "
             f"**Expectancy قبل از کارمزد {o['exp_gross']:.4f}$** | سود خالص {o['net']:.2f}$ | MaxDD {o['mdd']:.2f}$")
    if all_sessions and params["has_selected"]:
        L.append(f"- کل معاملات همه‌ی سشن‌ها: **{oa['n']}** | PF {_fmt(oa['pf'], 3)} | Expectancy {oa['exp']:.4f}$ | "
                 f"قبل از کارمزد {oa['exp_gross']:.4f}$ | سود خالص {oa['net']:.2f}$")
    L.append("")
    L.append("> اگر «Expectancy قبل از کارمزد» نزدیک صفر باشد، یعنی سیگنال‌ها خودشان هیچ برتری‌ای ندارند و کل ضرر "
             "همان کارمزد است.\n")
    L.append(f"## {label} (مرتب: PF، Expectancy، وین‌ریت، سود خالص)\n")
    L.append(_md_combo_table(sel_rows))
    rp = [r for r in sel_rows if r["verdict"] == "reliable_pos"]
    rn = [r for r in sel_rows if r["verdict"] == "reliable_neg"]
    L.append("\n## خلاصه\n")
    L.append(f"- **قابل اتکا و مثبت** ({len(rp)}): " + (", ".join(" | ".join(r["key"]) for r in rp) or "هیچ‌کدام"))
    L.append(f"- **قابل اتکا و منفی** ({len(rn)}): " + (", ".join(" | ".join(r["key"]) for r in rn) or "هیچ‌کدام"))
    L.append("- بقیه یا فرضیه‌اند یا نمونه‌شان کم است؛ تصمیم «حذف/نگه‌دار» با خودتان است.\n")

    if all_sessions:
        sess = tables["session"][1]
        L.append("## هر سشن (جمع همه‌ی استراتژی‌ها)\n")
        L.append(_md_simple_table(sorted(sess, key=lambda r: r["key"][0]), "سشن"))
        if params["has_selected"]:
            L.append("\n## چک سوگیری انتخاب: همین استراتژی‌ها در همه‌ی سشن‌ها\n")
            L.append("اگر ترکیب‌های انتخاب‌شده فقط به‌خاطر انتخاب از روی داده‌ی زنده خوب به نظر می‌رسیدند، روی تاریخچه‌ی بلند "
                     "باید از بقیه‌ی سشن‌ها بهتر نباشند. (هر سشن مستقل شبیه‌سازی شده؛ اعداد کلی با حالت پیش‌فرض ممکن است "
                     "کمی فرق کند.)\n")
        else:
            L.append("\n## هر استراتژی در هر سشن\n")
            tfo = params.get("timeframe_override")
            if not tfo and params.get("strategy_tf"):
                by_tf: dict[str, list[str]] = {}
                for sn, tf in sorted(params["strategy_tf"].items()):
                    by_tf.setdefault(tf, []).append(sn)
                L.append("هر استراتژی روی تایم‌فریم اصلی خودش اجرا شده: " +
                         " | ".join(f"**{tf}**: {len(v)} استراتژی" for tf, v in sorted(by_tf.items())) + "\n")
                for tf, v in sorted(by_tf.items()):
                    L.append(f"- {tf}: {', '.join(v)}")
                if params.get("skipped_unbuildable"):
                    L.append(f"- ⚠️ رد شده (تایم‌فریمشان از داده‌ی ۱۵ دقیقه‌ای ساخته نمی‌شود): {', '.join(params['skipped_unbuildable'])}")
                L.append("")
            if tfo:
                L.append(f"همه‌ی استراتژی‌ها روی تایم‌فریم **{tfo}** اجرا شده‌اند (نه تایم‌فریم اصلی خودشان)؛ پارامترهای هر استراتژی "
                         "بر حسب «کندل» است، پس نتیجه‌ی این جدول برای تایم‌فریم اصلی آن استراتژی صادق نیست.\n")
        rows = sorted(tables["strategy_session"][1], key=lambda r: (r["key"][0], r["key"][1]))
        L.append("| استراتژی | سشن | انتخاب‌شده؟ | n | PF | Expectancy $ | PF نیمه ۱ | PF نیمه ۲ | q | وضعیت |\n"
                 "|---|---|---|--:|--:|--:|--:|--:|--:|---|")
        for r in rows:
            L.append(f"| {r['key'][0]} | {r['key'][1]} | {'✅' if r['selected'] else ''} | {r['n']} | {_fmt(r['pf'])} | "
                     f"{r['exp']:.4f} | {_fmt(r['pf1'])} | {_fmt(r['pf2'])} | {_fmt(r['q'], 3)} | "
                     f"{VERDICT_FA[r['verdict']]} |")
        sel_cells = [r for r in rows if r["selected"]]
        oth_cells = [r for r in rows if not r["selected"]]
        if sel_cells and oth_cells:
            def pooled(rs):
                n = sum(r["n"] for r in rs)
                return sum(r["n"] * r["exp"] for r in rs) / n if n else 0.0
            L.append(f"\nمیانگین Expectancy (وزن‌دار با n): ترکیب‌های انتخاب‌شده {pooled(sel_cells):.4f}$ در برابر "
                     f"بقیه‌ی سشن‌ها {pooled(oth_cells):.4f}$ ؛ تعداد خانه‌های مثبت: "
                     f"{sum(r['exp'] > 0 for r in rows)} از {len(rows)}.\n")

    sym = [r for r in tables["symbol"][1] if r["n"] >= 300]
    if sym:
        by_exp = sorted(sym, key=lambda r: -r["exp"])
        L.append("## نمادها (n≥۳۰۰؛ جمع همه‌ی استراتژی‌ها" + ("" if all_sessions else " و سشن‌های انتخاب‌شده") + ")\n")
        L.append(f"نمادهای مثبت: {sum(r['exp'] > 0 for r in sym)} از {len(sym)}. "
                 "با ۱۸۱ نماد، همیشه چندتایی شانسی خوب به نظر می‌رسند؛ به q و ثبات دو نیمه نگاه کن.\n")
        L.append("**۱۰ نماد برتر (Expectancy)**\n")
        L.append(_md_simple_table(by_exp[:10], "نماد"))
        L.append("\n**۱۰ نماد ضعیف‌تر**\n")
        L.append(_md_simple_table(by_exp[-10:][::-1], "نماد"))

    L.append("\n## نکات مهم\n")
    L.append("- وین‌ریت از علامت واقعی PnL بعد از کارمزد است، نه «برخورد به TP».")
    L.append("- q = p-value تصحیح‌شده برای مقایسه‌ی چندگانه (Benjamini–Hochberg) درون همان جدول. p-value پایه از آزمون t روی "
             "«جمع PnL هر روز» ساخته شده، چون معاملات هم‌زمان روی نمادهای هم‌بسته مستقل نیستند.")
    L.append("- «قابل اتکا» یعنی n≥۱۰۰ ، q<۰.۰۵ ، علامت Expectancy در هر دو نیمه‌ی بازه یکی؛ و برای «مثبت» باید با "
             f"لغزش فرضی {SLIP_STRESS_PCT}% هر طرف هم مثبت بماند.")
    L.append("- در جدول‌های بزرگ (سشن×نماد، استراتژی×نماد و سه‌بعدی) هزاران گروه مقایسه شده‌اند؛ هر نتیجه‌ی خوب آن‌ها "
             "فرضیه است نه نتیجه، مگر با ثبات دو نیمه و q کوچک.")
    L.append("- هر سیگنال فقط یک‌بار شمرده می‌شود؛ در اجرای زنده‌ی فعلی، سیگنال‌های ۱ساعته گاهی چند بار پشت‌سرهم باز و "
             "همان لحظه استاپ می‌خورند، پس تعداد معاملات زنده بیشتر و وین‌ریت زنده کمتر از اینجا خواهد بود.")
    L.append("- داده‌ی بایننس اسپات است؛ ممکن است نماد/سشنی در Tabdeal رفتار کمی متفاوت داشته باشد (اسپرد، لغزش).")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _rnd(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: _rnd(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rnd(v) for v in x]
    return x


def _write_dashboard(path: str, tables: dict, o: dict, oa: dict, meta: dict, equity: dict, days: list[str],
                     label: str = "") -> None:
    def pack(rows):
        return [[r["key"], 1 if r.get("selected") else 0] + [r.get(c) for c in COLS] for r in rows]

    payload = {
        "cols": COLS,
        "tables": {k: {"keys": v[0], "rows": pack(v[1])} for k, v in tables.items()},
        "overall": {c: o.get(c) for c in COLS}, "overall_all": {c: oa.get(c) for c in COLS},
        "equity": equity, "days": days, "sel_label": label,
        "meta": {"start": meta["period_start"][:10], "end": meta["period_end"][:10], "span": meta["span_days"],
                 "mid": meta["half_boundary"][:10], "slip": SLIP_STRESS_PCT,
                 "p": {k: v for k, v in meta["params"].items()
                       if k in ("fee_pct", "slippage_pct", "trade_value", "min_reward_pct", "all_sessions",
                                "symbols_with_data", "symbols_requested")}},
        "verdict_fa": VERDICT_FA, "half_fa": HALF_FA,
    }
    html = _TEMPLATE.replace("__DATA__", json.dumps(_rnd(payload), ensure_ascii=False, separators=(",", ":")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


_TEMPLATE = r"""<!DOCTYPE html><html lang="fa" dir="rtl"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>بک‌تست ترکیب‌ها</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#f4f5f7;--card:#fff;--t:#1d2330;--s:#667085;--b:#e3e6eb;--pos:#0a8f5b;--neg:#c93b3b;--acc:#3454d1}
@media(prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b202b;--t:#e8ebf1;--s:#98a2b3;--b:#2c3342;--pos:#3ecf8e;--neg:#ff6b6b;--acc:#7c93ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--t);font-family:Tahoma,system-ui,sans-serif;font-size:14px;line-height:1.7}
.w{max-width:1500px;margin:auto;padding:16px}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 10px}.sub{color:var(--s);font-size:12px}
.card{background:var(--card);border:1px solid var(--b);border-radius:10px;padding:14px;margin-bottom:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}.k{background:var(--card);border:1px solid var(--b);border-radius:10px;padding:10px 14px}
.k b{display:block;font-size:22px;direction:ltr;text-align:right}.k span{color:var(--s);font-size:12px}.pos{color:var(--pos)}.neg{color:var(--neg)}
.tabs,.flt{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}.tabs button,select,input{font:inherit;border:1px solid var(--b);background:var(--card);color:var(--t);border-radius:6px;padding:5px 10px}
.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.tw{overflow-x:auto;max-height:600px;overflow-y:auto}table{border-collapse:collapse;width:100%;font-size:12.5px;white-space:nowrap}th,td{padding:6px 8px;border-bottom:1px solid var(--b);text-align:right}
th{position:sticky;top:0;background:var(--card);cursor:pointer;color:var(--s);font-weight:600}td.n{direction:ltr}
.tag{padding:1px 8px;border-radius:10px;font-size:11px}.t1{background:#d8f3e6;color:#065f3d}.t2{background:#fde0e0;color:#8b1f1f}.t3{background:#fff0c9;color:#6a4a00}.t4{background:#e6e8ee;color:#505868}
.cw{position:relative;height:340px}.sum{background:var(--bg);border-radius:8px;padding:6px 10px;margin:6px 0 10px;font-size:13px}
.hm td{text-align:center;min-width:110px}.hm td.sel{outline:2px solid var(--acc);outline-offset:-2px}
</style></head><body><div class="w">
<h1>بک‌تست ترکیب‌های استراتژی × سشن × نماد</h1><div class="sub" id="meta"></div><br>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>نقشه‌ی PF هر استراتژی در هر سشن</h2><div class="sub">سبز = PF بالای ۱ ، قرمز = زیر ۱ ؛ کادر آبی = ترکیب انتخاب‌شده. عدد زیر هر خانه: Expectancy به دلار.</div>
<div class="tw" style="max-height:none"><table class="hm" id="hm"></table></div></div>
<div class="card"><h2>رتبه‌بندی (PF، Expectancy، وین‌ریت، سود خالص)</h2><div class="tabs" id="tabs"></div>
<div class="flt"><span id="fs_w">استراتژی: <select id="fs"></select></span><span id="fe_w">سشن: <select id="fe"></select></span>
حداقل n: <input type="number" id="minn" value="30" style="width:80px">
وضعیت: <select id="vf"><option value="">همه</option><option value="reliable_pos">قابل اتکا: مثبت</option><option value="reliable_neg">قابل اتکا: منفی</option><option value="hypothesis">فرضیه</option><option value="low_n">نمونه ناکافی</option></select>
ثبات دو نیمه: <select id="stab"><option value="">همه</option><option value="pos">هر دو نیمه مثبت</option><option value="neg">هر دو نیمه منفی</option><option value="mixed">ناپایدار</option></select>
<label><input type="checkbox" id="onlysel"> فقط ترکیب‌های انتخاب‌شده</label>
<input id="q" placeholder="جستجو (نماد، استراتژی...)"><span class="sub" id="cnt"></span></div>
<div class="sum" id="fsum"></div>
<div class="tw"><table><thead id="th"></thead><tbody id="tb"></tbody></table></div>
<div class="sub" style="margin-top:8px">p: آزمون t روی جمع PnL هر روز (معاملات هم‌زمان مستقل نیستند). q: اصلاح Benjamini–Hochberg درون همین جدول. «قابل اتکا» = n≥۱۰۰ و q&lt;۰.۰۵ و Expectancy هر دو نیمه هم‌علامت (برای مثبت: بعد از لغزش فرضی هم مثبت). در جدول‌های بزرگ هزاران گروه مقایسه شده؛ نتیجه‌ی خوب آن‌ها بدون ثبات و q کوچک فقط فرضیه است. «Exp قبل از کارمزد» = Expectancy + کارمزد هر معامله.</div></div>
<div class="card"><h2>منحنی PnL تجمعی روزانه</h2><div class="flt"><select id="es"></select><span id="dd" class="sub"></span></div><div class="cw"><canvas id="eq"></canvas></div></div>
</div><script>
const D=__DATA__;const $=i=>document.getElementById(i);
const NAMES={session:'سشن',symbol:'نماد',session_symbol:'سشن × نماد',strategy:'استراتژی',strategy_session:'استراتژی × سشن',strategy_symbol:'استراتژی × نماد',strategy_session_symbol:'استراتژی × سشن × نماد'};
const CL={low:'کم',mid:'متوسط',high:'زیاد'};let lv='strategy_session',sk='rank',sd=-1;
const f=(x,d=2)=>x==null?'—':(+x).toFixed(d),pc=x=>x==null?'—':(x*100).toFixed(1)+'%';
const VC={reliable_pos:'t1',reliable_neg:'t2',hypothesis:'t3',low_n:'t4'};
const T={};Object.keys(D.tables).forEach(k=>{T[k]={keys:D.tables[k].keys,rows:D.tables[k].rows.map(a=>{const o={key:a[0],selected:a[1]===1};D.cols.forEach((c,i)=>o[c]=a[i+2]);return o})}});
const COLS=[['key','گروه'],['n','n'],['conf','اطمینان'],['wr','وین‌ریت'],['ci','بازه ۹۵٪'],['pf','PF'],['exp','Expectancy $'],['exp_gross','Exp قبل از کارمزد'],['exp_slip','Exp با لغزش'],['med','میانه $'],['payoff','Payoff'],['be','سربه‌سر'],['net','سود خالص $'],['mdd','MaxDD $'],['streak','باخت پیاپی'],['tpd','معامله/روز'],['exp1','Exp نیمه ۱'],['exp2','Exp نیمه ۲'],['t_day','t روزانه'],['p_day','p'],['q','q'],['verdict','وضعیت']];
function val(r,k){if(k==='key')return r.key.join(' | ');if(k==='ci')return r.wr_lo;if(k==='conf')return{low:0,mid:1,high:2}[r.conf];if(k==='verdict')return D.verdict_fa[r.verdict];return r[k]}
function rankv(r){return[(r.pf==null?99:r.pf),r.exp,r.wr,r.net]}
function uniq(tbl,key){const i=T[tbl].keys.indexOf(key);return[...new Set(T[tbl].rows.map(r=>r.key[i]))].sort()}
function fillSel(){const t=T[lv],hasS=t.keys.includes('strategy'),hasE=t.keys.includes('session');
$('fs_w').style.display=hasS?'':'none';$('fe_w').style.display=hasE?'':'none';
if(hasS){const cur=$('fs').value;$('fs').innerHTML='<option value="">همه</option>'+uniq(lv,'strategy').map(x=>`<option>${x}</option>`).join('');$('fs').value=cur}
if(hasE){const cur=$('fe').value;$('fe').innerHTML='<option value="">همه</option>'+uniq(lv,'session').map(x=>`<option>${x}</option>`).join('');$('fe').value=cur}}
function render(){const t=T[lv];let R=t.rows.slice();const mn=+$('minn').value||0,vf=$('vf').value,st=$('stab').value,qq=$('q').value.trim(),os=$('onlysel').checked;
const si=t.keys.indexOf('strategy'),ei=t.keys.indexOf('session'),fs=$('fs').value,fe=$('fe').value;
R=R.filter(r=>r.n>=mn&&(!vf||r.verdict===vf)&&(!st||r.stable===st)&&(!qq||r.key.join(' ').includes(qq))&&(!os||r.selected)&&(si<0||!fs||r.key[si]===fs)&&(ei<0||!fe||r.key[ei]===fe));
if(sk==='rank')R.sort((a,b)=>{const x=rankv(a),y=rankv(b);for(let i=0;i<4;i++)if(x[i]!==y[i])return y[i]-x[i];return 0});
else R.sort((a,b)=>{const x=val(a,sk),y=val(b,sk);if(x==null)return 1;if(y==null)return -1;return(x>y?1:x<y?-1:0)*sd});
const sn=R.reduce((a,r)=>a+r.n,0),snet=R.reduce((a,r)=>a+r.net,0);
$('cnt').textContent=R.length+' گروه';
$('fsum').innerHTML=`ردیف‌های نمایش‌داده‌شده: ${R.length} گروه | مجموع معاملات ${sn.toLocaleString()} | سود خالص <b class="${snet>=0?'pos':'neg'}">$${snet.toFixed(2)}</b> | Expectancy تجمعی ${sn?(snet/sn).toFixed(4):'—'}$`;
$('th').innerHTML='<tr>'+COLS.map(c=>`<th data-k="${c[0]}">${c[1]}</th>`).join('')+'</tr>';
$('tb').innerHTML=R.slice(0,600).map(r=>`<tr${r.n<30?' style="opacity:.6"':''}><td>${r.key.join(' | ')}${r.selected?' <span class="tag t1">انتخاب‌شده</span>':''}</td><td class="n">${r.n}</td><td>${CL[r.conf]}</td><td class="n">${pc(r.wr)}</td><td class="n">${pc(r.wr_lo)}–${pc(r.wr_hi)}</td><td class="n ${r.pf==null||r.pf>=1?'pos':'neg'}">${r.pf==null?'∞':f(r.pf)}</td><td class="n ${r.exp>0?'pos':'neg'}">${f(r.exp,4)}</td><td class="n">${f(r.exp_gross,4)}</td><td class="n ${r.exp_slip>0?'pos':'neg'}">${f(r.exp_slip,4)}</td><td class="n">${f(r.med,4)}</td><td class="n">${f(r.payoff)}</td><td class="n">${pc(r.be)}</td><td class="n ${r.net>0?'pos':'neg'}">${f(r.net)}</td><td class="n">${f(r.mdd)}</td><td class="n">${r.streak}</td><td class="n">${f(r.tpd,2)}</td><td class="n">${f(r.exp1,4)}</td><td class="n">${f(r.exp2,4)}</td><td class="n">${f(r.t_day,2)}</td><td class="n">${r.p_day==null?'—':r.p_day<.001?'<.001':f(r.p_day,3)}</td><td class="n">${r.q==null?'—':r.q<.001?'<.001':f(r.q,3)}</td><td><span class="tag ${VC[r.verdict]}">${D.verdict_fa[r.verdict]}</span></td></tr>`).join('');
document.querySelectorAll('th').forEach(h=>h.onclick=()=>{const k=h.dataset.k;if(sk===k)sd=-sd;else{sk=k;sd=-1}render()})}
function setLevel(k){lv=k;sk='rank';document.querySelectorAll('#tabs button').forEach(x=>x.classList.toggle('on',x.dataset.l===k));fillSel();render()}
$('tabs').innerHTML=Object.keys(T).map(k=>`<button data-l="${k}" class="${k===lv?'on':''}">${NAMES[k]}</button>`).join('');
document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>setLevel(b.dataset.l));
['minn','vf','stab','q','fs','fe','onlysel'].forEach(i=>$(i).oninput=render);
const O=D.overall,M=D.meta;
$('meta').textContent=`از ${M.start} تا ${M.end} (${M.span.toFixed(0)} روز) | ${M.p.symbols_with_data} از ${M.p.symbols_requested} نماد | حجم ${M.p.trade_value}$ | کارمزد هر طرف ${M.p.fee_pct}% + اسلیپیج ${M.p.slippage_pct}% | مرز نیمه‌ها: ${M.mid} | ${M.p.all_sessions?'همه‌ی سشن‌ها شبیه‌سازی شده':'فقط سشن‌های انتخاب‌شده شبیه‌سازی شده'}`;
const kp=[['معاملات ('+D.sel_label+')',O.n,''],['وین‌ریت واقعی',pc(O.wr),''],['سربه‌سر لازم',pc(O.be),''],['Profit Factor',O.pf==null?'∞':f(O.pf,3),(O.pf==null||O.pf>=1)?'pos':'neg'],['Expectancy / معامله','$'+f(O.exp,4),O.exp>0?'pos':'neg'],['Exp قبل از کارمزد','$'+f(O.exp_gross,4),O.exp_gross>0?'pos':'neg'],['سود خالص','$'+f(O.net),O.net>0?'pos':'neg'],['Max Drawdown','$'+f(O.mdd),'neg']];
$('kpis').innerHTML=kp.map(k=>`<div class="k"><span>${k[0]}</span><b class="${k[2]}">${k[1]}</b></div>`).join('');
(function(){const rows=T.strategy_session.rows,S=[...new Set(rows.map(r=>r.key[0]))].sort(),E=[...new Set(rows.map(r=>r.key[1]))].sort();
let h='<thead><tr><th>استراتژی</th>'+E.map(e=>`<th>${e}</th>`).join('')+'</tr></thead><tbody>';
S.forEach(s=>{h+=`<tr><td style="text-align:right">${s}</td>`;E.forEach(e=>{const r=rows.find(x=>x.key[0]===s&&x.key[1]===e);
if(!r){h+='<td>—</td>';return}const pf=r.pf==null?3:r.pf,a=Math.min(Math.abs(pf-1)/0.25,1)*0.65+0.08;
const bg=pf>=1?`rgba(10,143,91,${a})`:`rgba(201,59,59,${a})`;
h+=`<td class="${r.selected?'sel':''}" style="background:${bg}"><b>${r.pf==null?'∞':f(r.pf)}</b><div class="sub">${f(r.exp,4)} | n=${r.n}</div></td>`});h+='</tr>'});
$('hm').innerHTML=h+'</tbody>'})();
Object.keys(D.equity).forEach(s=>$('es').add(new Option(s,s)));
const tc=getComputedStyle(document.documentElement).getPropertyValue('--s');let ec;
function eq(){const y=D.equity[$('es').value];let pk=0,md=0;const P=y.map(v=>{pk=Math.max(pk,v);md=Math.max(md,pk-v);return pk});
$('dd').textContent=`افت از قله (روزانه): $${md.toFixed(2)} | پایان: $${y[y.length-1].toFixed(2)}`;if(ec)ec.destroy();
ec=new Chart($('eq'),{type:'line',data:{labels:D.days,datasets:[{label:'PnL تجمعی',data:y,borderColor:'#3454d1',borderWidth:2,pointRadius:0,tension:.15},{label:'قله‌ی قبلی',data:P,borderColor:'#999',borderDash:[5,4],borderWidth:1,pointRadius:0}]},options:{animation:false,maintainAspectRatio:false,scales:{x:{ticks:{color:tc,maxTicksLimit:12}},y:{ticks:{color:tc}}},plugins:{legend:{labels:{color:tc}}}}})}
$('es').onchange=eq;fillSel();render();eq();
</script></body></html>
"""
