"""
خروجی‌های بک‌تستر ترکیب‌ها؛ همه داخل خود ریپو (پیش‌فرض results/backtest/combos/) ذخیره می‌شود:

    summary.md          گزارش خوانا (در گیت‌هاب مستقیم دیده می‌شود)
    dashboard.html      داشبورد تعاملی (فایل را دانلود کن و در مرورگر باز کن)
    by_combo.csv        استراتژی × سشن (جدول اصلی)
    by_strategy.csv     هر استراتژی (جمع سشن‌های انتخاب‌شده)
    by_combo_symbol.csv استراتژی × سشن × نماد
    by_symbol.csv       هر نماد
    by_all_sessions.csv (فقط با --all-sessions) همه‌ی سشن‌ها + ستون selected
    trades.csv.gz       همه‌ی معاملات شبیه‌سازی‌شده
    meta.json           پارامترها و وضعیت داده‌ی هر نماد
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from backtest.combo_stats import VERDICT_FA, build_table, group_stats

HALF_FA = {"pos": "هر دو نیمه مثبت", "neg": "هر دو نیمه منفی", "mixed": "ناپایدار", "na": "—"}


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
    first = [c for c in key_names if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]].round(4)


def _equity(trades: pd.DataFrame, days: list[str]) -> dict[str, list[float]]:
    t = trades.copy()
    t["day"] = t["exit_dt"].dt.strftime("%Y-%m-%d")

    def curve(g):
        return [round(float(v), 4) for v in g.groupby("day")["pnl_usdt"].sum().reindex(days, fill_value=0).cumsum()]

    eq = {"ALL": curve(t)}
    for (s, ses), g in t.groupby(["strategy", "session"]):
        eq[f"{s} | {ses}"] = curve(g)
    return eq


def _md_table(rows: list[dict], limit: int | None = None) -> str:
    head = "| ترکیب | n | وین‌ریت | بازه ۹۵٪ | PF | Expectancy $ | سود خالص $ | PF نیمه ۱ | PF نیمه ۲ | q | وضعیت |\n"
    head += "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|\n"
    body = []
    for r in rows[:limit]:
        name = " \\| ".join(r["key"])
        body.append(
            f"| {name} | {r['n']} | {r['wr']*100:.1f}% | {r['wr_lo']*100:.0f}–{r['wr_hi']*100:.0f}% | "
            f"{_fmt(r['pf'])} | {r['exp']:.4f} | {r['net']:.2f} | {_fmt(r['pf1'])} | {_fmt(r['pf2'])} | "
            f"{_fmt(r['q'], 3)} | {VERDICT_FA[r['verdict']]} |")
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
    tables = {
        "combo": (["strategy", "session"], build_table(sel, ["strategy", "session"], span)),
        "strategy": (["strategy"], build_table(sel, ["strategy"], span)),
        "combo_symbol": (["strategy", "session", "symbol"], build_table(sel, ["strategy", "session", "symbol"], span)),
        "symbol": (["symbol"], build_table(sel, ["symbol"], span)),
    }
    if params["all_sessions"]:
        rows = build_table(t, ["strategy", "session"], span)
        sel_map = {(s, ses) for s, ses_list in params["combos"].items() for ses in ses_list}
        for r in rows:
            r["selected"] = tuple(r["key"]) in sel_map
        tables["all_sessions"] = (["strategy", "session"], rows)

    overall = group_stats(sel, span)
    days = sorted(t["exit_dt"].dt.strftime("%Y-%m-%d").unique())
    equity = _equity(sel, days)

    # ---- CSV و معاملات
    for name, (keys, rows) in tables.items():
        fname = {"combo": "by_combo", "strategy": "by_strategy", "combo_symbol": "by_combo_symbol",
                 "symbol": "by_symbol", "all_sessions": "by_all_sessions"}[name]
        df = _flat(rows, keys)
        if name == "all_sessions" and "selected" in df.columns:
            df = df[keys + ["selected"] + [c for c in df.columns if c not in keys + ["selected"]]]
        df.to_csv(os.path.join(out_dir, f"{fname}.csv"), index=False)
    t.drop(columns=["entry_dt", "exit_dt"]).to_csv(os.path.join(out_dir, "trades.csv.gz"), index=False, compression="gzip")

    meta = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "period_start": str(t["entry_dt"].min()), "period_end": str(t["exit_dt"].max()), "span_days": round(span, 2),
            "half_boundary": str(mid), "trades_all": int(len(t)), "trades_selected": int(len(sel)),
            "params": params, "symbols": metas}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)

    _write_md(os.path.join(out_dir, "summary.md"), tables, overall, meta, params)
    _write_dashboard(os.path.join(out_dir, "dashboard.html"), tables, overall, meta, equity, days)
    print(f"\n✅ نتایج در {out_dir}/ ذخیره شد (summary.md ، dashboard.html ، CSVها ، trades.csv.gz)\n")
    print(_md_table(tables["combo"][1]))


def _write_md(path: str, tables: dict, o: dict, meta: dict, params: dict) -> None:
    combo = tables["combo"][1]
    L = []
    L.append("# نتیجه‌ی بک‌تست ترکیب‌های استراتژی × سشن\n")
    L.append(f"- بازه: {meta['period_start'][:10]} تا {meta['period_end'][:10]} ({meta['span_days']:.0f} روز) | "
             f"نیمه‌ی اول/دوم: مرز {meta['half_boundary'][:10]}")
    L.append(f"- نمادها: {params['symbols_with_data']} از {params['symbols_requested']} نماد دست‌ترید داده داشتند"
             + (f" (بدون داده: {', '.join(params['symbols_missing'][:10])}{'…' if len(params['symbols_missing']) > 10 else ''})"
                if params["symbols_missing"] else ""))
    L.append(f"- حجم ثابت {params['trade_value']:g}$ ، کارمزد هر طرف {params['fee_pct']}% ، اسلیپیج هر طرف {params['slippage_pct']}% ، "
             f"min_reward_pct={params['min_reward_pct']}")
    L.append(f"- کل معاملات ترکیب‌های انتخاب‌شده: **{o['n']}** | وین‌ریت {o['wr']*100:.1f}% "
             f"(سربه‌سر {_fmt(o['be'] and o['be']*100, 1)}%) | PF {_fmt(o['pf'], 3)} | Expectancy {o['exp']:.4f}$ | "
             f"سود خالص {o['net']:.2f}$ | MaxDD {o['mdd']:.2f}$\n")
    L.append("## ترکیب‌ها (مرتب: PF، Expectancy، وین‌ریت، سود خالص)\n")
    L.append(_md_table(combo))
    rel_pos = [r for r in combo if r["verdict"] == "reliable_pos"]
    rel_neg = [r for r in combo if r["verdict"] == "reliable_neg"]
    L.append("\n## خلاصه\n")
    L.append(f"- **قابل اتکا و مثبت** ({len(rel_pos)}): " + (", ".join(" | ".join(r["key"]) for r in rel_pos) or "هیچ‌کدام"))
    L.append(f"- **قابل اتکا و منفی** ({len(rel_neg)}): " + (", ".join(" | ".join(r["key"]) for r in rel_neg) or "هیچ‌کدام"))
    L.append(f"- بقیه یا فرضیه‌اند یا نمونه‌شان کم است؛ تصمیم «حذف/نگه‌دار» با خودتان است.\n")
    if "all_sessions" in tables:
        L.append("## چک سوگیری انتخاب: همین استراتژی‌ها در همه‌ی سشن‌ها\n")
        L.append("اگر ترکیب‌های انتخاب‌شده فقط به‌خاطر انتخاب از روی داده‌ی زنده خوب به نظر می‌رسیدند، روی تاریخچه‌ی بلند "
                 "باید از بقیه‌ی سشن‌ها بهتر نباشند. (در این حالت هر سشن مستقل شبیه‌سازی شده؛ اعداد کلی بالا با حالت پیش‌فرض "
                 "ممکن است کمی فرق کند.)\n")
        L.append("| استراتژی | سشن | انتخاب‌شده؟ | n | PF | Expectancy $ | q | وضعیت |\n|---|---|---|--:|--:|--:|--:|---|")
        for r in sorted(tables["all_sessions"][1], key=lambda r: (r["key"][0], r["key"][1])):
            L.append(f"| {r['key'][0]} | {r['key'][1]} | {'✅' if r['selected'] else ''} | {r['n']} | {_fmt(r['pf'])} | "
                     f"{r['exp']:.4f} | {_fmt(r['q'], 3)} | {VERDICT_FA[r['verdict']]} |")
        L.append("")
    L.append("## نکات مهم\n")
    L.append("- وین‌ریت از علامت واقعی PnL بعد از کارمزد است، نه «برخورد به TP».")
    L.append("- q = p-value تصحیح‌شده برای مقایسه‌ی چندگانه (Benjamini–Hochberg) درون همان جدول؛ بدون آن، "
             "در صدها ترکیب همیشه چند مورد شانسی خوب به نظر می‌رسند.")
    L.append("- «قابل اتکا» یعنی n≥۱۰۰ ، q<۰.۰۵ و علامت Expectancy در هر دو نیمه‌ی بازه یکی باشد.")
    L.append("- معاملات هم‌زمان روی نمادهای هم‌بسته (مثلاً در یک روز نزولیِ کل بازار) مستقل نیستند؛ p-valueها خوش‌بینانه‌اند.")
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


def _write_dashboard(path: str, tables: dict, o: dict, meta: dict, equity: dict, days: list[str]) -> None:
    payload = {
        "tables": {k: {"keys": v[0], "rows": v[1]} for k, v in tables.items()},
        "overall": o, "equity": equity, "days": days,
        "meta": {"start": meta["period_start"][:10], "end": meta["period_end"][:10], "span": meta["span_days"],
                 "mid": meta["half_boundary"][:10], "p": {k: v for k, v in meta["params"].items()
                                                          if k in ("fee_pct", "slippage_pct", "trade_value",
                                                                   "min_reward_pct", "all_sessions",
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
.w{max-width:1400px;margin:auto;padding:16px}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 10px}.sub{color:var(--s);font-size:12px}
.card{background:var(--card);border:1px solid var(--b);border-radius:10px;padding:14px;margin-bottom:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}.k{background:var(--card);border:1px solid var(--b);border-radius:10px;padding:10px 14px}
.k b{display:block;font-size:22px;direction:ltr;text-align:right}.k span{color:var(--s);font-size:12px}.pos{color:var(--pos)}.neg{color:var(--neg)}
.tabs,.flt{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}.tabs button,select,input{font:inherit;border:1px solid var(--b);background:var(--card);color:var(--t);border-radius:6px;padding:5px 10px}
.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.tw{overflow-x:auto;max-height:560px;overflow-y:auto}table{border-collapse:collapse;width:100%;font-size:12.5px;white-space:nowrap}th,td{padding:6px 8px;border-bottom:1px solid var(--b);text-align:right}
th{position:sticky;top:0;background:var(--card);cursor:pointer;color:var(--s);font-weight:600}td.n{direction:ltr}
.tag{padding:1px 8px;border-radius:10px;font-size:11px}.t1{background:#d8f3e6;color:#065f3d}.t2{background:#fde0e0;color:#8b1f1f}.t3{background:#fff0c9;color:#6a4a00}.t4{background:#e6e8ee;color:#505868}
.cw{position:relative;height:340px}
</style></head><body><div class="w">
<h1>بک‌تست ترکیب‌های استراتژی × سشن</h1><div class="sub" id="meta"></div><br>
<div class="kpis" id="kpis"></div>
<div class="card"><h2>رتبه‌بندی (PF، Expectancy، وین‌ریت، سود خالص)</h2><div class="tabs" id="tabs"></div>
<div class="flt">حداقل n: <input type="number" id="minn" value="10" style="width:70px">
وضعیت: <select id="vf"><option value="">همه</option><option value="reliable_pos">قابل اتکا: مثبت</option><option value="reliable_neg">قابل اتکا: منفی</option><option value="hypothesis">فرضیه</option><option value="low_n">نمونه ناکافی</option></select>
ثبات دو نیمه: <select id="stab"><option value="">همه</option><option value="pos">هر دو نیمه مثبت</option><option value="neg">هر دو نیمه منفی</option><option value="mixed">ناپایدار</option></select>
<input id="q" placeholder="جستجو..."><span class="sub" id="cnt"></span></div>
<div class="tw"><table><thead id="th"></thead><tbody id="tb"></tbody></table></div>
<div class="sub" style="margin-top:8px">p: آزمون binomial نسبت به وین‌ریت سربه‌سر همان گروه. q: اصلاح Benjamini–Hochberg درون همین جدول. «قابل اتکا» = n≥۱۰۰ و q&lt;۰.۰۵ و Expectancy هر دو نیمه‌ی بازه هم‌علامت. معاملات هم‌زمان مستقل نیستند، پس p خوش‌بینانه است.</div></div>
<div class="card"><h2>منحنی PnL تجمعی روزانه</h2><div class="flt"><select id="es"></select><span id="dd" class="sub"></span></div><div class="cw"><canvas id="eq"></canvas></div></div>
</div><script>
const D=__DATA__;const $=i=>document.getElementById(i);
const NAMES={combo:'استراتژی × سشن',strategy:'استراتژی',combo_symbol:'استراتژی × سشن × نماد',symbol:'نماد',all_sessions:'همه‌ی سشن‌ها (چک سوگیری)'};
const CL={low:'کم',mid:'متوسط',high:'زیاد'};let lv='combo',sk='rank',sd=-1;
const f=(x,d=2)=>x==null?'—':(+x).toFixed(d),pc=x=>x==null?'—':(x*100).toFixed(1)+'%';
const VC={reliable_pos:'t1',reliable_neg:'t2',hypothesis:'t3',low_n:'t4'};
const COLS=[['key','گروه'],['n','n'],['conf','اطمینان'],['wr','وین‌ریت'],['ci','بازه ۹۵٪'],['pf','PF'],['exp','Expectancy $'],['med','میانه $'],['payoff','Payoff'],['be','سربه‌سر'],['net','سود خالص $'],['mdd','MaxDD $'],['streak','باخت پیاپی'],['tpd','معامله/روز'],['exp1','Exp نیمه ۱'],['exp2','Exp نیمه ۲'],['p','p'],['q','q'],['verdict','وضعیت']];
function val(r,k){if(k==='key')return r.key.join(' | ');if(k==='ci')return r.wr_lo;if(k==='conf')return{low:0,mid:1,high:2}[r.conf];if(k==='verdict')return D.verdict_fa[r.verdict];return r[k]}
function rankv(r){return[(r.pf==null?99:r.pf),r.exp,r.wr,r.net]}
function render(){const T=D.tables[lv];let R=T.rows.slice();const mn=+$('minn').value||0,vf=$('vf').value,st=$('stab').value,qq=$('q').value.trim();
R=R.filter(r=>r.n>=mn&&(!vf||r.verdict===vf)&&(!st||r.stable===st)&&(!qq||r.key.join(' ').includes(qq)));
if(sk==='rank')R.sort((a,b)=>{const x=rankv(a),y=rankv(b);for(let i=0;i<4;i++)if(x[i]!==y[i])return y[i]-x[i];return 0});
else R.sort((a,b)=>{const x=val(a,sk),y=val(b,sk);if(x==null)return 1;if(y==null)return -1;return(x>y?1:x<y?-1:0)*sd});
$('cnt').textContent=R.length+' گروه';
$('th').innerHTML='<tr>'+COLS.map(c=>`<th data-k="${c[0]}">${c[1]}</th>`).join('')+'</tr>';
$('tb').innerHTML=R.slice(0,500).map(r=>`<tr${r.n<30?' style="opacity:.6"':''}><td>${r.key.join(' | ')}${r.selected?' <span class="tag t1">انتخاب‌شده</span>':''}</td><td class="n">${r.n}</td><td>${CL[r.conf]}</td><td class="n">${pc(r.wr)}</td><td class="n">${pc(r.wr_lo)}–${pc(r.wr_hi)}</td><td class="n ${r.pf==null||r.pf>=1?'pos':'neg'}">${r.pf==null?'∞':f(r.pf)}</td><td class="n ${r.exp>0?'pos':'neg'}">${f(r.exp,4)}</td><td class="n">${f(r.med,4)}</td><td class="n">${f(r.payoff)}</td><td class="n">${pc(r.be)}</td><td class="n ${r.net>0?'pos':'neg'}">${f(r.net)}</td><td class="n">${f(r.mdd)}</td><td class="n">${r.streak}</td><td class="n">${f(r.tpd,2)}</td><td class="n">${f(r.exp1,4)}</td><td class="n">${f(r.exp2,4)}</td><td class="n">${r.p==null?'—':r.p<.001?'<.001':f(r.p,3)}</td><td class="n">${r.q==null?'—':r.q<.001?'<.001':f(r.q,3)}</td><td><span class="tag ${VC[r.verdict]}">${D.verdict_fa[r.verdict]}</span></td></tr>`).join('');
document.querySelectorAll('th').forEach(t=>t.onclick=()=>{const k=t.dataset.k;if(sk===k)sd=-sd;else{sk=k;sd=-1}render()})}
$('tabs').innerHTML=Object.keys(D.tables).map(k=>`<button data-l="${k}" class="${k===lv?'on':''}">${NAMES[k]}</button>`).join('');
document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>{lv=b.dataset.l;sk='rank';document.querySelectorAll('#tabs button').forEach(x=>x.classList.toggle('on',x===b));render()});
['minn','vf','stab','q'].forEach(i=>$(i).oninput=render);
const O=D.overall,M=D.meta;
$('meta').textContent=`از ${M.start} تا ${M.end} (${M.span.toFixed(0)} روز) | ${M.p.symbols_with_data} از ${M.p.symbols_requested} نماد | حجم ${M.p.trade_value}$ | کارمزد هر طرف ${M.p.fee_pct}% + اسلیپیج ${M.p.slippage_pct}% | مرز نیمه‌ها: ${M.mid}`;
const kp=[['معاملات',O.n,''],['وین‌ریت واقعی',pc(O.wr),''],['سربه‌سر لازم',pc(O.be),''],['Profit Factor',O.pf==null?'∞':f(O.pf,3),(O.pf==null||O.pf>=1)?'pos':'neg'],['Expectancy / معامله','$'+f(O.exp,4),O.exp>0?'pos':'neg'],['سود خالص','$'+f(O.net),O.net>0?'pos':'neg'],['Max Drawdown','$'+f(O.mdd),'neg'],['بیشترین باخت پیاپی',O.streak,'']];
$('kpis').innerHTML=kp.map(k=>`<div class="k"><span>${k[0]}</span><b class="${k[2]}">${k[1]}</b></div>`).join('');
Object.keys(D.equity).forEach(s=>$('es').add(new Option(s==='ALL'?'همه‌ی ترکیب‌ها':s,s)));
const tc=getComputedStyle(document.documentElement).getPropertyValue('--s');let ec;
function eq(){const y=D.equity[$('es').value];let pk=0,md=0;const P=y.map(v=>{pk=Math.max(pk,v);md=Math.max(md,pk-v);return pk});
$('dd').textContent=`افت از قله (روزانه): $${md.toFixed(2)} | پایان: $${y[y.length-1].toFixed(2)}`;if(ec)ec.destroy();
ec=new Chart($('eq'),{type:'line',data:{labels:D.days,datasets:[{label:'PnL تجمعی',data:y,borderColor:'#3454d1',borderWidth:2,pointRadius:0,tension:.15},{label:'قله‌ی قبلی',data:P,borderColor:'#999',borderDash:[5,4],borderWidth:1,pointRadius:0}]},options:{animation:false,maintainAspectRatio:false,scales:{x:{ticks:{color:tc,maxTicksLimit:12}},y:{ticks:{color:tc}}},plugins:{legend:{labels:{color:tc}}}}})}
$('es').onchange=eq;render();eq();
</script></body></html>
"""
