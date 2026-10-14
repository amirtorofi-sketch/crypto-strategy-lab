"""
خروجی‌های آزمون روندپیروی و مومنتوم (backtest/trend_test.py) — داخل ریپو در results/backtest/trend_test/ :

    summary.md       گزارش خوانا
    dashboard.html   داشبورد تعاملی؛ کارمزد هر طرف ، بهره‌ی قرض شورت و اهرم را خودت وارد می‌کنی و همه‌چیز زنده دوباره
                     حساب می‌شود (پس به صرافی خاصی وابسته نیست)
    variants.csv     آمار هر واریانت (پیش‌فرض: کارمزد default-fee)
    daily_returns.csv.gz  سری‌های روزانه (gross ، turnover ، short) هر واریانت + بنچمارک
    meta.json
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from backtest.trend_test import ANN, FEE_SCENARIOS, VERDICT_FA, VARIANTS, net_returns


def fmt(x, d=2, pct=False):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x * 100:.{d}f}%" if pct else f"{x:.{d}f}"


def _years_table(res: dict, name: str, fee: float, borrow: float) -> dict:
    r = net_returns(res["series"][name], fee, borrow)
    return {int(y): float((1 + g).prod() - 1) for y, g in r.groupby(r.index.year)}


def write_outputs(panel: pd.DataFrame, res: dict, out_dir: str, default_fee: float, default_borrow: float) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows, dates = res["rows"], res["dates"]
    n, hold, mid = res["n"], res["hold"], res["mid"]
    # ---- CSV
    flat = []
    for r in rows:
        flat.append({"variant": r["name"], "gross_ann_ret": r["gross"].get("ann_ret"), "gross_sharpe": r["gross"].get("sharpe"),
                     "t_gross_nw": r["t_gross"], "p_bonf": r["p_bonf"], "gross_first_half": r["gross_first"],
                     "gross_second_half": r["gross_second"], "gross_holdout": r["gross_holdout"],
                     "net_ann_ret": r["net"].get("ann_ret"), "net_cagr": r["net"].get("cagr"), "net_sharpe": r["net"].get("sharpe"),
                     "net_mdd": r["net"].get("mdd"), "net_holdout_ann": r["net_holdout"], "alpha_net_ann": r["alpha_net"]["alpha_ann"],
                     "beta": r["alpha_net"]["beta"], "t_alpha_net": r["alpha_net"]["t_alpha"], "turnover_per_year": r["turnover_yr"],
                     "rebalances_per_year": r["trades_yr"], "short_avg": r["short_avg"], "verdict": r["verdict"]})
    pd.DataFrame(flat).round(5).to_csv(os.path.join(out_dir, "variants.csv"), index=False)
    daily = []
    for name, s in res["series"].items():
        d = s.copy()
        d.insert(0, "variant", name)
        daily.append(d)
    bench = pd.DataFrame({"gross": res["ew_bh"], "turnover": 0.0, "short": 0.0})
    bench.insert(0, "variant", "BENCH_EW_BUYHOLD")
    btc = pd.DataFrame({"gross": res["btc"], "turnover": 0.0, "short": 0.0})
    btc.insert(0, "variant", "BENCH_BTC_BUYHOLD")
    pd.concat(daily + [bench, btc]).rename_axis("date").round(6).to_csv(os.path.join(out_dir, "daily_returns.csv.gz"), compression="gzip")
    meta = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "start": str(dates[0].date()), "end": str(dates[-1].date()),
            "days": n, "half_boundary": str(dates[mid].date()), "holdout_start": str(dates[-hold].date()), "holdout_days": hold,
            "symbols_total": int(panel.shape[1]), "symbols_first_day": int(res["symbols_over_time"].iloc[0]),
            "symbols_last_day": int(res["symbols_over_time"].iloc[-1]), "default_fee_pct": default_fee, "default_borrow_daily_pct": default_borrow,
            "bench": res["bench"], "variants": len(VARIANTS)}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)
    _write_md(os.path.join(out_dir, "summary.md"), res, meta, default_fee, default_borrow)
    _write_dashboard(os.path.join(out_dir, "dashboard.html"), res, meta, default_fee, default_borrow)
    print(f"\n✅ نتایج در {out_dir}/ ذخیره شد (summary.md ، dashboard.html ، variants.csv ، daily_returns.csv.gz)\n")
    print(_md_table(res, default_fee, default_borrow))


def _md_table(res: dict, fee: float, borrow: float) -> str:
    h = ("| واریانت | t ناخالص (NW) | p تعدیل‌شده | ناخالص سالانه | Sharpe ناخالص | خالص سالانه | Sharpe خالص | MaxDD خالص | "
         "ناخالص نیمه ۱ / ۲ | ناخالص سال آخر | گردش/سال | α خالص (t) | وضعیت |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|\n")
    L = []
    for r in sorted(res["rows"], key=lambda r: -(r["t_gross"] if not math.isnan(r["t_gross"]) else -9)):
        L.append(f"| {r['name']} | {fmt(r['t_gross'])} | {fmt(r['p_bonf'], 3)} | {fmt(r['gross'].get('ann_ret'), 1, True)} | "
                 f"{fmt(r['gross'].get('sharpe'))} | {fmt(r['net'].get('ann_ret'), 1, True)} | {fmt(r['net'].get('sharpe'))} | "
                 f"{fmt(r['net'].get('mdd'), 0, True)} | {fmt(r['gross_first'], 0, True)} / {fmt(r['gross_second'], 0, True)} | "
                 f"{fmt(r['gross_holdout'], 0, True)} | {r['turnover_yr']:.1f} | {fmt(r['alpha_net']['alpha_ann'], 1, True)} "
                 f"({fmt(r['alpha_net']['t_alpha'])}) | {VERDICT_FA[r['verdict']]} |")
    return h + "\n".join(L) + "\n"


def _write_md(path: str, res: dict, meta: dict, fee: float, borrow: float) -> None:
    rows = res["rows"]
    L = ["# آزمون روندپیروی و مومنتوم روی کندل روزانه\n"]
    L.append(f"- بازه‌ی ارزیابی: {meta['start']} تا {meta['end']} ({meta['days']} روز ≈ {meta['days'] / 365:.1f} سال) | "
             f"مرز نیمه‌ها: {meta['half_boundary']} | سال آخر جدا (holdout): از {meta['holdout_start']} ({meta['holdout_days']} روز)")
    L.append(f"- نمادها: {meta['symbols_total']} (از {meta['symbols_first_day']} نماد در روز اول تا {meta['symbols_last_day']} در روز آخر؛ نمادهای تازه‌لیست‌شده وقتی تاریخچه‌شان به "
             "پنجره‌ی لازم رسید وارد می‌شوند)")
    L.append(f"- {meta['variants']} واریانت از پیش مشخص (بدون جست‌وجوی پارامتر دیگر)؛ بدون اهرم؛ وزن مساوی؛ کارمزد پیش‌فرض گزارش {fee}٪ هر طرف، بهره‌ی قرض شورت {borrow}٪ روزانه.")
    b = meta["bench"]
    L.append(f"- بنچمارک «خرید و نگه‌داشتن» با وزن مساوی: سالانه {fmt(b['ew_bh'].get('ann_ret'), 1, True)} ، Sharpe {fmt(b['ew_bh'].get('sharpe'))} ، "
             f"MaxDD {fmt(b['ew_bh'].get('mdd'), 0, True)} ؛ فقط BTC: سالانه {fmt(b['btc_bh'].get('ann_ret'), 1, True)} ، Sharpe {fmt(b['btc_bh'].get('sharpe'))} ، "
             f"MaxDD {fmt(b['btc_bh'].get('mdd'), 0, True)}.\n")
    L.append("## واریانت‌ها (مرتب بر اساس t ناخالص)\n")
    L.append(_md_table(res, fee, borrow))
    pos = [r for r in rows if r["verdict"] in ("edge_net_pos", "edge_below_cost")]
    good = [r for r in rows if r["verdict"] == "edge_net_pos"]
    L.append("\n## خلاصه\n")
    L.append(f"- **برتری ناخالص معنادار و پایدار** (p تعدیل‌شده<0.05 ، هر دو نیمه و سال آخر مثبت): {len(pos)} از {len(rows)} واریانت"
             + (": " + "، ".join(r['name'] for r in pos) if pos else "."))
    L.append(f"- از این‌ها بعد از کارمزد {fee}٪ هنوز مثبت: {len(good)}" + (": " + "، ".join(r['name'] for r in good) if good else "."))
    best = max(rows, key=lambda r: r["net"].get("sharpe", -9))
    L.append(f"- بهترین Sharpe خالص: {best['name']} ({fmt(best['net'].get('sharpe'))}) ؛ سال آخر (خالص) {fmt(best['net_holdout'], 0, True)} سالانه.")
    L.append("- «α» بازده خالص بعد از حذف اثر بازار (رگرسیون روی بنچمارک وزن‌مساوی)؛ استراتژی‌های فقط‌لانگ بخشی از بازدهشان همان رشد بازار است.\n")
    L.append("## نکات مهم\n")
    L.append(f"- p تعدیل‌شده = p آزمون Newey–West روی بازده روزانه‌ی ناخالص × {meta['variants']} (اصلاح Bonferroni برای تعداد واریانت‌ها).")
    L.append("- سیگنال از بسته‌شدن روز t ساخته می‌شود و فقط بازده روز t+1 را می‌گیرد؛ هیچ نگاه‌به‌آینده‌ای نیست (تست دارد).")
    L.append("- تورش بقا: نمادها همان‌هایی‌اند که امروز پرنقدینگی‌اند؛ ارزهای ورشکسته یا حذف‌شده در داده نیستند و نتایج فقط‌لانگ را خوش‌بینانه می‌کند. مومنتوم مقطعی خنثی نسبت به بازار کمتر آسیب می‌بیند.")
    L.append("- شورت در عمل نیازمند مارجین است: بهره‌ی قرض ، ریسک لیکوئید شدن و محدودیت نماد وجود دارد. بهره را در داشبورد وارد کن.")
    L.append("- داده‌ی بایننس اسپات است؛ اسپرد و لغزش واقعی صرافی جدا باید لحاظ شود.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _rnd(x):
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else round(x, 6)
    if isinstance(x, dict):
        return {k: _rnd(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rnd(v) for v in x]
    return x


def _write_dashboard(path: str, res: dict, meta: dict, fee: float, borrow: float) -> None:
    dates = [d.strftime("%Y-%m-%d") for d in res["dates"]]
    data = {
        "dates": dates,
        "variants": {name: {"g": [round(float(v), 6) for v in s["gross"]], "t": [round(float(v), 5) for v in s["turnover"]],
                            "s": [round(float(v), 5) for v in s["short"]]} for name, s in res["series"].items()},
        "bench": {"EW": [round(float(v), 6) for v in res["ew_bh"]], "BTC": [round(float(v), 6) for v in res["btc"]]},
        "stats": {r["name"]: {"t_gross": r["t_gross"], "p_bonf": r["p_bonf"], "turnover_yr": r["turnover_yr"], "trades_yr": r["trades_yr"],
                              "kind": r["kind"], "L": r["L"], "R": r["R"], "verdict": r["verdict"],
                              "alpha_t": r["alpha_net"]["t_alpha"]} for r in res["rows"]},
        "meta": {"mid": res["mid"], "hold": res["hold"], "start": meta["start"], "end": meta["end"], "fee": fee, "borrow": borrow,
                 "symbols": meta["symbols_total"], "holdout_start": meta["holdout_start"], "half_boundary": meta["half_boundary"],
                 "variants": meta["variants"]},
        "verdict_fa": VERDICT_FA, "scen": FEE_SCENARIOS,
    }
    html = _TEMPLATE.replace("__DATA__", json.dumps(_rnd(data), ensure_ascii=False, separators=(",", ":")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


_TEMPLATE = r"""<!DOCTYPE html><html lang="fa" dir="rtl"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>آزمون روندپیروی و مومنتوم</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#f4f5f7;--card:#fff;--t:#1d2330;--s:#667085;--b:#e3e6eb;--pos:#0a8f5b;--neg:#c93b3b;--acc:#3454d1;--soft:#eef0f4}
@media(prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b202b;--t:#e8ebf1;--s:#98a2b3;--b:#2c3342;--pos:#3ecf8e;--neg:#ff6b6b;--acc:#7c93ff;--soft:#242a38}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--t);font-family:Tahoma,system-ui,sans-serif;font-size:14px;line-height:1.75}
.w{max-width:1500px;margin:auto;padding:16px}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 10px}.sub{color:var(--s);font-size:12px}
.card{background:var(--card);border:1px solid var(--b);border-radius:10px;padding:14px;margin-bottom:14px}
.flt{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:10px}select,input{font:inherit;border:1px solid var(--b);background:var(--card);color:var(--t);border-radius:6px;padding:5px 10px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.tw{overflow-x:auto;max-height:620px;overflow-y:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px;white-space:nowrap}th,td{padding:6px 8px;border-bottom:1px solid var(--b);text-align:right}
th{position:sticky;top:0;background:var(--card);cursor:pointer;color:var(--s);font-weight:600}td.n{direction:ltr}tr.sel td{background:var(--soft)}
.tag{padding:1px 8px;border-radius:10px;font-size:11px}.t1{background:#d8f3e6;color:#065f3d}.t2{background:#fff0c9;color:#6a4a00}.t3{background:#e6e8ee;color:#505868}
.cw{position:relative;height:360px}.kp{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:10px 0}.k{background:var(--soft);border-radius:8px;padding:8px 12px}
.k b{display:block;font-size:19px;direction:ltr;text-align:right}.k span{font-size:12px;color:var(--s)}
.btn{border:1px solid var(--b);background:var(--card);color:var(--t);border-radius:6px;padding:4px 10px;cursor:pointer;font:inherit}
</style></head><body><div class="w">
<h1>آزمون روندپیروی و مومنتوم (کندل روزانه)</h1><div class="sub" id="meta"></div>
<div class="card"><h2>هزینه‌ها: کارمزد و بهره را خودت وارد کن (همه‌چیز زنده دوباره حساب می‌شود)</h2>
<div class="flt">کارمزد هر طرف (٪): <input type="number" id="fee" step="0.01" min="0" style="width:90px">
<span id="presets"></span>
بهره‌ی قرض شورت (٪ روزانه): <input type="number" id="bor" step="0.01" min="0" style="width:90px">
اهرم: <select id="lev"><option>1</option><option>2</option><option>3</option></select>
<span class="sub">اهرم فقط بازده و هزینه و بهره را ضرب می‌کند (ریسک لیکوئید شدن در این مدل لحاظ نمی‌شود).</span></div></div>
<div class="card"><h2>واریانت‌ها</h2><div class="sub">روی هر ردیف کلیک کن تا منحنی‌اش پایین کشیده شود. t ناخالص با Newey–West؛ p تعدیل‌شده = p × <span id="nv"></span> واریانت (Bonferroni). «سال آخر» = داده‌ی جداشده برای تصمیم نهایی.</div>
<div class="tw"><table><thead id="th"></thead><tbody id="tb"></tbody></table></div></div>
<div class="card"><h2>منحنی سرمایه (مقیاس لگاریتمی) — <span id="vn"></span></h2><div class="flt"><label><input type="checkbox" id="showBench" checked> خرید و نگه‌داشتن</label>
<span class="sub">خط‌چین عمودی = شروع سال آخر (holdout)</span></div><div class="cw"><canvas id="eq"></canvas></div></div>
<div class="card"><h2>بازده سال‌به‌سال — <span id="vn2"></span></h2><div class="tw" style="max-height:none"><table id="yr"></table></div></div>
<div class="card"><h2>نکات</h2><ul class="sub" style="line-height:2"><li>سیگنال از بسته‌شدن روز t ساخته می‌شود و فقط بازده روز t+1 را می‌گیرد. وزن هر نماد برابر و بدون اهرم (اهرم بالا فقط ضرب می‌کند).</li>
<li>تورش بقا: نمادها همان‌هایی‌اند که امروز پرنقدینگی‌اند؛ ارزهای ورشکسته در داده نیستند و نتایج فقط‌لانگ را خوش‌بینانه می‌کند.</li>
<li>شورت واقعی به مارجین، بهره و ریسک لیکوئید شدن نیاز دارد؛ برای واریانت‌های لانگ‌شورت بهره را بالا ببر و ببین چه می‌شود.</li>
<li>هیچ تصمیم قطعی گرفته نشده؛ فقط واریانتی با برتری معنادار + پایداری دو نیمه + سال آخر مثبت + بعد از کارمزد واقعی مثبت ارزش آزمون پیش‌رو دارد.</li></ul></div>
</div><script>
const D=__DATA__;const $=i=>document.getElementById(i);const N=D.dates.length,M=D.meta;
const f=(x,d=2)=>x==null||isNaN(x)?'—':(+x).toFixed(d),pc=(x,d=1)=>x==null||isNaN(x)?'—':(x*100).toFixed(d)+'%';
$('meta').textContent=`از ${M.start} تا ${M.end} (${N} روز ≈ ${(N/365).toFixed(1)} سال) | ${M.symbols} نماد | ${M.variants} واریانت | مرز نیمه‌ها ${M.half_boundary} | سال آخر (holdout) از ${M.holdout_start}`;$('nv').textContent=M.variants;
$('fee').value=M.fee;$('bor').value=M.borrow;
$('presets').innerHTML=Object.entries(D.scen).map(([k,v])=>`<button class="btn" data-v="${v}">${k}</button>`).join(' ');
document.querySelectorAll('#presets button').forEach(b=>b.onclick=()=>{$('fee').value=b.dataset.v;recalc()});
let sel=Object.keys(D.variants)[0],sk='sharpe',sd=-1,ch=null;
function net(name){const v=D.variants[name],fee=(+$('fee').value||0)/100,bor=(+$('bor').value||0)/100,lev=+$('lev').value,out=new Array(N);
for(let i=0;i<N;i++)out[i]=lev*(v.g[i]-v.t[i]*fee-v.s[i]*bor);return out}
function stats(r){const n=r.length;let m=0;for(const x of r)m+=x;m/=n;let v=0;for(const x of r)v+=(x-m)*(x-m);const sd=Math.sqrt(v/(n-1));
let eq=1,pk=1,mdd=0;for(const x of r){eq*=1+x;if(eq>pk)pk=eq;const d=eq/pk-1;if(d<mdd)mdd=d}
const yrs=n/365;return{ann:m*365,cagr:eq>0?Math.pow(eq,1/yrs)-1:-1,sharpe:sd>0?m/sd*Math.sqrt(365):0,mdd,total:eq-1}}
function seg(a,from,to){return a.slice(from,to)}
let ROWS=[];
function recalc(){ROWS=Object.keys(D.variants).map(n=>{const r=net(n),s=D.stats[n],all=stats(r),h=seg(r,N-M.hold,N),b=seg(r,0,N-M.hold);
return{name:n,all,hold:stats(h),pre:stats(b),s,verd:D.verdict_fa[s.verdict]}});render();draw()}
const COLS=[['name','واریانت'],['t','t ناخالص'],['pb','p تعدیل‌شده'],['ann','سالانه'],['cagr','رشد مرکب'],['sharpe','Sharpe'],['mdd','MaxDD'],['pre','قبل از سال آخر'],['hold','سال آخر'],['to','گردش/سال'],['tr','بازتنظیم/سال'],['v','وضعیت']];
function val(r,k){switch(k){case'name':return r.name;case't':return r.s.t_gross;case'pb':return r.s.p_bonf;case'ann':return r.all.ann;case'cagr':return r.all.cagr;case'sharpe':return r.all.sharpe;case'mdd':return r.all.mdd;case'pre':return r.pre.ann;case'hold':return r.hold.ann;case'to':return r.s.turnover_yr;case'tr':return r.s.trades_yr;case'v':return r.verd}}
function render(){const R=ROWS.slice().sort((a,b)=>{const x=val(a,sk),y=val(b,sk);if(x==null)return 1;if(y==null)return -1;return(x>y?1:x<y?-1:0)*sd});
$('th').innerHTML='<tr>'+COLS.map(c=>`<th data-k="${c[0]}">${c[1]}</th>`).join('')+'</tr>';
$('tb').innerHTML=R.map(r=>`<tr class="${r.name===sel?'sel':''}" data-n="${r.name}"><td>${r.name}</td><td class="n">${f(r.s.t_gross)}</td><td class="n">${f(r.s.p_bonf,3)}</td><td class="n ${r.all.ann>0?'pos':'neg'}">${pc(r.all.ann)}</td><td class="n ${r.all.cagr>0?'pos':'neg'}">${pc(r.all.cagr)}</td><td class="n">${f(r.all.sharpe)}</td><td class="n">${pc(r.all.mdd,0)}</td><td class="n ${r.pre.ann>0?'pos':'neg'}">${pc(r.pre.ann)}</td><td class="n ${r.hold.ann>0?'pos':'neg'}">${pc(r.hold.ann)}</td><td class="n">${f(r.s.turnover_yr,1)}</td><td class="n">${f(r.s.trades_yr,1)}</td><td><span class="tag ${r.s.verdict==='edge_net_pos'?'t1':r.s.verdict==='edge_below_cost'?'t2':'t3'}">${r.verd}</span></td></tr>`).join('');
document.querySelectorAll('#tb tr').forEach(tr=>tr.onclick=()=>{sel=tr.dataset.n;render();draw()});
document.querySelectorAll('th').forEach(h=>h.onclick=()=>{const k=h.dataset.k;if(sk===k)sd=-sd;else{sk=k;sd=-1}render()})}
function curve(r){let e=1;return r.map(x=>(e*=1+x,e))}
function draw(){$('vn').textContent=sel;$('vn2').textContent=sel;const r=net(sel),tc=getComputedStyle(document.documentElement).getPropertyValue('--s');
const ds=[{label:sel+' (خالص)',data:curve(r),borderColor:'#3454d1',borderWidth:2,pointRadius:0}];
if($('showBench').checked){ds.push({label:'خرید و نگه‌داشتن وزن‌مساوی',data:curve(D.bench.EW),borderColor:'#999',borderWidth:1,pointRadius:0,borderDash:[4,3]});ds.push({label:'BTC',data:curve(D.bench.BTC),borderColor:'#e08a00',borderWidth:1,pointRadius:0,borderDash:[2,3]})}
if(ch)ch.destroy();const hs=N-M.hold;
ch=new Chart($('eq'),{type:'line',data:{labels:D.dates,datasets:ds},options:{animation:false,maintainAspectRatio:false,scales:{x:{ticks:{color:tc,maxTicksLimit:10}},y:{type:'logarithmic',ticks:{color:tc}}},plugins:{legend:{labels:{color:tc}},tooltip:{mode:'index',intersect:false}}},
plugins:[{id:'hold',afterDraw(c){const x=c.scales.x.getPixelForValue(hs);const a=c.chartArea;const g=c.ctx;g.save();g.setLineDash([5,4]);g.strokeStyle='#c93b3b';g.beginPath();g.moveTo(x,a.top);g.lineTo(x,a.bottom);g.stroke();g.restore()}}]});
const yrs={};D.dates.forEach((d,i)=>{const y=d.slice(0,4);(yrs[y]=yrs[y]||[]).push(i)});
const ys=Object.keys(yrs).sort(),row=a=>ys.map(y=>{let e=1;for(const i of yrs[y])e*=1+a[i];return e-1}),mine=row(r),ew=row(D.bench.EW),btc=row(D.bench.BTC);
const cell=v=>`<td class="n ${v>0?'pos':'neg'}">${pc(v,0)}</td>`;
$('yr').innerHTML='<thead><tr><th></th>'+ys.map(y=>`<th>${y}</th>`).join('')+'</tr></thead><tbody><tr><td>'+sel+'</td>'+mine.map(cell).join('')+'</tr><tr><td>خرید و نگه‌داشتن وزن‌مساوی</td>'+ew.map(cell).join('')+'</tr><tr><td>BTC</td>'+btc.map(cell).join('')+'</tr></tbody>'}
['fee','bor','lev'].forEach(i=>$(i).oninput=recalc);$('showBench').onchange=draw;
recalc();
</script></body></html>
"""
