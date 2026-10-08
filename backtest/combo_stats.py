"""
توابع آماری بک‌تستر ترکیب‌ها (بدون وابستگی به scipy).

همه‌چیز بر اساس علامت واقعی pnl_usdt (بعد از کارمزد) است، نه «برخورد به TP».
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

MIN_N_LOW = 30      # زیر این تعداد: «نمونه ناکافی»
MIN_N_HIGH = 100    # از این تعداد به بالا: اطمینان «زیاد» (در صورت پایداری و معناداری)
MIN_DAYS_FOR_T = 20  # حداقل تعداد روزِ دارای معامله برای آزمون خوشه‌بندی‌شده


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """بازه‌ی اطمینان ۹۵٪ ویلسون برای نسبت برد."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


@lru_cache(maxsize=1)
def _lgamma_table(size: int) -> np.ndarray:
    return np.array([math.lgamma(i + 1) for i in range(size + 1)])


def binom_two_sided_p(k: int, n: int, p0: float) -> float | None:
    """p-value دوطرفه‌ی دقیق binomial (همان تعریف scipy.stats.binomtest)."""
    if n <= 0 or not (0.0 < p0 < 1.0):
        return None
    lg = _lgamma_table(max(n, 2000))
    ks = np.arange(n + 1)
    logpmf = lg[n] - lg[ks] - lg[n - ks] + ks * math.log(p0) + (n - ks) * math.log(1 - p0)
    pmf = np.exp(logpmf)
    pk = pmf[k]
    return float(min(1.0, pmf[pmf <= pk * (1 + 1e-7)].sum()))


def max_drawdown(pnl: np.ndarray) -> float:
    """بیشترین افت پشت‌سرهم از قله‌ی منحنی PnL تجمعی (به دلار)."""
    if len(pnl) == 0:
        return 0.0
    eq = np.concatenate([[0.0], np.cumsum(pnl)])
    peak = np.maximum.accumulate(eq)
    return float((peak - eq).max())


def longest_losing_streak(win: np.ndarray) -> int:
    best = cur = 0
    for w in win:
        cur = 0 if w else cur + 1
        best = max(best, cur)
    return best


def _pf(pnl: np.ndarray):
    gw = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    if gl > 0:
        return float(gw / gl)
    return None if gw > 0 else 0.0     # None = بی‌نهایت (هیچ باختی نبود)


def group_stats(g: pd.DataFrame, span_days: float, slip_cost: float = 0.0) -> dict:
    """
    g باید ستون‌های pnl_usdt، win (bool)، exit_dt (datetime) و half (1|2) داشته باشد.
    ترتیب برای drawdown و streak بر اساس زمان خروج است.

    slip_cost: هزینه‌ی اضافه‌ی فرضی در هر معامله (به دلار) برای تست حساسیت به لغزش (exp_slip / pf_slip).
    p_day: آزمون t روی «جمع PnL هر روز» (خوشه‌بندی روزانه). معاملات هم‌زمان روی ده‌ها نماد هم‌بسته‌اند و
           آزمون binomial (p) آن‌ها را مستقل فرض می‌کند و معناداری را بیش از حد نشان می‌دهد؛ q از p_day ساخته می‌شود.
    """
    g = g.sort_values("exit_dt")
    pnl = g["pnl_usdt"].to_numpy(float)
    win = g["win"].to_numpy(bool)
    n = len(g)
    k = int(win.sum())
    lo, hi = wilson(k, n)
    avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0
    avg_loss = float(-pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else None
    be = (avg_loss / (avg_win + avg_loss)) if (avg_win + avg_loss) > 0 and k > 0 and k < n else None
    p = binom_two_sided_p(k, n, be) if (be is not None and n >= 5) else None
    r = {
        "n": n, "wins": k, "wr": k / n if n else 0.0, "wr_lo": lo, "wr_hi": hi,
        "pf": _pf(pnl), "exp": float(pnl.mean()) if n else 0.0,
        "med": float(np.median(pnl)) if n else 0.0,
        "net": float(pnl.sum()), "payoff": payoff, "be": be, "p": p,
        "mdd": max_drawdown(pnl), "streak": longest_losing_streak(win),
        "tpd": n / span_days if span_days > 0 else 0.0,
    }
    for h in (1, 2):
        gh = g[g["half"] == h]["pnl_usdt"].to_numpy(float)
        r[f"n{h}"] = int(len(gh))
        r[f"exp{h}"] = float(gh.mean()) if len(gh) else None
        r[f"pf{h}"] = _pf(gh) if len(gh) else None
    if r["n1"] >= 5 and r["n2"] >= 5 and r["exp1"] is not None and r["exp2"] is not None:
        if r["exp1"] > 0 and r["exp2"] > 0:
            r["stable"] = "pos"
        elif r["exp1"] < 0 and r["exp2"] < 0:
            r["stable"] = "neg"
        else:
            r["stable"] = "mixed"
    else:
        r["stable"] = "na"
    r["conf"] = "low" if n < MIN_N_LOW else ("mid" if n < MIN_N_HIGH else "high")

    # آزمون خوشه‌بندی‌شده‌ی روزانه
    dsum = g.groupby(g["exit_dt"].dt.strftime("%Y-%m-%d"))["pnl_usdt"].sum().to_numpy(float)
    k_days = len(dsum)
    t_day = p_day = None
    if k_days >= MIN_DAYS_FOR_T:
        sd = float(dsum.std(ddof=1))
        if sd > 0:
            t_day = float(dsum.mean() / (sd / math.sqrt(k_days)))
            p_day = float(math.erfc(abs(t_day) / math.sqrt(2)))
    r["days"], r["t_day"], r["p_day"] = k_days, t_day, p_day

    # حساسیت به لغزش
    pn = pnl - slip_cost
    r["exp_slip"] = float(pn.mean()) if n else 0.0
    r["pf_slip"] = _pf(pn) if n else 0.0
    return r


def bh_adjust(rows: list[dict], field: str = "p") -> None:
    """اصلاح Benjamini–Hochberg روی p-valueهای یک جدول (ستون field)؛ نتیجه در r['q']."""
    idx = [(i, r[field]) for i, r in enumerate(rows) if r.get(field) is not None]
    idx.sort(key=lambda x: x[1])
    m = len(idx)
    prev = 1.0
    for rank, (i, p) in reversed(list(enumerate(idx, 1))):
        prev = min(prev, p * m / rank)
        rows[i]["q"] = prev
    for r in rows:
        r.setdefault("q", None)


def verdict(r: dict) -> str:
    """
    فقط سه برچسب محتاطانه؛ تصمیم «حذف/نگه‌دار» با خود شماست.
      reliable_pos / reliable_neg : n>=100 ، q<0.05 (q از آزمون روزانه‌ی خوشه‌بندی‌شده)،
                                    علامت Expectancy هر دو نیمه‌ی بازه با آن یکی؛
                                    برای «مثبت» علاوه‌بر این Expectancy بعد از لغزش فرضی هم باید مثبت بماند
      low_n                       : n<30
      hypothesis                  : بقیه
    """
    if r["n"] < MIN_N_LOW:
        return "low_n"
    if r["n"] >= MIN_N_HIGH and r.get("q") is not None and r["q"] < 0.05:
        if r["stable"] == "pos" and r["exp"] > 0 and r.get("exp_slip", r["exp"]) > 0:
            return "reliable_pos"
        if r["stable"] == "neg" and r["exp"] < 0:
            return "reliable_neg"
    return "hypothesis"


VERDICT_FA = {
    "reliable_pos": "قابل اتکا: مثبت",
    "reliable_neg": "قابل اتکا: منفی",
    "low_n": "نمونه ناکافی",
    "hypothesis": "فرضیه",
}


def rank_key(r: dict):
    """ترتیب: Profit Factor، Expectancy، وین‌ریت، سود خالص (همه نزولی)."""
    pf = r["pf"] if r["pf"] is not None else 99.0
    return (-pf, -r["exp"], -r["wr"], -r["net"])


def build_table(trades: pd.DataFrame, keys: list[str], span_days: float,
                min_n: int = 1, slip_cost: float = 0.0) -> list[dict]:
    rows = []
    if trades.empty:
        return rows
    for k, g in trades.groupby(keys, sort=False):
        if len(g) < min_n:
            continue
        k = k if isinstance(k, tuple) else (k,)
        r = group_stats(g, span_days, slip_cost)
        r["key"] = list(k)
        rows.append(r)
    bh_adjust(rows, "p_day")
    for r in rows:
        r["verdict"] = verdict(r)
    rows.sort(key=rank_key)
    return rows
