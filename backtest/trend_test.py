"""
آزمون روندپیروی و مومنتوم روی کندل «روزانه» — کم‌معامله، مستقل از صرافی (کارمزد و بهره پارامتر است).

چرا این آزمون؟ آزمون سیگنال نشان داد ورودهای کوتاه‌مدت اندیکاتوری برتری ندارند و هزینه‌ی معامله (۰.۱ تا ۰.۷٪ رفت‌وبرگشت)
از هر برتری ممکن بزرگ‌تر است. در تایم‌فریم روزانه با ۱۰ تا ۳۰ معامله در سال، هزینه نسبت به حرکت‌های چند ده درصدی کوچک
می‌شود. اینجا چند فرضیه‌ی **از پیش مشخص** (۱۳ واریانت؛ هیچ جست‌وجوی پارامتر دیگری انجام نمی‌شود) با داده‌ی چندساله سنجیده می‌شود:

  tsmom_ls_L_R   روندپیروی سری‌زمانی، لانگ‌شورت: هر نماد لانگ اگر قیمتش از L روز پیش بالاتر است ، وگرنه شورت
  tsmom_lo_L_R   همان فقط لانگ (وگرنه نقد)
  csmom_L_R      مومنتوم مقطعی خنثی نسبت به بازار: ۲۰٪ نمادهای با بیشترین بازده L روز گذشته لانگ ، ۲۰٪ ضعیف‌ترین‌ها شورت
  R = هر چند روز پوزیشن‌ها بازتنظیم می‌شوند (۷ یا ۳۰)

قواعد بدون نگاه‌به‌آینده: سیگنال از قیمت بسته‌شدن روز t ساخته می‌شود و فقط بازده روز t+1 را می‌گیرد. وزن‌ها: هر نماد فعال
وزن مساوی؛ مجموع مطلق وزن‌ها حداکثر ۱ (بدون اهرم). هزینه = گردش (turnover) × کارمزد هر طرف ؛ بهره‌ی قرض = موجودی شورت × نرخ روزانه.
برای هر واریانت سه سری روزانه ذخیره می‌شود: بازده ناخالص ، گردش و موجودی شورت؛ بازده خالص برای هر کارمزد و بهره‌ای
بعداً از روی همین سه سری ساخته می‌شود (داشبورد اینها را زنده حساب می‌کند).

اجرا (از ریشه‌ی ریپو، بعد از python -m backtest.download_daily):
    python -m backtest.trend_test
    python -m backtest.trend_test --holdout-days 365 --min-symbols 5
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

ANN = 365
NW_LAG = 10
MIN_SYMBOLS_DEFAULT = 5
HOLDOUT_DEFAULT = 365
FEE_SCENARIOS = {"کم (0.04٪)": 0.04, "اهرم حرفه‌ای (0.10٪)": 0.10, "اهرم آسان (0.17٪)": 0.17, "اسپات (0.35٪)": 0.35}


# ----------------------------------------------------------------------------- داده
def load_panel(data_dir: str) -> pd.DataFrame:
    """ماتریس قیمت بسته‌شدن روزانه: ردیف = روز (UTC) ، ستون = نماد (BTC/USDT ...)."""
    cols = {}
    for p in sorted(glob.glob(os.path.join(data_dir, "*_1d.csv"))):
        sym = os.path.basename(p).replace("_1d.csv", "")
        df = pd.read_csv(p, parse_dates=["date"])
        s = df.drop_duplicates("date").set_index("date")["close"].astype(float)
        cols[f"{sym[:-4]}/USDT" if sym.endswith("USDT") else sym] = s
    if not cols:
        raise SystemExit("هیچ فایل روزانه‌ای نیست؛ اول python -m backtest.download_daily را اجرا کن.")
    panel = pd.DataFrame(cols).sort_index()
    panel = panel.reindex(pd.date_range(panel.index.min(), panel.index.max(), freq="D"))
    return panel


# ----------------------------------------------------------------------------- واریانت‌ها
VARIANTS = (
    [("tsmom_ls", L, R) for L, R in ((30, 7), (90, 7), (90, 30), (180, 7), (180, 30))] +
    [("tsmom_lo", L, R) for L, R in ((30, 7), (90, 7), (90, 30), (180, 7), (180, 30))] +
    [("csmom", L, R) for L, R in ((30, 7), (90, 7), (90, 30))]
)
MAX_LOOKBACK = max(v[1] for v in VARIANTS)


def variant_name(kind: str, L: int, R: int) -> str:
    return f"{kind}_{L}_{R}"


def target_weights(close: pd.DataFrame, kind: str, L: int, R: int, start_pos: int) -> pd.DataFrame:
    """
    وزن هدف در پایان هر روز t (فقط با داده‌ی تا همان روز). بازتنظیم هر R روز، از ردیف start_pos؛
    بین بازتنظیم‌ها وزن‌ها ثابت می‌ماند.
    """
    ret_l = close / close.shift(L) - 1                      # NaN اگر کمتر از L روز تاریخچه داشته باشد
    valid = ret_l.notna() & close.notna()
    n_valid = valid.sum(axis=1).replace(0, np.nan)
    if kind in ("tsmom_ls", "tsmom_lo"):
        p = pd.DataFrame(np.where(ret_l > 0, 1.0, -1.0 if kind == "tsmom_ls" else 0.0), index=close.index, columns=close.columns)
        p = p.where(valid, 0.0)
        w = p.div(n_valid, axis=0).fillna(0.0)
    elif kind == "csmom":
        rank = ret_l.where(valid).rank(axis=1, method="first", ascending=True)       # ۱ = ضعیف‌ترین
        k = np.maximum(1, np.floor(n_valid * 0.2)).fillna(1)
        top = rank.gt(n_valid.sub(k, axis=0), axis=0) & valid
        bot = rank.le(k, axis=0) & valid
        w = top.astype(float).mul(0.5 / k, axis=0) - bot.astype(float).mul(0.5 / k, axis=0)
        w = w.mul((n_valid >= 5).astype(float), axis=0).fillna(0.0)                    # مقطعی با کمتر از ۵ نماد بی‌معناست
    else:
        raise ValueError(kind)
    rebal = np.zeros(len(close), dtype=bool)
    rebal[start_pos::R] = True
    w = w.copy()
    w.iloc[~rebal] = np.nan
    w = w.ffill().fillna(0.0)
    w.iloc[:start_pos] = 0.0
    return w


def daily_series(close: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """
    بازده ناخالص ، گردش و موجودی شورتِ هر روز. وزنِ تصمیم‌گرفته‌شده در پایان روز t-1 بازده روز t را می‌گیرد؛
    هزینه‌ی تغییر وزن (گردش) روز t حساب می‌شود.
    """
    ret1 = close.pct_change().fillna(0.0)
    w_prev = w.shift(1).fillna(0.0)
    gross = (w_prev * ret1).sum(axis=1)
    turnover = (w_prev - w.shift(2).fillna(0.0)).abs().sum(axis=1)
    short = (-w_prev).clip(lower=0.0).sum(axis=1)
    return pd.DataFrame({"gross": gross, "turnover": turnover, "short": short})


def net_returns(s: pd.DataFrame, fee_pct: float, borrow_daily_pct: float = 0.0) -> pd.Series:
    return s["gross"] - s["turnover"] * fee_pct / 100 - s["short"] * borrow_daily_pct / 100


# ----------------------------------------------------------------------------- آمار
def equity_stats(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 30:
        return {"n": n}
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    mdd = float((eq / peak - 1).min())
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    years = n / ANN
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    return {"n": n, "ann_ret": mean * ANN, "cagr": cagr, "vol": sd * math.sqrt(ANN),
            "sharpe": (mean / sd * math.sqrt(ANN)) if sd > 0 else 0.0, "mdd": mdd, "total": float(eq.iloc[-1] - 1)}


def nw_tstat(x: np.ndarray, lag: int = NW_LAG) -> tuple[float, float]:
    """t میانگین با خطای استاندارد Newey–West؛ برگشت: (t, p دوطرفه نرمال)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 30:
        return float("nan"), float("nan")
    m = x.mean()
    u = x - m
    var = float((u @ u) / n)
    for l in range(1, lag + 1):
        var += 2 * (1 - l / (lag + 1)) * float((u[l:] @ u[:-l]) / n)
    se = math.sqrt(max(var, 0) / n)
    if se == 0:
        return float("nan"), float("nan")
    t = m / se
    return t, math.erfc(abs(t) / math.sqrt(2))


def alpha_beta(y: np.ndarray, b: np.ndarray, lag: int = NW_LAG) -> dict:
    """رگرسیون y = α + β b ؛ خطای استاندارد HAC (Newey–West). α روزانه."""
    y, b = np.asarray(y, float), np.asarray(b, float)
    n = len(y)
    if n < 60:
        return {"alpha_ann": float("nan"), "beta": float("nan"), "t_alpha": float("nan")}
    X = np.column_stack([np.ones(n), b])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    u = y - X @ coef
    S = (X * u[:, None]).T @ (X * u[:, None]) / n
    for l in range(1, lag + 1):
        Gl = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None]) / n
        S += (1 - l / (lag + 1)) * (Gl + Gl.T)
    cov = XtX_inv @ (S * n) @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return {"alpha_ann": float(coef[0] * ANN), "beta": float(coef[1]), "t_alpha": float(coef[0] / se[0]) if se[0] > 0 else float("nan")}


def verdict(gross_p_bonf: float, half_pos: bool, holdout_gross_pos: bool, net_default_pos: bool, n_days: int) -> str:
    if n_days < 365 * 2:
        return "low_n"
    if gross_p_bonf < 0.05 and half_pos and holdout_gross_pos:
        return "edge_net_pos" if net_default_pos else "edge_below_cost"
    return "hypothesis"


VERDICT_FA = {
    "edge_net_pos": "برتری معنادار و بعد از کارمزد مثبت",
    "edge_below_cost": "برتری معنادار ولی کمتر از کارمزد",
    "hypothesis": "فرضیه (معنادار نیست)",
    "low_n": "داده‌ی ناکافی",
}


# ----------------------------------------------------------------------------- اجرا
def run(panel: pd.DataFrame, min_symbols: int, holdout_days: int, default_fee: float = 0.10, default_borrow: float = 0.0) -> dict:
    # شروع ارزیابی: اولین روزی که حداقل min_symbols نماد ≥ MAX_LOOKBACK+1 روز تاریخچه دارند (برای همه‌ی واریانت‌ها یکسان)
    hist = panel.notna().cumsum()
    ok = (hist > MAX_LOOKBACK).sum(axis=1) >= min_symbols
    if not ok.any():
        raise SystemExit(f"هیچ روزی حداقل {min_symbols} نماد با ≥ {MAX_LOOKBACK + 1} روز تاریخچه ندارد؛ داده‌ی بیشتری بگیر.")
    start_pos = int(np.argmax(ok.to_numpy()))
    idx = panel.index
    series: dict[str, pd.DataFrame] = {}
    for kind, L, R in VARIANTS:
        w = target_weights(panel, kind, L, R, start_pos)
        series[variant_name(kind, L, R)] = daily_series(panel, w).iloc[start_pos:]
    ret1 = panel.pct_change()
    ew_bh = ret1.mean(axis=1, skipna=True).fillna(0.0).iloc[start_pos:]
    btc = (ret1["BTC/USDT"].fillna(0.0) if "BTC/USDT" in ret1 else ew_bh).iloc[start_pos:]
    dates = idx[start_pos:]
    n = len(dates)
    hold = min(holdout_days, n // 3)
    mid = n // 2
    V = len(VARIANTS)
    rows = []
    for name, s in series.items():
        g = s["gross"]
        t, p = nw_tstat(g.to_numpy())
        pb = min(1.0, p * V) if not math.isnan(p) else float("nan")
        net = net_returns(s, default_fee, default_borrow)
        ab = alpha_beta(net.to_numpy(), ew_bh.to_numpy())
        gab = alpha_beta(g.to_numpy(), ew_bh.to_numpy())
        row = {"name": name, "kind": name.split("_")[0] + "_" + name.split("_")[1] if name.startswith("tsmom") else "csmom",
               "L": int(name.split("_")[-2]), "R": int(name.split("_")[-1]),
               "gross": equity_stats(g), "net": equity_stats(net), "t_gross": t, "p_gross": p, "p_bonf": pb,
               "gross_first": float(g.iloc[:mid].mean() * ANN), "gross_second": float(g.iloc[mid:].mean() * ANN),
               "gross_holdout": float(g.iloc[-hold:].mean() * ANN), "net_holdout": float(net.iloc[-hold:].mean() * ANN),
               "net_pre_holdout": float(net.iloc[:-hold].mean() * ANN),
               "alpha_net": ab, "alpha_gross": gab,
               "turnover_yr": float(s["turnover"].mean() * ANN), "short_avg": float(s["short"].mean()),
               "trades_yr": float((s["turnover"] > 1e-9).sum() / (n / ANN))}
        row["verdict"] = verdict(pb if not math.isnan(pb) else 1.0, row["gross_first"] > 0 and row["gross_second"] > 0,
                                 row["gross_holdout"] > 0, row["net"].get("ann_ret", -1) > 0, n)
        rows.append(row)
    bench = {"ew_bh": equity_stats(ew_bh), "btc_bh": equity_stats(btc)}
    return {"series": series, "ew_bh": ew_bh, "btc": btc, "dates": dates, "rows": rows, "bench": bench,
            "start_pos": start_pos, "mid": mid, "hold": hold, "n": n,
            "symbols_over_time": panel.notna().sum(axis=1).iloc[start_pos:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="backtest_data_daily")
    ap.add_argument("--out", default="results/backtest/trend_test")
    ap.add_argument("--min-symbols", type=int, default=MIN_SYMBOLS_DEFAULT)
    ap.add_argument("--holdout-days", type=int, default=HOLDOUT_DEFAULT, help="آخرین N روز جدا نگه‌داشته می‌شود (برای تصمیم نهایی)")
    ap.add_argument("--default-fee", type=float, default=0.10, help="کارمزد هر طرف (٪) برای ستون‌های «بعد از کارمزد» گزارش متنی")
    ap.add_argument("--default-borrow", type=float, default=0.0, help="بهره‌ی قرض روزانه‌ی شورت (٪) برای گزارش متنی")
    a = ap.parse_args()
    panel = load_panel(a.data)
    res = run(panel, a.min_symbols, a.holdout_days, a.default_fee, a.default_borrow)
    from backtest.trend_report import write_outputs
    write_outputs(panel, res, a.out, a.default_fee, a.default_borrow)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
