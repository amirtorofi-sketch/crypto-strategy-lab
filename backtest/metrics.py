from __future__ import annotations
import pandas as pd
import numpy as np


def compute_metrics(trades_df: pd.DataFrame, initial_balance: float = 1000.0, risk_pct: float = 1.0) -> dict:
    if trades_df.empty or "result" not in trades_df.columns:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "max_drawdown_pct": 0.0, "net_return_pct": 0.0, "avg_rr": 0.0,
            "final_balance": initial_balance,
        }
    closed = trades_df[trades_df["result"].isin(["win", "loss"])].copy()
    if closed.empty:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "max_drawdown_pct": 0.0, "net_return_pct": 0.0, "avg_rr": 0.0,
            "final_balance": initial_balance,
        }

    # هر معامله ریسک risk_pct از بالانس لحظه‌ای را می‌گیرد (ساده‌سازی: بر مبنای بالانس اولیه)
    balance = initial_balance
    equity_curve = [balance]
    for _, row in closed.iterrows():
        risk_amount = balance * (risk_pct / 100)
        # pnl_pct از حرکت قیمت است؛ آن‌را به نسبت SL دوباره مقیاس نمی‌کنیم، فرض حجم ثابت بر مبنای ریسک درصدی از استاپ:
        stop_dist_pct = abs(row["entry_price"] - row["stop_loss"]) / row["entry_price"] * 100
        if stop_dist_pct <= 0:
            continue
        position_size_usdt = risk_amount / (stop_dist_pct / 100)
        pnl_usdt = position_size_usdt * (row["pnl_pct"] / 100)
        balance += pnl_usdt
        equity_curve.append(balance)

    equity = pd.Series(equity_curve)
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    max_dd = drawdown.min()

    wins = closed[closed["result"] == "win"]
    losses = closed[closed["result"] == "loss"]
    gross_profit = wins["pnl_pct"].clip(lower=0).sum()
    gross_loss = losses["pnl_pct"].clip(upper=0).abs().sum()
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    return {
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 2),
        "profit_factor": profit_factor,
        "max_drawdown_pct": round(max_dd, 2),
        "net_return_pct": round((balance - initial_balance) / initial_balance * 100, 2),
        "avg_pnl_pct_per_trade": round(closed["pnl_pct"].mean(), 3),
        "final_balance": round(balance, 2),
    }
