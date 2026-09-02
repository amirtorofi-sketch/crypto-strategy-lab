# کاتالوگ ۱۰۰ استراتژی

علامت ✅ یعنی در کد پیاده‌سازی شده و در `strategies/registry.py` فعال است
(الان **۴۰ استراتژی** پیاده‌سازی‌شده‌اند). بقیه به‌عنوان بک‌لاگ لیست شده‌اند
تا به مرور با همون الگو (یک کلاس در فایل دسته‌ی مربوطه) اضافه شوند.

هر استراتژی پیاده‌سازی‌شده یک **تایم‌فریم مخصوص به خودش** دارد (به‌جای
یک تایم‌فریم مشترک برای همه) - چون مثلاً منطق Silver Bullet ذاتاً برای
5m طراحی شده ولی Golden Cross 50/200 روی 1d معنا دارد. تایم‌فریم هر
استراتژی جلوی اسمش با `[tf]` نوشته شده و همان‌جا هم در کد
(`strategy.timeframe`) واکشی داده و هم در اجرای زنده و بک‌تست استفاده
می‌شود - برای دیدنش با دستور `python -m strategies.registry`.

## چطور یک استراتژی جدید اضافه کنم؟
۱. فایل دسته مربوطه را باز کن، مثلاً `strategies/ict/strategies.py`
۲. یک کلاس جدید بساز که از `Strategy` ارث‌بری کند و `generate_signal(df)`
   یک `Signal(side, entry, stop_loss, take_profit, reason)` برگرداند یا `None`
۳. کلاس را به لیست `STRATEGIES` همان فایل اضافه کن
۴. همین! هم در بک‌تست هم در ران‌تایم زنده و هم در تلگرام خودکار فعال می‌شود.

---

## ۱. ICT (Inner Circle Trader) — ۲۰ مورد
1. ✅ Optimal Trade Entry (OTE) — `ict_optimal_trade_entry_ote` `[15m]`
2. ✅ Silver Bullet (پنجره NY AM) — `ict_silver_bullet_ny_am` `[5m]`
3. ✅ Judas Swing — `ict_judas_swing` `[5m]`
4. ✅ Power of Three (AMD: Accumulation-Manipulation-Distribution) — `ict_power_of_three` `[1h]`
5. ✅ Breakaway / Imbalance Candle — `ict_breakaway_imbalance` `[15m]`
6. ✅ Mitigation Block — `ict_mitigation_block` `[1h]`
7. ✅ London Killzone Breakout — `ict_london_killzone_breakout` `[15m]`
8. ✅ New York Killzone Reversal — `ict_ny_killzone_reversal` `[15m]`
9. ✅ Asian Range Liquidity Grab — `ict_asian_range_liquidity_grab` `[15m]`
10. Turtle Soup (False Breakout ICT)
11. Unicorn Model (OB + FVG هم‌پوشان)
12. Venom Model (دو Liquidity Sweep متوالی)
13. Rejection Block
14. Propulsion Block
15. Daily Bias + H1 CHoCH Confluence
16. Weekly Profile (High/Low of Week) Reversal
17. Midnight Open Reference Price
18. True Day Open Algorithm
19. IPDA Data Range (20/40/60 روز)
20. Seek & Destroy (چند سویپ متوالی قبل از حرکت اصلی)

## ۲. Smart Money Concepts (SMC) — ۲۰ مورد
21. ✅ BOS Continuation — `smc_bos_continuation` `[1h]`
22. ✅ CHoCH Reversal — `smc_choch_reversal` `[1h]`
23. ✅ Order Block Retest — `smc_order_block_retest` `[1h]`
24. ✅ Fair Value Gap Fill — `smc_fvg_fill_entry` `[15m]`
25. ✅ Liquidity Sweep Reversal — `smc_liquidity_sweep_reversal` `[15m]`
26. ✅ Premium/Discount Zone — `smc_premium_discount_zone` `[1h]`
27. ✅ Breaker Block — `smc_breaker_block` `[1h]`
28. ✅ Equal Highs/Lows Liquidity Grab — `smc_equal_highs_lows_grab` `[15m]`
29. Internal vs External Structure Shift
30. Multi-Timeframe Order Block Confluence
31. Liquidity Pool Mapping (Buy-side/Sell-side)
32. Inducement + Order Block Combo
33. Wyckoff Spring در SMC
34. Wyckoff Upthrust در SMC
35. Smart Money Divergence (قیمت/حجم دلتا)
36. Volume Profile POC Rejection
37. Institutional Candle (حجم غیرعادی + بدنه بزرگ)
38. Range Expansion بعد از تراکم SMC
39. Multi-Leg Liquidity Run
40. Reversal at Daily/Weekly Open با تایید ساختار

## ۳. پرایس اکشن (Price Action) — ۲۰ مورد
41. ✅ Engulfing Candle — `pa_engulfing_candle` `[4h]`
42. ✅ Pin Bar Rejection — `pa_pin_bar_rejection` `[4h]`
43. ✅ Inside Bar Breakout — `pa_inside_bar_breakout` `[1h]`
44. ✅ Double Top/Bottom — `pa_double_top_bottom` `[4h]`
45. ✅ Head & Shoulders — `pa_head_and_shoulders` `[1d]`
46. ✅ Support/Resistance Bounce — `pa_support_resistance_bounce` `[1h]`
47. ✅ Trendline Break — `pa_trendline_break` `[4h]`
48. ✅ Three White Soldiers / Black Crows — `pa_three_white_soldiers_crows` `[4h]`
49. Morning Star / Evening Star
50. Harami Pattern
51. Tweezer Top/Bottom
52. Flag / Pennant Continuation
53. Ascending/Descending Triangle Breakout
54. Symmetrical Triangle Breakout
55. Wedge Reversal (Rising/Falling)
56. Cup and Handle
57. Rounding Bottom/Top
58. Fakeout / False Breakout Fade
59. Break-Retest-Continuation کلاسیک
60. Multiple Timeframe Price Action Confluence

## ۴. کلاسیک (اندیکاتور-محور) — ۲۵ مورد
61. ✅ EMA20/EMA50 Crossover — `classic_ma_crossover_20_50` `[4h]`
62. ✅ Golden/Death Cross 50/200 — `classic_golden_cross_50_200` `[1d]`
63. ✅ RSI Oversold/Overbought Reversal — `classic_rsi_reversal_30_70` `[1h]`
64. ✅ MACD Signal Crossover — `classic_macd_signal_cross` `[1h]`
65. ✅ Bollinger Band Mean Reversion — `classic_bollinger_mean_reversion` `[1h]`
66. ✅ Bollinger Squeeze Breakout — `classic_bollinger_breakout` `[4h]`
67. ✅ Stochastic Crossover — `classic_stochastic_cross` `[1h]`
68. ✅ Donchian Channel Breakout — `classic_donchian_channel_breakout` `[4h]`
69. ✅ ADX Trend Following — `classic_adx_trend_following` `[4h]`
70. ✅ Ichimoku Kumo Breakout — `classic_ichimoku_kumo_breakout` `[4h]`
71. ✅ Volume Spike Breakout — `classic_volume_spike_breakout` `[15m]`
72. Parabolic SAR Trend Flip
73. Williams %R Reversal
74. CCI (Commodity Channel Index) Extremes
75. Keltner Channel Breakout
76. SuperTrend Flip
77. VWAP Reversion (Intraday)
78. VWAP Breakout با حجم
79. Heikin Ashi Trend Continuation
80. Triple EMA (5/13/21) Ribbon
81. Awesome Oscillator Twin Peaks
82. Fibonacci Retracement 61.8% Bounce
83. Fibonacci Extension Target Trade
84. Pivot Point Reversal (Classic/Camarilla)
85. ATR-based Volatility Breakout System (مشابه Turtle)

## ۵. رنج / RTM (Read The Market) / عرضه-تقاضا — ۱۵ مورد
86. ✅ Supply/Demand Zone Reaction — `rtm_supply_demand_zone` `[1h]`
87. ✅ Range Fade at Extremes — `range_fade_extremes` `[15m]`
88. ✅ Range Breakout + Retest — `range_breakout_retest` `[15m]`
89. ✅ Imbalance/Rebalance (RTM) — `rtm_imbalance_rebalance` `[15m]`
90. Drop-Base-Rally / Rally-Base-Drop Zone
91. Fresh vs Tested Zone Priority
92. Curved / Sloped Supply-Demand Zone
93. Multi-Timeframe Zone Confluence (RTM)
94. Extreme Zone + Momentum Confirmation
95. Compression before Expansion (رنج فشرده قبل از شکست بزرگ)
96. Mean Reversion به VWAP در رنج
97. Grid Trading در رنج مشخص
98. Range High/Low Liquidity Sweep + RTM Entry
99. Zone Overlap (Supply روی Order Block هم‌پوشان)
100. News/Session Open Range Breakout (ORB)

---

### یادداشت صادقانه
هیچ ترکیبی از این استراتژی‌ها تضمین سود نیست. عدد ۱۰۰ صرفاً پوشش گسترده
سبک‌ها را نشان می‌دهد؛ خیلی از این‌ها هم‌پوشانی مفهومی دارند (مثلاً چند
مورد ICT و SMC اساساً یک ایده با نام‌های متفاوت‌اند). قبل از ترید واقعی،
حتماً روی چند نماد/تایم‌فریم و چند دوره زمانی متفاوت بک‌تست بگیر و به
Overfitting (بیش‌برازش روی دیتای گذشته) مشکوک باش.
