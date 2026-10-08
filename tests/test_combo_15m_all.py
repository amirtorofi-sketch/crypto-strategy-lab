name: Backtest 15m - all strategies (manual)

# فقط دستی اجرا می‌شود: Actions → "Backtest 15m - all strategies" → Run workflow.
#
# چه می‌کند:
#   - ۳۶ نماد پرنقدینگی (backtest/combos.py: LIQUID_SYMBOLS)
#   - ۲۰ استراتژی (همه‌ی استراتژی‌های ثبت‌شده منهای EXCLUDED_STRATEGIES) همگی روی تایم‌فریم ۱۵ دقیقه
#   - هر پنج سشن، هر سشن مستقل (برای مقایسه‌ی منصفانه‌ی سشن‌ها)
#   - بازه‌ی پیش‌فرض ۳۶۵ روز
#
# چرا matrix؟ یک job گیت‌هاب حداکثر ۶ ساعت اجرا می‌شود؛ کل کار حدود ۱۷ ساعت CPU است. نمادها در ۶ تکه
# (هر تکه ۶ نماد) روی ۶ job موازی اجرا می‌شوند و job آخر نتیجه‌ها را ادغام و گزارش را می‌سازد.
# اگر یکی از تکه‌ها شکست بخورد، گزارش ساخته نمی‌شود (تا نتیجه‌ی ناقص با کامل اشتباه گرفته نشود).
on:
  workflow_dispatch:
    inputs:
      days:
        description: "تعداد روز تاریخچه"
        default: "365"
      fee_pct:
        description: "کارمزد هر طرف (٪) — تبدیل سطح ۱: 0.35 ؛ ستون «Exp قبل از کارمزد» در گزارش برای هر کارمزدی قابل استفاده است"
        default: "0.04"
      slippage_pct:
        description: "اسلیپیج هر طرف (٪)"
        default: "0"

permissions:
  contents: write

concurrency:
  group: backtest-15m-all
  cancel-in-progress: false

env:
  SHARDS: "6"

defaults:
  run:
    shell: bash          # شامل pipefail: خطای python پشت tee گم نمی‌شود

jobs:
  simulate:
    name: simulate (shard ${{ matrix.shard }}/6)
    runs-on: ubuntu-latest
    timeout-minutes: 340
    strategy:
      fail-fast: false
      matrix:
        shard: [1, 2, 3, 4, 5, 6]
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: pip install pandas numpy PyYAML ccxt requests

      - name: Download klines (this shard's symbols)
        env:
          DAYS: ${{ inputs.days }}
        run: |
          mkdir -p shards
          python -m backtest.download_klines --days "$DAYS" --shard "${{ matrix.shard }}/$SHARDS" 2>&1 | tee "shards/download_report_${{ matrix.shard }}.txt"

      - name: Simulate (15m, all strategies, all sessions)
        env:
          FEE_PCT: ${{ inputs.fee_pct }}
          SLIPPAGE_PCT: ${{ inputs.slippage_pct }}
        run: |
          python -m backtest.combo_backtest \
            --all-strategies --timeframe 15m --shard "${{ matrix.shard }}/$SHARDS" \
            --workers "$(nproc)" --fee-pct "$FEE_PCT" --slippage-pct "$SLIPPAGE_PCT" \
            --trades-out "shards/shard_${{ matrix.shard }}.pkl.gz" 2>&1 | tee "shards/log_${{ matrix.shard }}.txt"

      - uses: actions/upload-artifact@v4
        with:
          name: shard-${{ matrix.shard }}
          path: shards/
          if-no-files-found: error
          retention-days: 7

  report:
    name: merge + report
    needs: simulate
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: pip install pandas numpy PyYAML ccxt requests

      - uses: actions/download-artifact@v4
        with:
          pattern: shard-*
          path: shards
          merge-multiple: true

      - name: Merge shards and build report
        run: |
          OUT=results/backtest/combos_15m_all
          python -m backtest.combo_backtest --merge shards/shard_*.pkl.gz --out "$OUT" 2>&1 | tee merge_log.txt
          cat shards/download_report_*.txt > "$OUT/download_report.txt"
          # فایل معاملات حجیم است؛ داخل ریپو commit نمی‌شود و فقط در Artifact می‌ماند
          mkdir -p heavy
          mv "$OUT/trades.csv.gz" heavy/trades.csv.gz

      - name: Show summary on the run page
        if: always()
        run: |
          if [ -f results/backtest/combos_15m_all/summary.md ]; then cat results/backtest/combos_15m_all/summary.md >> "$GITHUB_STEP_SUMMARY"; fi

      - uses: actions/upload-artifact@v4
        with:
          name: backtest-15m-all-results
          path: |
            results/backtest/combos_15m_all/
            heavy/trades.csv.gz
            merge_log.txt
          if-no-files-found: warn

      - name: Commit results
        run: |
          git config user.name "trading-bot"
          git config user.email "bot@users.noreply.github.com"
          git add -A results/backtest/combos_15m_all
          git diff --quiet --cached || git commit -m "chore: update 15m all-strategies backtest results [skip ci]"
          for i in 1 2 3; do
            git pull --rebase origin "${GITHUB_REF_NAME}" && git push && break
            sleep 5
          done
