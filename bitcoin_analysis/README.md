# Bitcoin Analysis 說明

本資料夾使用與論文相同的偵測邏輯（`AMAX(k)` + `SMIN(m,n)`），但資料改為**比特幣週線**。

## 執行方式
在專案根目錄執行：

```bash
python3 bitcoin_analysis/run_bitcoin_analysis.py
```

可調參數（範例）：

```bash
python3 bitcoin_analysis/run_bitcoin_analysis.py --k 10 --m 10 --n 2 --start-date 2014-01-01 --monitor-start-date 2017-01-01
```

論文風格日期取樣（季線）：

```bash
python3 bitcoin_analysis/run_bitcoin_analysis.py --sampling quarterly
```

## 主要輸出
- `bitcoin_analysis/data/bitcoin_daily_prices.csv`：CoinGecko 抓回的 BTC 日資料。
- `bitcoin_analysis/data/bitcoin_daily_prices.csv`：Stooq 抓回的 BTC 日資料。
- `bitcoin_analysis/data/bitcoin_weekly_close.csv`：轉為週線收盤後的資料。
- `bitcoin_analysis/data/bitcoin_quarterly_close.csv`：轉為季線收盤後的資料（使用 `--sampling quarterly` 時）。
- `bitcoin_analysis/plots/bitcoin_bubble_detection_weekly.png`：AMAX 週線泡沫偵測圖。
- `bitcoin_analysis/plots/bitcoin_crash_detection_weekly.png`：SMIN 週線崩跌偵測圖。
- `bitcoin_analysis/plots/bitcoin_bubble_detection_quarterly.png`：AMAX 季線泡沫偵測圖。
- `bitcoin_analysis/plots/bitcoin_crash_detection_quarterly.png`：SMIN 季線崩跌偵測圖。
- `bitcoin_analysis/reports/BTC_REPORT.md`：本次參數、偵測日期、結果解讀與與論文差異說明。
- `bitcoin_analysis/reports/BTC_REPORT_PAPER_STYLE.md`：季線（論文風格取樣）結果報告。

## 方法對照
- 相同：使用 AMAX/SMIN 同一套即時監控判定邏輯與訓練期臨界值機制。
- 不同：資產從美國房價租金比（季線）改為比特幣（週線），因此偵測時間點反映 BTC 市場特性。
