# 論文重現分析（AMAX / SMIN）

本文件說明如何用本專案程式重現 Whitehouse et al. (2023) 的偵測流程，並比較結果是否與論文一致。

## 1. 執行環境與資料
- Python 3.10+
- 套件：`numpy`、`matplotlib`、`requests`
- 資料來源（與論文一致）：
  - OECD `HOUSE_PRICES` 資料集
  - 系列代碼：`Q.USA.HPI_RPI`（US house price-to-rent ratio）
  - 樣本期間：`1975-Q4` 到 `2021-Q1`
  - 監控起點：`1998-Q1`

安裝：
```bash
pip install numpy matplotlib requests
```

## 2. 如何跑出結果
### 2.1 泡沫開始偵測（AMAX）
```bash
python3 bubble_detector.py
```

### 2.2 崩跌開始偵測（AMAX -> SMIN）
```bash
python3 crash_detector.py
```

### 2.3 驗證 n=2 的延遲效果
```bash
python3 crash_detector.py --n 2
```

## 3. 本次實跑結果
### 3.1 AMAX（泡沫）
- 偵測結果：`2000-Q1`
- 論文基準：`2000-Q1`
- 差異：`0` 季（完全一致）

### 3.2 SMIN（崩跌，n=1）
- 偵測結果：`2006-Q2`
- 論文基準：`2006-Q2`
- 差異：`0` 季（完全一致）

### 3.3 SMIN（崩跌，n=2）
- 偵測結果：`2006-Q3`
- 論文敘述：`n>1` 會有最多 `n-1` 季延遲；文中對 `n=2/3` 觀察為 `2006-Q3`
- 差異：相對 `2006-Q2` 晚 `1` 季（與論文敘述一致）

## 4. 結論
在預設真實資料設定下，本專案的結果與論文關鍵日期一致：
- 泡沫開始：`2000-Q1`
- 崩跌開始（`n=1`）：`2006-Q2`
- 崩跌開始（`n=2`）：`2006-Q3`

因此目前程式已可重現論文核心實證結論與 `n` 參數帶來的偵測延遲特性。
