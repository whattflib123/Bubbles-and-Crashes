# Whitehouse et al. (2023) 論文筆記

## 這篇論文是什麼？
這篇論文是：
**Whitehouse, Harvey, Leybourne (2023), _Real-Time Monitoring of Bubbles and Crashes_, Oxford Bulletin of Economics and Statistics.**

核心問題是：當資產價格泡沫已經出現時，如何在「即時（real-time）」資料更新下，盡快偵測泡沫是否開始崩跌（crash）。

## 核心貢獻
- 提出兩階段即時監控架構：  
  1. 先監控是否進入泡沫（爆炸性動態）  
  2. 一旦偵測到泡沫，再監控是否轉為崩跌（平穩收斂動態）
- 崩跌檢定統計量利用「一階差分平均數符號差異」來區分泡沫與崩跌狀態。
- 以訓練期資料建立臨界值，控制連續監控下的誤報率（false positive rate）。
- 理論與模擬顯示：可快速偵測崩跌，且不會在崩跌真正發生前提前誤判。

## 方法直觀
- 泡沫期：價格序列呈爆炸性自我增長（explosive AR）。
- 崩跌期：泡沫結束後轉為平穩收斂（stationary AR）。
- 監控流程會隨新資料點到來反覆更新檢定值，屬於序列式即時偵測。

## 論文中的實證結果
- 應用在美國房價租金比（US house price-to-rent ratio）的擬即時監控：
  - 偵測到泡沫：**2000Q1**
  - 偵測到崩跌：**2006Q2**
- 顯示方法可對 2000 年代中期美國房市反轉提供相對即時的警訊。

## 為什麼重要？
- 對政策制定者與風險管理者，重點不只是在事後辨識泡沫，而是在崩跌開始時盡快收到可操作訊號。
- 此方法特別處理了 real-time 多次檢定問題，避免「一直測就一直誤報」的偏誤。

## 專案內容
- `Whitehouse et al(2023)Real‐Time Monitoring of Bubbles and Crashes_OBES.pdf`：原始論文全文。
- `bubble_detector.py`：以 `AMAX(k)` 模擬並偵測泡沫開始。
- `crash_detector.py`：先用 `AMAX(k)` 偵測泡沫，再用 `SMIN(m,n)` 偵測崩跌開始。

## 使用說明
### 1) 安裝需求
建議使用 Python 3.10+，並安裝套件：

```bash
pip install numpy matplotlib requests
```

### 2) 執行泡沫開始偵測
```bash
python3 bubble_detector.py
```

執行後會輸出：
- 終端文字：論文基準日期（2000-Q1）與實際偵測季度、臨界值
- 圖檔：`bubble_detection.png`

預設資料為論文同來源：
- OECD `Q.USA.HPI_RPI`（US house price-to-rent ratio）
- 期間：`1975-Q4` 到 `2021-Q1`
- 監控起點：`1998-Q1`

### 3) 執行崩跌開始偵測
```bash
python3 crash_detector.py
```

執行後會輸出：
- 終端文字：泡沫/崩跌偵測季度，及與論文基準（2000-Q1、2006-Q2）差異
- 圖檔：`crash_detection.png`

### 4) 常用參數
- `--k`：AMAX 視窗長度（泡沫監控）
- `--m --n`：SMIN 前後視窗長度（崩跌監控）
- `--oecd-start --oecd-end`：樣本期間（季度格式，例如 `1975-Q4`）
- `--monitor-start`：開始即時監控的季度（預設 `1998-Q1`）
- `--source sim`：切回合成資料模式（研究方法時可用）

範例：
```bash
python3 crash_detector.py --k 10 --m 10 --n 2
```

## 參考文獻
Whitehouse, E. J., Harvey, D. I., & Leybourne, S. J. (2023). Real-Time Monitoring of Bubbles and Crashes. *Oxford Bulletin of Economics and Statistics*, 85(3), 482-521. https://doi.org/10.1111/obes.12540
