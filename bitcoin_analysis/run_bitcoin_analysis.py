#!/usr/bin/env python3
"""
用途：
    將論文 AMAX/SMIN 偵測邏輯套用到比特幣，支援週線與論文風格季線取樣，
    並自動輸出資料、圖表與分析報告到 `bitcoin_analysis/`。

使用方式：
    - 週線（預設）：
      python3 bitcoin_analysis/run_bitcoin_analysis.py
    - 季線（論文風格日期取樣）：
      python3 bitcoin_analysis/run_bitcoin_analysis.py --sampling quarterly

主要參數：
    --k, --m, --n            AMAX/SMIN 視窗參數
    --start-date             分析起始日期（YYYY-MM-DD）
    --monitor-start-date     開始監控日期（YYYY-MM-DD）
    --sampling               `weekly` 或 `quarterly`
    --vs-currency            報價貨幣，現階段僅支援 `usd`

輸出與副作用：
    - 寫入 `bitcoin_analysis/data/*.csv`
    - 寫入 `bitcoin_analysis/plots/*.png`
    - 寫入 `bitcoin_analysis/reports/*.md`
    - 會透過網路下載 BTCUSD 歷史資料（Stooq）

注意事項：
    - 本檔只改變取樣與資料來源，不改 AMAX/SMIN 核心偵測邏輯。
    - 由於資料源與頻率不同，結果不應直接對照論文房市日期。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sys
import io

import numpy as np
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bubble_detector import detect_bubble_amax, plot_bubble_result
from crash_detector import detect_crash_smin, plot_crash_result


DATA_DIR = Path("bitcoin_analysis/data")
PLOTS_DIR = Path("bitcoin_analysis/plots")
REPORT_DIR = Path("bitcoin_analysis/reports")
STOOQ_CSV_URL = "https://stooq.com/q/d/l/?s=btcusd&i=d"


@dataclass
class RunConfig:
    k: int = 10
    m: int = 10
    n: int = 1
    start_date: str = "2014-01-01"
    monitor_start_date: str = "2017-01-01"
    vs_currency: str = "usd"
    sampling: str = "weekly"


def ensure_dirs() -> None:
    """建立輸出資料夾，確保後續寫檔不會因路徑不存在而失敗。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_bitcoin_daily_prices(vs_currency: str = "usd") -> list[tuple[datetime, float]]:
    """
    從 Stooq 下載 BTCUSD 歷史日資料。

    參數：
        vs_currency: 報價貨幣（目前僅接受 `usd`）
    回傳：
        依時間排序的 `(datetime_utc, close_price)` 清單
    """
    if vs_currency.lower() != "usd":
        raise ValueError("Current data source provides BTCUSD only; use --vs-currency usd.")

    resp = requests.get(STOOQ_CSV_URL, timeout=60)
    resp.raise_for_status()
    rows: list[tuple[datetime, float]] = []
    reader = csv.DictReader(io.StringIO(resp.text))
    for r in reader:
        dt = datetime.fromisoformat(r["Date"] + "T00:00:00+00:00").astimezone(UTC)
        rows.append((dt, float(r["Close"])))
    rows.sort(key=lambda x: x[0])
    return rows


def write_daily_csv(rows: list[tuple[datetime, float]], output: Path) -> None:
    """將原始日資料落地成 CSV，便於追蹤與重現。"""
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["datetime_utc", "price"])
        for dt, p in rows:
            w.writerow([dt.isoformat(), f"{p:.8f}"])


def aggregate_weekly_close(rows: list[tuple[datetime, float]]) -> tuple[np.ndarray, list[str], list[str]]:
    """
    將日資料彙整成週線收盤（每 ISO 週最後可得價格）。

    回傳：
        values: 週收盤價向量
        labels: 週標籤（如 `YYYY-Www`）
        end_dates: 每週最後交易日日期（`YYYY-MM-DD`）
    """
    weekly_map: dict[tuple[int, int], tuple[datetime, float]] = {}
    for dt, price in rows:
        iso_year, iso_week, _ = dt.isocalendar()
        key = (iso_year, iso_week)
        prev = weekly_map.get(key)
        if prev is None or dt > prev[0]:
            weekly_map[key] = (dt, price)

    keys = sorted(weekly_map.keys())
    vals: list[float] = []
    labels: list[str] = []
    end_dates: list[str] = []
    for y, w in keys:
        dt, price = weekly_map[(y, w)]
        vals.append(price)
        labels.append(f"{y}-W{w:02d}")
        end_dates.append(dt.date().isoformat())
    return np.array(vals, dtype=float), labels, end_dates


def aggregate_quarterly_close(rows: list[tuple[datetime, float]]) -> tuple[np.ndarray, list[str], list[str]]:
    """
    將日資料彙整成季線收盤（每季最後可得價格）。

    回傳：
        values: 季收盤價向量
        labels: 季標籤（如 `YYYY-Qn`）
        end_dates: 每季最後交易日日期（`YYYY-MM-DD`）
    """
    q_map: dict[tuple[int, int], tuple[datetime, float]] = {}
    for dt, price in rows:
        quarter = (dt.month - 1) // 3 + 1
        key = (dt.year, quarter)
        prev = q_map.get(key)
        if prev is None or dt > prev[0]:
            q_map[key] = (dt, price)

    keys = sorted(q_map.keys())
    vals: list[float] = []
    labels: list[str] = []
    end_dates: list[str] = []
    for y, q in keys:
        dt, price = q_map[(y, q)]
        vals.append(price)
        labels.append(f"{y}-Q{q}")
        end_dates.append(dt.date().isoformat())
    return np.array(vals, dtype=float), labels, end_dates


def write_weekly_csv(values: np.ndarray, labels: list[str], end_dates: list[str], output: Path) -> None:
    """將週/季取樣後的時間序列寫入 CSV（函式名稱沿用舊命名）。"""
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["week_label", "week_end_date", "close_price"])
        for lab, d, v in zip(labels, end_dates, values):
            w.writerow([lab, d, f"{v:.8f}"])


def find_first_index_by_date(end_dates: list[str], date_s: str) -> int | None:
    """回傳第一個 `end_date >= date_s` 的 1-based 索引；找不到回傳 `None`。"""
    for i, d in enumerate(end_dates, start=1):
        if d >= date_s:
            return i
    return None


def filter_by_start(values: np.ndarray, labels: list[str], end_dates: list[str], start_date: str) -> tuple[np.ndarray, list[str], list[str]]:
    """依起始日期切資料，確保監控只在指定觀測區間內進行。"""
    idx = find_first_index_by_date(end_dates, start_date)
    if idx is None:
        raise ValueError(f"start_date={start_date} is after available data.")
    s = idx - 1
    return values[s:], labels[s:], end_dates[s:]


def build_report(
    cfg: RunConfig,
    n_obs: int,
    first_date: str,
    last_date: str,
    monitor_start: str,
    bubble_crit: float,
    crash_crit: float | None,
    bubble_detect_idx: int | None,
    bubble_detect_label: str | None,
    crash_detect_idx: int | None,
    crash_detect_label: str | None,
) -> str:
    """
    組合報告 Markdown 內容。

    參數：
        cfg: 本次執行參數
        n_obs: 取樣後觀測筆數
        first_date, last_date: 分析區間
        monitor_start: 監控起點日期
        bubble_crit, crash_crit: 兩階段臨界值
        bubble_detect_idx/crash_detect_idx: 偵測點索引（1-based，可為 None）
        bubble_detect_label/crash_detect_label: 偵測點標籤（可為 None）
    回傳：
        可直接寫入 `.md` 的完整報告字串。
    """
    bubble_line = "未偵測到泡沫" if bubble_detect_label is None else f"{bubble_detect_label}（索引 {bubble_detect_idx}）"
    crash_line = "未偵測到崩跌" if crash_detect_label is None else f"{crash_detect_label}（索引 {crash_detect_idx}）"
    crash_crit_line = "N/A（未啟動崩跌監控）" if crash_crit is None else f"{crash_crit:.4f}"

    freq_zh = "週線收盤" if cfg.sampling == "weekly" else "季線收盤"
    freq_desc = "每 ISO 週最後可得價格" if cfg.sampling == "weekly" else "每季最後可得價格"
    title = "比特幣週線泡沫/崩跌偵測報告" if cfg.sampling == "weekly" else "比特幣季線（論文風格取樣）泡沫/崩跌偵測報告"

    return f"""# {title}

## 方法
- 使用與論文一致的兩階段即時監控邏輯：
  - 第 1 階段：`AMAX(k)` 偵測泡沫開始
  - 第 2 階段：在泡沫被偵測後，用 `SMIN(m,n)` 偵測崩跌開始

## 資料
- 來源：Stooq BTCUSD 歷史價格（`vs_currency={cfg.vs_currency}`）
- 原始頻率：日資料（CSV API）
- 分析頻率：{freq_zh}（{freq_desc}）
- 分析樣本：{first_date} 到 {last_date}（共 {n_obs} 週）
- 監控起點：{monitor_start}（取第一個週末日期 >= 此日）

## 參數
- `k={cfg.k}`
- `m={cfg.m}`
- `n={cfg.n}`

## 偵測結果
- 泡沫偵測（AMAX）：{bubble_line}
- 崩跌偵測（SMIN）：{crash_line}

## 臨界值
- `A*_max = {bubble_crit:.4f}`
- `S*_min = {crash_crit_line}`

## 與論文的關係
- 一致處：使用同一套統計邏輯（AMAX/SMIN）與訓練期校準方式。
- 差異處：
  - 論文資料是美國房價租金比（季線），本分析是比特幣價格（週線）。
  - 論文是 1975Q4–2021Q1、1998Q1 開始監控；本分析為加密資產時間範圍與頻率。
  - 因資產與頻率不同，偵測日期不應直接視為「與論文數值是否一致」的比較，而是方法在新市場的應用結果。
"""


def main() -> None:
    # 解析 CLI，允許同一支腳本產出週線與季線兩套結果
    parser = argparse.ArgumentParser(description="Run AMAX/SMIN analysis on Bitcoin weekly data.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--start-date", type=str, default="2014-01-01")
    parser.add_argument("--monitor-start-date", type=str, default="2017-01-01")
    parser.add_argument("--vs-currency", type=str, default="usd")
    parser.add_argument("--sampling", choices=["weekly", "quarterly"], default="weekly")
    args = parser.parse_args()

    cfg = RunConfig(
        k=args.k,
        m=args.m,
        n=args.n,
        start_date=args.start_date,
        monitor_start_date=args.monitor_start_date,
        vs_currency=args.vs_currency,
        sampling=args.sampling,
    )
    # 初始化輸出路徑，避免中途寫檔失敗
    ensure_dirs()

    # 1) 下載與落地原始日資料
    daily = fetch_bitcoin_daily_prices(vs_currency=cfg.vs_currency)
    write_daily_csv(daily, DATA_DIR / "bitcoin_daily_prices.csv")
    # 2) 依設定轉換成週線或季線
    if cfg.sampling == "weekly":
        values, labels, end_dates = aggregate_weekly_close(daily)
    else:
        values, labels, end_dates = aggregate_quarterly_close(daily)

    # 3) 切出分析期間並保存取樣後資料
    values, labels, end_dates = filter_by_start(
        values, labels, end_dates, cfg.start_date
    )
    freq_name = "weekly" if cfg.sampling == "weekly" else "quarterly"
    write_weekly_csv(values, labels, end_dates, DATA_DIR / f"bitcoin_{freq_name}_close.csv")

    monitor_idx = find_first_index_by_date(end_dates, cfg.monitor_start_date)
    if monitor_idx is None:
        raise ValueError("monitor_start_date is after available weekly data.")
    # 依論文邏輯：訓練期終點 = 監控起點 - k
    train_end = monitor_idx - cfg.k
    train_end = max(train_end, cfg.m + cfg.n + 1)

    # 第 1 階段：泡沫偵測（AMAX）
    bubble_res = detect_bubble_amax(
        values, k=cfg.k, train_end=train_end, monitor_start=monitor_idx
    )
    bubble_idx = bubble_res["detect_e"]
    bubble_label = labels[bubble_idx - 1] if bubble_idx is not None else None

    plot_bubble_result(
        y=values,
        result=bubble_res,
        output_path=str(PLOTS_DIR / f"bitcoin_bubble_detection_{freq_name}.png"),
        periods=labels,
        true_t1=None,
        title_prefix=f"Bitcoin {freq_name} close",
    )

    # 第 2 階段：僅在泡沫被偵測到後啟動崩跌偵測（SMIN）
    crash_crit: float | None = None
    crash_idx: int | None = None
    crash_label: str | None = None
    if bubble_idx is not None:
        crash_res = detect_crash_smin(
            y=values,
            m=cfg.m,
            n=cfg.n,
            train_end=train_end,
            bubble_detect_e=bubble_idx,
        )
        crash_crit = float(crash_res["crit"])
        crash_idx = crash_res["detect_e"]
        crash_label = labels[crash_idx - 1] if crash_idx is not None else None
        plot_crash_result(
            y=values,
            bubble_detect_e=bubble_idx,
            crash_result=crash_res,
            output_path=str(PLOTS_DIR / f"bitcoin_crash_detection_{freq_name}.png"),
            periods=labels,
            paper_t1=None,
            paper_t2=None,
            title_prefix=f"Bitcoin {freq_name} close",
        )

    report = build_report(
        cfg=cfg,
        n_obs=len(values),
        first_date=end_dates[0],
        last_date=end_dates[-1],
        monitor_start=cfg.monitor_start_date,
        bubble_crit=float(bubble_res["crit"]),
        crash_crit=crash_crit,
        bubble_detect_idx=bubble_idx,
        bubble_detect_label=bubble_label,
        crash_detect_idx=crash_idx,
        crash_detect_label=crash_label,
    )
    report_name = "BTC_REPORT.md" if cfg.sampling == "weekly" else "BTC_REPORT_PAPER_STYLE.md"
    (REPORT_DIR / report_name).write_text(report, encoding="utf-8")

    print("=== Bitcoin AMAX/SMIN Analysis Complete ===")
    print(f"Sampling: {cfg.sampling}")
    print(f"Sample: {end_dates[0]} .. {end_dates[-1]}, N={len(values)}")
    print(f"Parameters: k={cfg.k}, m={cfg.m}, n={cfg.n}")
    print(f"Monitor start date: {cfg.monitor_start_date} (index {monitor_idx})")
    print(f"Bubble A*_max={bubble_res['crit']:.4f}")
    if bubble_label is None:
        print("Bubble detection: None")
    else:
        print(f"Bubble detection: {bubble_label} (index {bubble_idx})")
    if crash_crit is None:
        print("Crash monitoring: not started (bubble not detected)")
    else:
        print(f"Crash S*_min={crash_crit:.4f}")
        if crash_label is None:
            print("Crash detection: None")
        else:
            print(f"Crash detection: {crash_label} (index {crash_idx})")

    print(f"Saved: {DATA_DIR / 'bitcoin_daily_prices.csv'}")
    print(f"Saved: {DATA_DIR / f'bitcoin_{freq_name}_close.csv'}")
    print(f"Saved: {PLOTS_DIR / f'bitcoin_bubble_detection_{freq_name}.png'}")
    if crash_crit is not None:
        print(f"Saved: {PLOTS_DIR / f'bitcoin_crash_detection_{freq_name}.png'}")
    print(f"Saved: {REPORT_DIR / report_name}")


if __name__ == "__main__":
    main()
