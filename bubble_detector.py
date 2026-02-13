#!/usr/bin/env python3
"""
用途：
    使用 AMAX(k) 即時監控統計量偵測泡沫起點，並輸出文字結果與圖表。

使用方式：
    1) 論文重現模式（預設，使用 OECD 美國房價租金比季線）：
       python3 bubble_detector.py
    2) 模擬資料模式：
       python3 bubble_detector.py --source sim --T 220 --k 10

主要參數：
    --source        資料來源，`oecd` 或 `sim`（預設 `oecd`）
    --k             AMAX 視窗長度（預設 10）
    --oecd-start    OECD 起始季度（預設 `1975-Q4`）
    --oecd-end      OECD 結束季度（預設 `2021-Q1`）
    --monitor-start 監控起點季度（預設 `1998-Q1`）

輸出與副作用：
    - 終端印出臨界值與偵測日期
    - 寫入圖檔（預設 `bubble_detection.png`）
    - 若使用 OECD 模式，會透過 HTTP 呼叫 DBnomics/OECD API

注意事項：
    - 本檔同時提供 `detect_bubble_amax` 供其他腳本重用。
    - 僅做偵測流程，不做交易策略或報酬評估。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import requests

OECD_SERIES_URL = "https://api.db.nomics.world/v22/series/OECD/HOUSE_PRICES/Q.USA.HPI_RPI?observations=1&format=json"


@dataclass
class SimParams:
    T: int = 220
    tau1: float = 0.55
    tau2: float = 0.82
    tau3: float = 0.95
    delta1: float = 0.08
    delta2: float = 0.12
    sigma: float = 0.8
    mu: float = 0.0
    seed: int = 11


def period_to_ord(period: str) -> int:
    """將 `YYYY-Qn` 字串轉為可比較大小的整數索引。"""
    year_s, q_s = period.split("-")
    q = int(q_s.replace("Q", ""))
    return int(year_s) * 4 + q


def find_period_index(periods: list[str], target: str) -> Optional[int]:
    """回傳目標季度在序列中的 1-based 索引，找不到時回傳 `None`。"""
    try:
        return periods.index(target) + 1
    except ValueError:
        return None


def fetch_oecd_us_price_to_rent(start_period: str, end_period: str) -> tuple[np.ndarray, list[str]]:
    """
    下載 OECD 美國房價租金比季線資料，並依指定季度範圍切片。

    參數：
        start_period: 起始季度（例如 `1975-Q4`）
        end_period:   結束季度（例如 `2021-Q1`）
    回傳：
        (values, periods)
        - values: np.ndarray，季度數值
        - periods: list[str]，季度標籤
    """
    resp = requests.get(OECD_SERIES_URL, timeout=30)
    resp.raise_for_status()
    doc = resp.json()["series"]["docs"][0]
    periods = np.array(doc["period"], dtype=object)
    values = np.array(doc["value"], dtype=float)

    lo = period_to_ord(start_period)
    hi = period_to_ord(end_period)
    ords = np.array([period_to_ord(p) for p in periods], dtype=int)
    mask = (ords >= lo) & (ords <= hi)

    filtered_periods = periods[mask].tolist()
    filtered_values = values[mask]
    if filtered_values.size == 0:
        raise ValueError("No OECD observations found for requested period range.")
    return filtered_values, filtered_periods


def simulate_series(params: SimParams) -> tuple[np.ndarray, int, int, int]:
    """
    產生單根→爆炸→收斂崩跌的模擬序列（供方法測試）。

    回傳：
        y, t1, t2, t3
        - y: 模擬價格序列
        - t1: 泡沫啟動位置（1-based）
        - t2: 崩跌啟動位置（1-based）
        - t3: 崩跌段結束位置（1-based）
    """
    rng = np.random.default_rng(params.seed)
    T = params.T
    eps = rng.normal(0.0, params.sigma, T)
    u = np.zeros(T)
    u[0] = rng.normal(0.0, params.sigma)

    t1 = int(np.floor(params.tau1 * T))
    t2 = int(np.floor(params.tau2 * T))
    t3 = int(np.floor(params.tau3 * T))
    t1 = max(2, min(t1, T - 3))
    t2 = max(t1 + 2, min(t2, T - 2))
    t3 = max(t2 + 1, min(t3, T))

    for t in range(1, T):
        tt = t + 1
        if tt <= t1:
            phi = 1.0
        elif tt <= t2:
            phi = 1.0 + params.delta1
        elif tt <= t3:
            phi = 1.0 - params.delta2
        else:
            phi = 1.0
        u[t] = phi * u[t - 1] + eps[t]

    y = params.mu + u
    return y, t1, t2, t3


def a_stat(y: np.ndarray, e: int, k: int) -> float:
    """
    計算 AMAX 使用的單點統計量 `A_{e,k}`。

    參數：
        y: 價格序列
        e: 視窗終點（1-based）
        k: 視窗長度
    回傳：
        單一浮點統計量，值越大代表爆炸性訊號越強。
    """
    dy = np.diff(y)
    t_start = e - k + 1
    t_end = e
    window = dy[t_start - 2 : t_end - 1]
    if window.size != k:
        raise ValueError("Invalid window length for A statistic.")
    w = np.arange(1, k + 1, dtype=float)
    numer = np.sum(w * window)
    denom = np.sqrt(np.sum((w * window) ** 2))
    return 0.0 if denom == 0.0 else float(numer / denom)


def detect_bubble_amax(
    y: np.ndarray, k: int, train_end: int, monitor_start: Optional[int] = None
) -> dict:
    """
    以 AMAX(k) 執行泡沫即時監控。

    參數：
        y: 價格序列
        k: 視窗長度
        train_end: 訓練期終點 T*（1-based，含該點）
        monitor_start: 監控起點（1-based）；若為 None，使用 train_end + k
    回傳：
        dict，包含訓練統計量、監控統計量、臨界值與首次偵測點。
    """
    T = y.size
    if monitor_start is None:
        monitor_start = train_end + k
    if monitor_start > T:
        raise ValueError("Monitoring start is beyond series length.")
    if train_end < k + 1:
        raise ValueError("Training sample too short for chosen k.")

    train_es = np.arange(k + 1, train_end + 1)
    train_stats = np.array([a_stat(y, int(e), k) for e in train_es], dtype=float)
    amax_crit = float(np.max(train_stats))

    monitor_es = np.arange(monitor_start, T + 1)
    monitor_stats = np.array([a_stat(y, int(e), k) for e in monitor_es], dtype=float)
    hits = monitor_es[monitor_stats > amax_crit]
    detect_e = int(hits[0]) if hits.size else None

    return {
        "crit": amax_crit,
        "train_es": train_es,
        "train_stats": train_stats,
        "monitor_es": monitor_es,
        "monitor_stats": monitor_stats,
        "detect_e": detect_e,
    }


def _set_quarter_ticks(ax: plt.Axes, periods: list[str], step: int = 20) -> None:
    """將季度標籤放到 x 軸，避免圖表只顯示整數索引。"""
    idx = np.arange(1, len(periods) + 1)
    tick_pos = idx[::step]
    tick_lab = [periods[i - 1] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha="right")


def plot_bubble_result(
    y: np.ndarray,
    result: dict,
    output_path: str,
    periods: Optional[list[str]] = None,
    true_t1: Optional[int] = None,
    title_prefix: str = "",
) -> None:
    """
    繪製泡沫偵測圖（上：序列；下：AMAX 監控統計量）。

    備註：
        `true_t1` 僅作對照線使用，不會影響偵測流程。
    """
    train_end = int(result["train_es"][-1])
    mon_e = result["monitor_es"]
    mon_stats = result["monitor_stats"]
    detect_e = result["detect_e"]
    crit = result["crit"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)

    t = np.arange(1, y.size + 1)
    axes[0].plot(t, y, color="steelblue", lw=1.6, label="y_t")
    axes[0].axvline(train_end, color="gray", ls=":", lw=1.2, label="Training end")
    if true_t1 is not None:
        axes[0].axvline(true_t1, color="darkgreen", ls="--", lw=1.2, label="Paper bubble date")
    if detect_e is not None:
        axes[0].axvline(detect_e, color="crimson", ls="-.", lw=1.4, label="Detected bubble")
    axes[0].set_title(f"{title_prefix} Series and Bubble Onset")
    axes[0].set_ylabel("y_t")
    axes[0].legend(loc="best")

    axes[1].plot(mon_e, mon_stats, color="black", lw=1.2, label="A_{e,k} (monitoring)")
    axes[1].axhline(crit, color="crimson", ls="--", lw=1.2, label="A*_max (critical value)")
    if detect_e is not None:
        axes[1].axvline(detect_e, color="crimson", ls="-.", lw=1.2)
    axes[1].set_title("AMAX(k) Monitoring Statistic")
    axes[1].set_xlabel("time index (e)")
    axes[1].set_ylabel("A_{e,k}")
    axes[1].legend(loc="best")

    if periods is not None:
        _set_quarter_ticks(axes[0], periods)
        _set_quarter_ticks(axes[1], periods)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    # 解析 CLI 參數：支援 OECD 重現與模擬資料兩種入口
    parser = argparse.ArgumentParser(description="Bubble detection with AMAX(k).")
    parser.add_argument("--source", choices=["oecd", "sim"], default="oecd")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=str, default="bubble_detection.png")

    parser.add_argument("--oecd-start", type=str, default="1975-Q4")
    parser.add_argument("--oecd-end", type=str, default="2021-Q1")
    parser.add_argument("--monitor-start", type=str, default="1998-Q1")

    parser.add_argument("--T", type=int, default=220)
    parser.add_argument("--tau1", type=float, default=0.55)
    parser.add_argument("--tau2", type=float, default=0.82)
    parser.add_argument("--tau3", type=float, default=0.95)
    parser.add_argument("--delta1", type=float, default=0.08)
    parser.add_argument("--delta2", type=float, default=0.12)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-ratio", type=float, default=0.45)
    args = parser.parse_args()

    if args.source == "oecd":
        # 論文重現流程：先載入 OECD 資料，再以訓練期最大值作為監控門檻
        y, periods = fetch_oecd_us_price_to_rent(args.oecd_start, args.oecd_end)
        monitor_start_idx = find_period_index(periods, args.monitor_start)
        if monitor_start_idx is None:
            raise ValueError(f"monitor-start {args.monitor_start} not found in selected OECD sample")

        train_end = monitor_start_idx - args.k
        result = detect_bubble_amax(y, k=args.k, train_end=train_end, monitor_start=monitor_start_idx)

        paper_t1_idx = find_period_index(periods, "2000-Q1")
        detect_e = result["detect_e"]

        print("=== Bubble Detection (AMAX, OECD data) ===")
        print(f"Series: Q.USA.HPI_RPI, sample={args.oecd_start}..{args.oecd_end}, N={len(y)}")
        print(f"k={args.k}, monitor_start={args.monitor_start}, train_end(T*) index={train_end}")
        print(f"Critical value A*_max={result['crit']:.4f}")
        print("Paper bubble detection benchmark: 2000-Q1")
        if detect_e is None:
            print("Detected bubble start: None")
        else:
            det_period = periods[detect_e - 1]
            print(f"Detected bubble start: e={detect_e} ({det_period})")
            if paper_t1_idx is not None:
                print(f"Difference vs paper 2000-Q1: {detect_e - paper_t1_idx} quarters")

        plot_bubble_result(
            y=y,
            result=result,
            output_path=args.output,
            periods=periods,
            true_t1=paper_t1_idx,
            title_prefix="OECD US price-to-rent",
        )
    else:
        # 模擬流程：產生帶有已知 regime 的序列，方便檢查偵測延遲
        params = SimParams(
            T=args.T,
            tau1=args.tau1,
            tau2=args.tau2,
            tau3=args.tau3,
            delta1=args.delta1,
            delta2=args.delta2,
            sigma=args.sigma,
            seed=args.seed,
        )
        y, t1, _, _ = simulate_series(params)
        train_end = int(np.floor(args.train_ratio * args.T))
        train_end = min(train_end, t1)
        train_end = max(train_end, args.k + 1)

        result = detect_bubble_amax(y, k=args.k, train_end=train_end)
        detect_e = result["detect_e"]

        print("=== Bubble Detection (AMAX, Simulated data) ===")
        print(f"T={args.T}, k={args.k}, train_end(T*)={train_end}")
        print(f"True bubble_start(t1)={t1}")
        print(f"Critical value A*_max={result['crit']:.4f}")
        if detect_e is None:
            print("Detected bubble start: None")
        else:
            print(f"Detected bubble start at e={detect_e}, delay={detect_e - t1}")

        plot_bubble_result(y=y, result=result, output_path=args.output, true_t1=t1, title_prefix="Simulated")

    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
