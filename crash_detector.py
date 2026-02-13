#!/usr/bin/env python3
"""
用途：
    在泡沫已被 AMAX 偵測後，使用 SMIN(m,n) 即時監控崩跌起點。

使用方式：
    1) 論文重現模式（預設）：
       python3 crash_detector.py
    2) 模擬資料模式：
       python3 crash_detector.py --source sim --k 10 --m 10 --n 1

主要參數：
    --k             泡沫監控 AMAX 視窗長度
    --m, --n        崩跌監控 SMIN 前後視窗長度
    --monitor-start OECD 模式的監控起點季度
    --source        `oecd` 或 `sim`

輸出與副作用：
    - 終端印出泡沫/崩跌偵測日期、臨界值與和論文基準差異
    - 寫入圖檔（預設 `crash_detection.png`）
    - OECD 模式會呼叫網路 API 載入資料

注意事項：
    - SMIN 監控只在泡沫被偵測後才會啟動。
    - `n` 越大通常可降低誤報，但可能增加偵測延遲。
"""

from __future__ import annotations

import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from bubble_detector import (
    SimParams,
    detect_bubble_amax,
    fetch_oecd_us_price_to_rent,
    find_period_index,
    simulate_series,
)


def _ols_residuals_delta_on_const_lag_y(y: np.ndarray, t_start: int, t_end: int) -> np.ndarray:
    """
    估計 `Δy_t ~ const + y_(t-1)` 的 OLS 殘差，用於 SMIN 標準化分母。

    參數：
        y: 價格序列
        t_start, t_end: 迴歸樣本範圍（1-based，含端點）
    回傳：
        殘差向量（np.ndarray）
    """
    dy = np.diff(y)
    idx = np.arange(t_start, t_end + 1)
    dep = dy[idx - 2]
    lag_y = y[idx - 2]
    X = np.column_stack([np.ones(dep.size), lag_y])
    beta, *_ = np.linalg.lstsq(X, dep, rcond=None)
    resid = dep - X @ beta
    return resid


def s_stat(y: np.ndarray, e: int, m: int, n: int) -> float:
    """
    計算崩跌監控統計量 `S_{e,m,n}`。

    參數：
        y: 價格序列
        e: 當前監控終點（1-based）
        m: 變化點前視窗長度
        n: 變化點後視窗長度
    回傳：
        浮點統計量；值越小越傾向崩跌訊號。
    """
    dy = np.diff(y)
    left_start = e - n - m + 1
    left_end = e - n
    right_start = e - n + 1
    right_end = e

    left_delta = dy[left_start - 2 : left_end - 1]
    right_delta = dy[right_start - 2 : right_end - 1]
    if left_delta.size != m or right_delta.size != n:
        raise ValueError("Invalid window lengths for S statistic.")

    resid_left = _ols_residuals_delta_on_const_lag_y(y, left_start, left_end)
    left_sum = float(np.sum(left_delta))
    right_sum = float(np.sum(right_delta))
    denom = np.sqrt(float(np.sum(resid_left**2)) * float(np.sum(right_delta**2)))
    if denom == 0.0:
        return 0.0
    return (left_sum * right_sum) / denom


def detect_crash_smin(
    y: np.ndarray,
    m: int,
    n: int,
    train_end: int,
    bubble_detect_e: int,
    crash_monitor_start: Optional[int] = None,
) -> dict:
    """
    在已知泡沫偵測點之後，執行 SMIN(m,n) 即時崩跌監控。

    參數：
        y: 價格序列
        m, n: SMIN 視窗參數
        train_end: 訓練期終點 T*（1-based）
        bubble_detect_e: 泡沫偵測點（1-based）
        crash_monitor_start: 崩跌監控起點；None 時使用 bubble_detect_e + 1
    回傳：
        dict，包含訓練/監控統計量、臨界值與首次崩跌偵測點。
    """
    T = y.size
    min_e = m + n + 1
    if train_end < min_e:
        raise ValueError("Training sample too short for chosen m,n.")

    if crash_monitor_start is None:
        crash_monitor_start = bubble_detect_e + 1
    crash_monitor_start = max(crash_monitor_start, min_e)
    if crash_monitor_start > T:
        raise ValueError("Crash monitoring start is beyond series length.")

    train_es = np.arange(min_e, train_end + 1)
    train_stats = np.array([s_stat(y, int(e), m, n) for e in train_es], dtype=float)
    smin_crit = float(np.min(train_stats))

    monitor_es = np.arange(crash_monitor_start, T + 1)
    monitor_stats = np.array([s_stat(y, int(e), m, n) for e in monitor_es], dtype=float)
    hits = monitor_es[monitor_stats < smin_crit]
    detect_e = int(hits[0]) if hits.size else None

    return {
        "crit": smin_crit,
        "train_es": train_es,
        "train_stats": train_stats,
        "monitor_es": monitor_es,
        "monitor_stats": monitor_stats,
        "detect_e": detect_e,
    }


def _set_quarter_ticks(ax: plt.Axes, periods: list[str], step: int = 20) -> None:
    """在季度資料圖上放置可讀的季度刻度。"""
    idx = np.arange(1, len(periods) + 1)
    tick_pos = idx[::step]
    tick_lab = [periods[i - 1] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha="right")


def plot_crash_result(
    y: np.ndarray,
    bubble_detect_e: Optional[int],
    crash_result: dict,
    output_path: str,
    periods: Optional[list[str]] = None,
    paper_t1: Optional[int] = None,
    paper_t2: Optional[int] = None,
    title_prefix: str = "",
) -> None:
    """
    繪製崩跌偵測圖（上：序列；下：SMIN 監控統計量）。

    備註：
        `paper_t1/paper_t2` 只用於視覺對照，不參與偵測運算。
    """
    mon_e = crash_result["monitor_es"]
    mon_stats = crash_result["monitor_stats"]
    detect_e = crash_result["detect_e"]
    crit = crash_result["crit"]
    train_end = int(crash_result["train_es"][-1])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    t = np.arange(1, y.size + 1)

    axes[0].plot(t, y, color="steelblue", lw=1.6, label="y_t")
    axes[0].axvline(train_end, color="gray", ls=":", lw=1.2, label="Training end")
    if paper_t1 is not None:
        axes[0].axvline(paper_t1, color="darkgreen", ls="--", lw=1.2, label="Paper bubble date")
    if paper_t2 is not None:
        axes[0].axvline(paper_t2, color="orange", ls="--", lw=1.2, label="Paper crash date")
    if bubble_detect_e is not None:
        axes[0].axvline(bubble_detect_e, color="purple", ls="-.", lw=1.2, label="Detected bubble")
    if detect_e is not None:
        axes[0].axvline(detect_e, color="crimson", ls="-.", lw=1.4, label="Detected crash")
    axes[0].set_title(f"{title_prefix} Series with Bubble and Crash Detection")
    axes[0].set_ylabel("y_t")
    axes[0].legend(loc="best")

    axes[1].plot(mon_e, mon_stats, color="black", lw=1.2, label="S_{e,m,n} (monitoring)")
    axes[1].axhline(crit, color="crimson", ls="--", lw=1.2, label="S*_min (critical value)")
    if detect_e is not None:
        axes[1].axvline(detect_e, color="crimson", ls="-.", lw=1.2)
    axes[1].set_title("SMIN(m,n) Monitoring Statistic")
    axes[1].set_xlabel("time index (e)")
    axes[1].set_ylabel("S_{e,m,n}")
    axes[1].legend(loc="best")

    if periods is not None:
        _set_quarter_ticks(axes[0], periods)
        _set_quarter_ticks(axes[1], periods)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    # 解析 CLI 參數：維持與 bubble_detector.py 一致的操作體驗
    parser = argparse.ArgumentParser(description="Crash detection with SMIN(m,n).")
    parser.add_argument("--source", choices=["oecd", "sim"], default="oecd")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--output", type=str, default="crash_detection.png")

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
        # 論文重現流程：先做 AMAX，再以其偵測點啟動 SMIN
        y, periods = fetch_oecd_us_price_to_rent(args.oecd_start, args.oecd_end)
        monitor_start_idx = find_period_index(periods, args.monitor_start)
        if monitor_start_idx is None:
            raise ValueError(f"monitor-start {args.monitor_start} not found in selected OECD sample")

        train_end = monitor_start_idx - args.k
        train_end = max(train_end, args.m + args.n + 1)

        bubble_result = detect_bubble_amax(y, k=args.k, train_end=train_end, monitor_start=monitor_start_idx)
        bubble_detect_e = bubble_result["detect_e"]

        print("=== Bubble + Crash Detection (AMAX -> SMIN, OECD data) ===")
        print(f"Series: Q.USA.HPI_RPI, sample={args.oecd_start}..{args.oecd_end}, N={len(y)}")
        print(f"k={args.k}, m={args.m}, n={args.n}, monitor_start={args.monitor_start}, train_end(T*) index={train_end}")
        print(f"Bubble critical A*_max={bubble_result['crit']:.4f}")

        paper_t1_idx = find_period_index(periods, "2000-Q1")
        paper_t2_idx = find_period_index(periods, "2006-Q2")

        if bubble_detect_e is None:
            print("Bubble was not detected; crash monitoring not started.")
            return

        bubble_period = periods[bubble_detect_e - 1]
        print(f"Detected bubble: e={bubble_detect_e} ({bubble_period})")
        if paper_t1_idx is not None:
            print(f"Difference vs paper bubble 2000-Q1: {bubble_detect_e - paper_t1_idx} quarters")

        crash_result = detect_crash_smin(
            y=y,
            m=args.m,
            n=args.n,
            train_end=train_end,
            bubble_detect_e=bubble_detect_e,
        )
        crash_detect_e = crash_result["detect_e"]
        print(f"Crash critical S*_min={crash_result['crit']:.4f}")

        if crash_detect_e is None:
            print("Detected crash: None")
        else:
            crash_period = periods[crash_detect_e - 1]
            print(f"Detected crash: e={crash_detect_e} ({crash_period})")
            if paper_t2_idx is not None:
                print(f"Difference vs paper crash 2006-Q2: {crash_detect_e - paper_t2_idx} quarters")

        plot_crash_result(
            y=y,
            bubble_detect_e=bubble_detect_e,
            crash_result=crash_result,
            output_path=args.output,
            periods=periods,
            paper_t1=paper_t1_idx,
            paper_t2=paper_t2_idx,
            title_prefix="OECD US price-to-rent",
        )
    else:
        # 模擬流程：用已知真值檢查泡沫/崩跌偵測延遲
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
        y, t1, t2, _ = simulate_series(params)

        train_end = int(np.floor(args.train_ratio * args.T))
        train_end = min(train_end, t1)
        train_end = max(train_end, max(args.k + 1, args.m + args.n + 1))

        bubble_result = detect_bubble_amax(y, k=args.k, train_end=train_end)
        bubble_detect_e = bubble_result["detect_e"]

        print("=== Bubble + Crash Detection (AMAX -> SMIN, Simulated data) ===")
        print(f"T={args.T}, k={args.k}, m={args.m}, n={args.n}, train_end(T*)={train_end}")
        print(f"True bubble_start={t1}, true crash_start={t2}")
        print(f"Bubble critical A*_max={bubble_result['crit']:.4f}")

        if bubble_detect_e is None:
            print("Bubble was not detected; crash monitoring not started.")
            return

        print(f"Detected bubble at e={bubble_detect_e}, delay={bubble_detect_e - t1}")

        crash_result = detect_crash_smin(
            y=y,
            m=args.m,
            n=args.n,
            train_end=train_end,
            bubble_detect_e=bubble_detect_e,
        )
        crash_detect_e = crash_result["detect_e"]
        print(f"Crash critical S*_min={crash_result['crit']:.4f}")
        if crash_detect_e is None:
            print("Detected crash: None")
        else:
            print(f"Detected crash at e={crash_detect_e}, delay={crash_detect_e - t2}")

        plot_crash_result(
            y=y,
            bubble_detect_e=bubble_detect_e,
            crash_result=crash_result,
            output_path=args.output,
            paper_t1=t1,
            paper_t2=t2,
            title_prefix="Simulated",
        )

    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
