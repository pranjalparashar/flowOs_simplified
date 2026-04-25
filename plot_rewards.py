#!/usr/bin/env python3
"""Plot GRPO training rewards from reward_log.csv in the SF winner style."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def find_latest_csv() -> Path | None:
    csvs = sorted(Path("outputs").glob("*/reward_log.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if csvs:
        return csvs[0]
    if Path("reward_log.csv").exists():
        return Path("reward_log.csv")
    return None


def load_csv(path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    episodes: list[int] = []
    totals: list[float] = []
    scores: list[float] = []
    solved: list[float] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("episode"):
                continue
            episodes.append(int(row["episode"]))
            totals.append(float(row.get("total_reward", 0.0)))
            scores.append(float(row.get("score", 0.0)))
            solved.append(1.0 if str(row.get("solved", "")).lower() == "true" else 0.0)
    return episodes, totals, scores, solved


def rolling_avg(values: list[float], window: int = 10) -> list[float]:
    if not values:
        return []
    window = min(window, len(values))
    return [sum(values[max(0, i - window + 1): i + 1]) / min(i + 1, window) for i in range(len(values))]


def plot(path: Path, save_path: Path | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes, totals, scores, solved = load_csv(path)
    if not episodes:
        print("No data yet.")
        return

    window = min(10, len(episodes))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(episodes, totals, alpha=0.25, color="blue", marker="o", markersize=3, label="Per episode reward")
    ax1.plot(episodes, rolling_avg(totals, window), color="blue", linewidth=2.5, label=f"Rolling avg ({window})")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Total Reward")
    ax1.set_title(f"FlowOS — GRPO Training Rewards ({len(episodes)} episodes)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, rolling_avg(scores, window), color="orange", linewidth=2, label="Score (rolling)")
    ax2.plot(episodes, rolling_avg(solved, window), color="green", linewidth=2, label="Solved rate (rolling)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Metric")
    ax2.set_ylim(min(0.0, min(scores + solved, default=0.0)) - 0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_path or path.with_suffix(".png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")
    print(f"\nEpisodes: {len(episodes)}")
    print(f"Latest reward:  {totals[-1]:.2f}")
    print(f"Avg (last 10):  {sum(totals[-10:]) / min(10, len(totals)):.2f}")
    print(f"Best reward:    {max(totals):.2f}")
    print(f"Worst reward:   {min(totals):.2f}")


def print_table(path: Path) -> None:
    episodes, totals, scores, solved = load_csv(path)
    if not episodes:
        print("No data yet.")
        return

    print(f"\n{'Ep':>4} | {'Reward':>8} | {'Score':>7} | {'Solved':>6} | {'Avg(10)':>8}")
    print("-" * 47)
    for i in range(len(episodes)):
        avg10 = sum(totals[max(0, i - 9): i + 1]) / min(i + 1, 10)
        marker = " *" if totals[i] == max(totals[: i + 1]) else ""
        print(f"{episodes[i]:>4} | {totals[i]:>8.2f} | {scores[i]:>7.2f} | {solved[i]:>6.0f} | {avg10:>8.2f}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FlowOS GRPO rewards")
    parser.add_argument("csv_path", nargs="?", help="Path to reward_log.csv")
    parser.add_argument("--live", action="store_true", help="Refresh every 30 seconds")
    parser.add_argument("--table", action="store_true", help="Print ASCII table instead of plot")
    parser.add_argument("--out", default=None, help="Output image path")
    args = parser.parse_args()

    path = Path(args.csv_path) if args.csv_path else find_latest_csv()
    if not path or not path.exists():
        print("No reward_log.csv found. Run training first or specify path.")
        sys.exit(1)

    print(f"Reading: {path}")
    if args.table:
        print_table(path)
        return

    if args.live:
        while True:
            try:
                plot(path, Path(args.out) if args.out else None)
                time.sleep(30)
            except KeyboardInterrupt:
                break
    else:
        plot(path, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
