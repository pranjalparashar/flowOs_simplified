"""Generate judge-friendly plots for FlowOS training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FlowOS training and evaluation metrics")
    parser.add_argument("--train-metrics", default=None, help="CSV from train_sft.py")
    parser.add_argument("--eval-metrics", default=None, help="CSV from eval.py")
    parser.add_argument("--output-dir", default="outputs/plots", help="Directory for PNG plots")
    return parser.parse_args()


def plot_training_metrics(csv_path: Path, output_dir: Path) -> list[Path]:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    outputs: list[Path] = []

    if "loss" in df.columns:
        loss_df = df[df["loss"].notna()].copy()
        if not loss_df.empty:
            x_values = loss_df["step"] if "step" in loss_df.columns else loss_df.index
            plt.figure(figsize=(8, 4))
            plt.plot(x_values, loss_df["loss"], marker="o", linewidth=2)
            plt.title("SFT Training Loss")
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.grid(True, alpha=0.3)
            output = output_dir / "sft_training_loss.png"
            plt.tight_layout()
            plt.savefig(output, dpi=180)
            plt.close()
            outputs.append(output)

    if "grad_norm" in df.columns:
        grad_df = df[df["grad_norm"].notna()].copy()
        if not grad_df.empty:
            x_values = grad_df["step"] if "step" in grad_df.columns else grad_df.index
            plt.figure(figsize=(8, 4))
            plt.plot(x_values, grad_df["grad_norm"], marker="o", linewidth=2)
            plt.title("SFT Gradient Norm")
            plt.xlabel("Training Step")
            plt.ylabel("Grad Norm")
            plt.grid(True, alpha=0.3)
            output = output_dir / "sft_grad_norm.png"
            plt.tight_layout()
            plt.savefig(output, dpi=180)
            plt.close()
            outputs.append(output)

    return outputs


def plot_eval_metrics(csv_path: Path, output_dir: Path) -> list[Path]:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    outputs: list[Path] = []
    if df.empty:
        return outputs

    grouped = df.groupby("label", as_index=False).agg(
        avg_reward=("total_reward", "mean"),
        avg_score=("score", "mean"),
        solved_rate=("solved", "mean"),
        avg_fallback_steps=("fallback_steps", "mean"),
        avg_valid_model_steps=("valid_model_steps", "mean"),
    )

    for metric, title, filename, ylabel in [
        ("avg_reward", "Average Reward by Policy", "eval_avg_reward.png", "Average Reward"),
        ("avg_score", "Average Score by Policy", "eval_avg_score.png", "Average Score"),
        ("solved_rate", "Solved Rate by Policy", "eval_solved_rate.png", "Solved Rate"),
        ("avg_fallback_steps", "Fallback Steps by Policy", "eval_fallback_steps.png", "Average Fallback Steps"),
        ("avg_valid_model_steps", "Valid Model Steps by Policy", "eval_valid_model_steps.png", "Average Valid Model Steps"),
    ]:
        plt.figure(figsize=(7, 4))
        plt.bar(grouped["label"], grouped[metric])
        plt.title(title)
        plt.xlabel("Policy")
        plt.ylabel(ylabel)
        if metric == "solved_rate":
            plt.ylim(0, 1)
        plt.grid(True, axis="y", alpha=0.3)
        output = output_dir / filename
        plt.tight_layout()
        plt.savefig(output, dpi=180)
        plt.close()
        outputs.append(output)

    return outputs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    if args.train_metrics:
        outputs.extend(plot_training_metrics(Path(args.train_metrics), output_dir))
    if args.eval_metrics:
        outputs.extend(plot_eval_metrics(Path(args.eval_metrics), output_dir))

    if outputs:
        print("Generated plots:")
        for output in outputs:
            print(output)
    else:
        print("No plots generated. Provide --train-metrics and/or --eval-metrics.")


if __name__ == "__main__":
    main()
