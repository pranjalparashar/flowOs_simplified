"""Supervised fine-tuning entrypoint for FlowOS."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT training for FlowOS traces")
    parser.add_argument("--dataset-path", default="outputs/sft_traces/traces.jsonl", help="JSONL trace file")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base instruct model")
    parser.add_argument("--output-dir", default="outputs/flowos-sft", help="Checkpoint output directory")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--logging-steps", type=int, default=1, help="Logging interval")
    parser.add_argument("--save-steps", type=int, default=25, help="Checkpoint save interval")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Sequence truncation length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory",
    )
    parser.add_argument(
        "--optim",
        default="paged_adamw_8bit",
        help="Transformers Trainer optimizer name",
    )
    parser.add_argument("--metrics-file", default="training_metrics.csv", help="CSV file for trainer log history")
    parser.add_argument("--report-to", default="none", choices=("none", "tensorboard", "wandb"), help="Logging backend")
    parser.add_argument(
        "--min-trace-rank",
        type=float,
        default=0.0,
        help="Minimum episode-level trace_rank_score required to keep a trace",
    )
    parser.add_argument(
        "--top-trace-fraction",
        type=float,
        default=1.0,
        help="Keep only the top fraction of episodes ranked by trace_rank_score",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Optional cap on the number of highest-ranked episodes to keep after filtering",
    )
    return parser.parse_args()


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def with_timestamp_suffix(output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"{output_dir.rstrip('/')}-{timestamp}"


def filter_ranked_rows(
    rows: list[dict[str, Any]],
    min_trace_rank: float,
    top_trace_fraction: float,
    max_episodes: int,
) -> list[dict[str, Any]]:
    if not rows:
        return rows

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    episode_scores: dict[tuple[str, str, int], float] = {}
    for row in rows:
        episode_key = (
            str(row.get("episode_id", "")),
            str(row.get("scenario_id", "")),
            int(row.get("episode_steps", row.get("step", 0))),
        )
        grouped.setdefault(episode_key, []).append(row)
        episode_scores[episode_key] = max(
            episode_scores.get(episode_key, 0.0),
            float(row.get("trace_rank_score", 0.0)),
        )

    selected = [
        episode_key
        for episode_key, score in sorted(
            episode_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if score >= min_trace_rank
    ]
    if top_trace_fraction < 1.0:
        keep_count = max(1, int(len(selected) * top_trace_fraction))
        selected = selected[:keep_count]
    if max_episodes > 0:
        selected = selected[:max_episodes]

    selected_keys = set(selected)
    filtered: list[dict[str, Any]] = []
    for episode_key, episode_rows in grouped.items():
        if episode_key in selected_keys:
            filtered.extend(episode_rows)
    return filtered


@dataclass
class FlowOSSFTDataset:
    samples: list[dict[str, Any]]
    tokenizer: Any
    max_seq_length: int

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        prompt_text = apply_chat_template(
            self.tokenizer,
            [
                {"role": "system", "content": sample["system_prompt"]},
                {"role": "user", "content": sample["user_prompt"]},
            ],
            add_generation_prompt=True,
        )
        prompt_ids_full = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        target_ids = self.tokenizer(
            sample["target_action"],
            add_special_tokens=False,
        )["input_ids"]

        # Always preserve target tokens; truncate prompt first if needed.
        if len(target_ids) >= self.max_seq_length:
            target_ids = target_ids[: self.max_seq_length - 1]

        available_prompt_tokens = max(1, self.max_seq_length - len(target_ids))
        prompt_ids = prompt_ids_full[-available_prompt_tokens:]
        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(prompt_ids)) + target_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def summarize_truncation(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, int]:
    stats = {
        "prompt_truncated_rows": 0,
        "target_truncated_rows": 0,
        "empty_target_rows": 0,
        "max_prompt_tokens": 0,
        "max_target_tokens": 0,
    }
    for row in rows:
        prompt_text = apply_chat_template(
            tokenizer,
            [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["user_prompt"]},
            ],
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(row["target_action"], add_special_tokens=False)["input_ids"]
        stats["max_prompt_tokens"] = max(stats["max_prompt_tokens"], len(prompt_ids))
        stats["max_target_tokens"] = max(stats["max_target_tokens"], len(target_ids))
        if len(target_ids) == 0:
            stats["empty_target_rows"] += 1
        if len(target_ids) >= max_seq_length:
            stats["target_truncated_rows"] += 1
        elif len(prompt_ids) + len(target_ids) > max_seq_length:
            stats["prompt_truncated_rows"] += 1
    return stats


class SFTCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_len = max(feature["input_ids"].shape[0] for feature in features)
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - feature["input_ids"].shape[0]
            batch["input_ids"].append(
                torch.cat([feature["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            )
            batch["attention_mask"].append(
                torch.cat([feature["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            batch["labels"].append(
                torch.cat([feature["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )
        return {key: torch.stack(value) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    args.output_dir = with_timestamp_suffix(args.output_dir)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    dataset_path = Path(args.dataset_path)
    rows = load_jsonl(dataset_path)
    if not rows:
        raise SystemExit(f"No traces found at {dataset_path}")
    original_row_count = len(rows)
    rows = filter_ranked_rows(
        rows,
        min_trace_rank=args.min_trace_rank,
        top_trace_fraction=args.top_trace_fraction,
        max_episodes=args.max_episodes,
    )
    if not rows:
        raise SystemExit("All traces were filtered out. Lower --min-trace-rank or increase --top-trace-fraction.")
    if len(rows) != original_row_count:
        print(
            f"Filtered traces from {original_row_count} step examples down to {len(rows)} using ranked-episode selection."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    truncation_stats = summarize_truncation(rows, tokenizer, args.max_seq_length)
    print(
        "Truncation stats:",
        truncation_stats,
    )
    if truncation_stats["empty_target_rows"] > 0:
        raise SystemExit(
            f"Found {truncation_stats['empty_target_rows']} rows with empty target_action. Fix the trace data first."
        )

    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    bfloat16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bfloat16_supported else torch.float16

    model_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "torch_dtype": compute_dtype if torch.cuda.is_available() else torch.float32,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    train_dataset = FlowOSSFTDataset(rows, tokenizer, args.max_seq_length)
    collator = SFTCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=[] if args.report_to == "none" else [args.report_to],
        remove_unused_columns=False,
        bf16=bfloat16_supported,
        fp16=torch.cuda.is_available() and not bfloat16_supported,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=False,
        optim=args.optim,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics_path = Path(args.output_dir) / args.metrics_file
    history_rows = [row for row in trainer.state.log_history if any(key in row for key in ("loss", "train_loss", "epoch"))]
    if history_rows:
        fieldnames = sorted({key for row in history_rows for key in row.keys()})
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in history_rows:
                writer.writerow(row)
        print(f"Saved training metrics to {metrics_path}")
    print(f"Saved SFT model to {args.output_dir}")


if __name__ == "__main__":
    main()
