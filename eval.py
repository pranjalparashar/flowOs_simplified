"""Evaluate base and fine-tuned FlowOS policies against the OpenEnv server."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
from pathlib import Path
from typing import Any

try:
    from .baseline import action_is_valid, fallback_action
    from .training_utils import (
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_MAX_TURNS,
        DEFAULT_TRAIN_MODEL,
        TRAIN_SYSTEM_PROMPT,
        build_episode_samples,
        build_turn_prompt,
        parse_action_json,
        run_episode,
    )
except ImportError:
    from baseline import action_is_valid, fallback_action
    from training_utils import (
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_MAX_TURNS,
        DEFAULT_TRAIN_MODEL,
        TRAIN_SYSTEM_PROMPT,
        build_episode_samples,
        build_turn_prompt,
        parse_action_json,
        run_episode,
    )

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FlowOS policies before and after GRPO fine-tuning")
    parser.add_argument("--model-id", default=DEFAULT_TRAIN_MODEL, help="Base model id")
    parser.add_argument("--checkpoint-path", default=None, help="Optional fine-tuned checkpoint or LoRA adapter path")
    parser.add_argument(
        "--backend",
        default="transformers",
        choices=("transformers", "unsloth"),
        help="Model loading backend for evaluation",
    )
    parser.add_argument("--env-url", default="http://localhost:7860", help="FlowOS OpenEnv server URL")
    parser.add_argument("--task-scope", default="all", help="Task scope: all or comma-separated task ids")
    parser.add_argument("--episodes", type=int, default=12, help="Total evaluation episodes per policy")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Max environment turns per episode")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max tokens per action generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for evaluation")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantized loading when supported")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length for model loading")
    parser.add_argument("--debug-actions", action="store_true", help="Log raw completions and fallback actions")
    parser.add_argument("--disable-fallback", action="store_true", help="Evaluate the raw policy without fallback rescue")
    parser.add_argument("--results-dir", default=None, help="Optional directory to write eval metrics and summaries")
    return parser.parse_args()


def _clean_preview(text: str, limit: int = 220) -> str:
    normalized = text.replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return "<empty>"
    if len(normalized) > limit:
        return normalized[:limit] + "..."
    return normalized


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def load_policy(
    model_id: str,
    checkpoint_path: str | None,
    backend: str,
    load_in_4bit: bool,
    max_seq_length: int,
) -> tuple[Any, Any, str]:
    if backend == "unsloth":
        from unsloth import FastLanguageModel

        model_name = checkpoint_path or model_id
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        label = "tuned" if checkpoint_path else "base"
        return model, tokenizer, label

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path or model_id)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    label = "base"
    if checkpoint_path:
        try:
            from peft import AutoPeftModelForCausalLM

            model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path)
            label = "tuned"
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
            label = "tuned"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, label


def generate_action(
    model: Any,
    tokenizer: Any,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[dict[str, Any] | None, str]:
    import torch

    prompt_text = apply_chat_template(
        tokenizer,
        [
            {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = generated[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return parse_action_json(text), text


def summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "avg_reward": statistics.fmean(result["total_reward"] for result in results) if results else 0.0,
        "avg_score": statistics.fmean(result["score"] for result in results) if results else 0.0,
        "solved_rate": (
            sum(1 for result in results if result["solved"]) / len(results) if results else 0.0
        ),
        "avg_steps": statistics.fmean(result["steps"] for result in results) if results else 0.0,
    }


def run_policy(
    label: str,
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    samples = build_episode_samples(args.task_scope, args.episodes)
    results: list[dict[str, Any]] = []
    for sample in samples:
        def policy(observation: Any, transcript: list[dict[str, Any]]) -> dict[str, Any]:
            user_prompt = build_turn_prompt(observation, transcript)
            action, raw_text = generate_action(
                model=model,
                tokenizer=tokenizer,
                user_prompt=user_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            if args.debug_actions:
                logger.info(
                    "Eval completion task=%s step=%s text=%s parsed=%s",
                    sample.task_id,
                    observation.step_count + 1,
                    _clean_preview(raw_text),
                    action,
                )
            model_action_valid = action_is_valid(action, observation)
            used_fallback = False
            if not model_action_valid and not args.disable_fallback:
                fallback = fallback_action(sample.task_id, observation)
                used_fallback = True
                if args.debug_actions:
                    logger.info(
                        "Eval fallback task=%s step=%s fallback=%s",
                        sample.task_id,
                        observation.step_count + 1,
                        fallback,
                    )
                action = fallback
            return {
                "action": action,
                "metadata": {
                    "text": raw_text,
                    "used_fallback": used_fallback,
                    "model_action_valid": model_action_valid,
                },
            }

        metrics = run_episode(
            base_url=args.env_url,
            sample=sample,
            max_turns=args.max_turns,
            policy=policy,
        )
        results.append(
            {
                "label": label,
                "task_id": metrics.task_id,
                "scenario_id": metrics.scenario_id,
                "total_reward": metrics.total_reward,
                "score": metrics.score,
                "solved": metrics.solved,
                "steps": metrics.steps,
                "fallback_steps": metrics.fallback_steps,
                "valid_model_steps": metrics.valid_model_steps,
            }
        )
    return results


def write_results(results_dir: str | None, rows: list[dict[str, Any]]) -> None:
    if not results_dir:
        return
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_episode_path = output_dir / "eval_episode_metrics.csv"
    summary_path = output_dir / "eval_summary.json"

    if rows:
        fieldnames = list(rows[0].keys())
        with per_episode_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)
    for label, group in grouped.items():
        aggregate = summarize(group)
        aggregate["policy"] = label
        aggregate["episodes"] = len(group)
        aggregate["avg_fallback_steps"] = statistics.fmean(item["fallback_steps"] for item in group)
        aggregate["avg_valid_model_steps"] = statistics.fmean(item["valid_model_steps"] for item in group)
        summary_rows.append(aggregate)

    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote eval metrics to %s", per_episode_path)
    logger.info("Wrote eval summary to %s", summary_path)


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ("policy", "avg_reward", "avg_score", "solved_rate", "avg_steps")
    line = "{:<10} {:>11} {:>10} {:>12} {:>10}"
    print(line.format(*headers))
    print(line.format(*("-" * len(header) for header in headers)))
    for row in rows:
        print(
            line.format(
                row["policy"],
                f"{row['avg_reward']:.3f}",
                f"{row['avg_score']:.3f}",
                f"{row['solved_rate']:.2%}",
                f"{row['avg_steps']:.2f}",
            )
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base_model, base_tokenizer, base_label = load_policy(
        args.model_id,
        None,
        args.backend,
        args.load_in_4bit,
        args.max_seq_length,
    )
    rows: list[dict[str, Any]] = []
    base_results = run_policy(base_label, base_model, base_tokenizer, args)
    rows.append({"policy": base_label, **summarize(base_results)})

    if args.checkpoint_path:
        tuned_model, tuned_tokenizer, tuned_label = load_policy(
            args.model_id,
            args.checkpoint_path,
            args.backend,
            args.load_in_4bit,
            args.max_seq_length,
        )
        tuned_results = run_policy(tuned_label, tuned_model, tuned_tokenizer, args)
        rows.append({"policy": tuned_label, **summarize(tuned_results)})
        base_results.extend(tuned_results)
    write_results(args.results_dir, base_results)

    print_table(rows)


if __name__ == "__main__":
    main()
