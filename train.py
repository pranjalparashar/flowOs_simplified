"""Minimal GRPO training entrypoint for FlowOS."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from datetime import datetime
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
        parse_sample_prompt,
        persist_episode_artifacts,
        run_episode,
        samples_to_dataset_prompts,
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
        parse_sample_prompt,
        persist_episode_artifacts,
        run_episode,
        samples_to_dataset_prompts,
    )

logger = logging.getLogger(__name__)
DEBUG_GENERATION_LOG_LIMIT = 8


def _clean_preview(text: str, limit: int = 220) -> str:
    normalized = text.replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return "<empty>"
    if len(normalized) > limit:
        return normalized[:limit] + "..."
    return normalized


def with_timestamp_suffix(output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"{output_dir.rstrip('/')}-{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Colab-first GRPO training for FlowOS")
    parser.add_argument("--model-id", default=DEFAULT_TRAIN_MODEL, help="Base instruct model to fine-tune")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional local or Hub PEFT checkpoint to continue GRPO from instead of starting from the base model",
    )
    parser.add_argument("--env-url", default="http://localhost:7860", help="FlowOS OpenEnv server URL")
    parser.add_argument("--dataset-size", type=int, default=24, help="Number of training episode prompts")
    parser.add_argument("--num-generations", type=int, default=4, help="GRPO generations per prompt")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Max environment turns per episode")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max tokens per action generation")
    parser.add_argument("--learning-rate", type=float, default=5e-7, help="Trainer learning rate")
    parser.add_argument("--output-dir", default=None, help="Checkpoint output directory")
    parser.add_argument("--task-scope", default="all", help="Task scope: all or comma-separated task ids")
    parser.add_argument("--report-to", default="none", choices=("none", "tensorboard", "wandb"), help="Logging backend")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of GRPO epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max trainer steps; -1 lets TRL infer from epochs")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--save-steps", type=int, default=10, help="Checkpoint save interval")
    parser.add_argument("--logging-steps", type=int, default=1, help="Logging interval")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--log-step-rewards",
        action="store_true",
        help="Log each rollout step's action, reward, and feedback during GRPO training",
    )
    parser.add_argument(
        "--rollout-backend",
        default="manual",
        choices=("manual", "trl"),
        help="Generation backend for GRPO rollouts. 'manual' uses model.generate directly.",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--reward-log", default="reward_log.csv", help="CSV filename for episode rewards")
    parser.add_argument(
        "--fallback-mode",
        default="warmup",
        choices=("always", "warmup", "never"),
        help="How invalid model actions are handled during GRPO rollouts",
    )
    parser.add_argument(
        "--fallback-warmup-episodes",
        type=int,
        default=4,
        help="Number of early episodes that may use fallback when --fallback-mode=warmup",
    )
    parser.add_argument("--upload-repo-id", default=None, help="Optional Hugging Face repo id for uploading outputs")
    parser.add_argument(
        "--upload-repo-type",
        default="dataset",
        choices=("dataset", "model"),
        help="Repo type for --upload-repo-id",
    )
    parser.add_argument(
        "--upload-path-in-repo",
        default=None,
        help="Optional path prefix inside the upload repo (defaults to output dir name)",
    )
    return parser.parse_args()


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


def generate_with_trainer(
    trainer: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    rollout_backend: str,
) -> dict[str, Any]:
    if rollout_backend == "trl":
        try:
            from trl.experimental.openenv import generate_rollout_completions

            episode = generate_rollout_completions(trainer, [prompt_text], max_new_tokens=max_new_tokens)[0]
            return {
                "prompt_ids": list(episode.get("prompt_ids", [])),
                "completion_ids": list(episode.get("completion_ids", [])),
                "logprobs": list(episode.get("logprobs", [])),
                "text": episode.get("text")
                or tokenizer.decode(episode.get("completion_ids", []), skip_special_tokens=True),
            }
        except Exception as exc:
            logger.warning("TRL rollout generation failed; falling back to manual generation: %s", exc)

    import torch

    model = getattr(trainer, "model_wrapped", None) or getattr(trainer, "model", None)
    if model is None:
        raise RuntimeError("Could not access trainer model for fallback generation.")

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

    prompt_ids = inputs["input_ids"][0].tolist()
    full_ids = generated[0]
    completion_ids = full_ids[len(prompt_ids) :].tolist()

    with torch.no_grad():
        logits = model(full_ids.unsqueeze(0)).logits[0]
        token_logprobs: list[float] = []
        for offset, token_id in enumerate(completion_ids):
            logit_index = len(prompt_ids) - 1 + offset
            distribution = torch.log_softmax(logits[logit_index], dim=-1)
            token_logprobs.append(float(distribution[token_id].item()))

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": token_logprobs,
        "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
    }


def reward_total(completions: list[str], **kwargs: Any) -> list[float]:
    rewards = kwargs.get("total_reward", [])
    return [float(value) for value in rewards] if rewards else [0.0] * len(completions)


def reward_format(completions: list[str], **kwargs: Any) -> list[float]:
    rewards = kwargs.get("format_reward", [])
    return [float(value) for value in rewards] if rewards else [0.0] * len(completions)


def reward_valid_action(completions: list[str], **kwargs: Any) -> list[float]:
    rewards = kwargs.get("valid_action_reward", [])
    return [float(value) for value in rewards] if rewards else [0.0] * len(completions)


def reward_score(completions: list[str], **kwargs: Any) -> list[float]:
    rewards = kwargs.get("score_reward", [])
    return [float(value) for value in rewards] if rewards else [0.0] * len(completions)


def reward_solved(completions: list[str], **kwargs: Any) -> list[float]:
    solved = kwargs.get("solved_reward", [])
    return [float(value) for value in solved] if solved else [0.0] * len(completions)


def upload_outputs(output_dir: Path, repo_id: str, repo_type: str, path_in_repo: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(output_dir),
        path_in_repo=path_in_repo or output_dir.name,
    )


def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        args.output_dir = with_timestamp_suffix(args.output_dir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    try:
        from datasets import Dataset
        from peft import AutoPeftModelForCausalLM, LoraConfig
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them with `pip install -e \".[train]\"`."
        ) from exc

    samples = build_episode_samples(args.task_scope, args.dataset_size)
    dataset = Dataset.from_dict({"prompt": samples_to_dataset_prompts(samples)})

    tokenizer_source = args.init_checkpoint or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = Path(args.output_dir or Path("outputs") / f"flowos-grpo-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_log_path = output_dir / args.reward_log
    artifact_output_dir = output_dir / "artifacts" / "sim_runs"
    with reward_log_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "task_id", "scenario_id", "total_reward", "score", "solved", "steps"])

    episode_counter = [0]
    generation_debug_counter = [0]

    def log_episode(metrics: Any) -> None:
        episode_counter[0] += 1
        with reward_log_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    episode_counter[0],
                    metrics.task_id,
                    metrics.scenario_id,
                    f"{metrics.total_reward:.4f}",
                    f"{metrics.score:.4f}",
                    str(bool(metrics.solved)).lower(),
                    metrics.steps,
                ]
            )
        logger.info(
            "Episode %s task=%s scenario=%s reward=%.3f score=%.3f solved=%s steps=%s",
            episode_counter[0],
            metrics.task_id,
            metrics.scenario_id,
            metrics.total_reward,
            metrics.score,
            metrics.solved,
            metrics.steps,
        )
        if args.log_step_rewards:
            for step_idx, entry in enumerate(metrics.transcript, start=1):
                logger.info(
                    "Episode %s step=%s reward=%.3f action=%s feedback=%s error=%s",
                    episode_counter[0],
                    step_idx,
                    float(entry.get("reward", 0.0)),
                    entry.get("action_text", "invalid_action()"),
                    entry.get("feedback") or "none",
                    entry.get("error") or "null",
                )
        artifact_dir = persist_episode_artifacts(artifact_output_dir, metrics)
        if artifact_dir is not None:
            logger.info("Saved episode artifacts to %s", artifact_dir)

    def rollout_func(prompts: list[str], trainer: Any) -> dict[str, list]:
        prompt_batches: list[list[int]] = []
        completion_batches: list[list[int]] = []
        logprob_batches: list[list[float]] = []
        total_rewards: list[float] = []
        score_rewards: list[float] = []
        solved_rewards: list[float] = []
        format_rewards: list[float] = []
        valid_action_rewards: list[float] = []

        for prompt in prompts:
            sample = parse_sample_prompt(prompt)
            planned_episode_number = episode_counter[0] + len(total_rewards) + 1

            def policy(observation: Any, transcript: list[dict[str, Any]]) -> dict[str, Any]:
                user_prompt = build_turn_prompt(observation, transcript)
                prompt_text = apply_chat_template(
                    tokenizer,
                    [
                        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                generation = generate_with_trainer(
                    trainer=trainer,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    rollout_backend=args.rollout_backend,
                )
                parsed_action = parse_action_json(generation["text"])
                json_format_valid = parsed_action is not None
                valid_action = action_is_valid(parsed_action, observation)
                if generation_debug_counter[0] < DEBUG_GENERATION_LOG_LIMIT:
                    logger.info(
                        "Raw rollout completion %s task=%s step=%s text=%s parsed=%s",
                        generation_debug_counter[0] + 1,
                        sample.task_id,
                        observation.step_count + 1,
                        _clean_preview(generation["text"]),
                        parsed_action,
                    )
                    generation_debug_counter[0] += 1
                allow_fallback = args.fallback_mode == "always" or (
                    args.fallback_mode == "warmup" and planned_episode_number <= args.fallback_warmup_episodes
                )
                if not valid_action and allow_fallback:
                    fallback = fallback_action(sample.task_id, observation)
                    logger.info(
                        "Using fallback action task=%s step=%s fallback=%s",
                        sample.task_id,
                        observation.step_count + 1,
                        fallback,
                    )
                    parsed_action = fallback
                    used_fallback = True
                elif not valid_action:
                    logger.info(
                        "Invalid action penalized task=%s step=%s fallback_mode=%s",
                        sample.task_id,
                        observation.step_count + 1,
                        args.fallback_mode,
                    )
                    used_fallback = False
                else:
                    used_fallback = False
                return {
                    "action": parsed_action,
                    "metadata": {
                        **generation,
                        "used_fallback": used_fallback,
                        "model_action_valid": valid_action,
                        "json_format_valid": json_format_valid,
                    },
                }

            metrics = run_episode(
                base_url=args.env_url,
                sample=sample,
                max_turns=args.max_turns,
                policy=policy,
            )

            prompt_batches.append(metrics.prompt_ids or [])
            completion_batches.append(metrics.completion_ids or [])
            logprob_batches.append(metrics.logprobs or [])
            total_rewards.append(metrics.total_reward)
            if metrics.steps > 0:
                json_fraction = metrics.json_format_steps / metrics.steps
                valid_fraction = metrics.valid_model_steps / metrics.steps
            else:
                json_fraction = 0.0
                valid_fraction = 0.0
            format_rewards.append(3.0 if json_fraction == 1.0 else -3.0 * (1.0 - json_fraction))
            valid_action_rewards.append((2.0 * valid_fraction) - 1.0)
            score_rewards.append(metrics.score)
            solved_rewards.append(1.0 if metrics.solved else 0.0)
            log_episode(metrics)

        return {
            "prompt_ids": prompt_batches,
            "completion_ids": completion_batches,
            "logprobs": logprob_batches,
            "total_reward": total_rewards,
            "format_reward": format_rewards,
            "valid_action_reward": valid_action_rewards,
            "score_reward": score_rewards,
            "solved_reward": solved_rewards,
        }

    trainer_model: Any
    peft_config: LoraConfig | None
    if args.init_checkpoint:
        trainer_model = AutoPeftModelForCausalLM.from_pretrained(
            args.init_checkpoint,
            is_trainable=True,
            low_cpu_mem_usage=True,
        )
        trainer_model.config.use_cache = False
        peft_config = None
    else:
        trainer_model = args.model_id
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        use_vllm=False,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to=args.report_to,
        temperature=args.temperature,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = GRPOTrainer(
        model=trainer_model,
        processing_class=tokenizer,
        reward_funcs=[reward_format, reward_valid_action, reward_total, reward_score, reward_solved],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
        peft_config=peft_config,
    )
    trainer_model_ref = getattr(trainer, "model", None)
    if trainer_model_ref is not None and getattr(trainer_model_ref, "config", None) is not None:
        trainer_model_ref.config.use_cache = False

    logger.info("Starting FlowOS GRPO training")
    logger.info(
        "model=%s init_checkpoint=%s env=%s dataset_size=%s task_scope=%s",
        args.model_id,
        args.init_checkpoint,
        args.env_url,
        args.dataset_size,
        args.task_scope,
    )
    try:
        trainer.train()
    finally:
        try:
            try:
                from .plot_rewards import plot as plot_reward_curve
            except ImportError:
                from plot_rewards import plot as plot_reward_curve
            plot_reward_curve(reward_log_path, output_dir / "reward_curve.png")
            logger.info("Reward curve written to %s", output_dir / "reward_curve.png")
        except Exception as exc:  # pragma: no cover - plotting should be best effort
            logger.warning("Could not generate reward curve: %s", exc)

    trainer.save_model(str(output_dir))
    logger.info("Training finished. Model saved to %s", output_dir)
    logger.info("Reward log written to %s", reward_log_path)

    if args.upload_repo_id:
        try:
            upload_outputs(
                output_dir=output_dir,
                repo_id=args.upload_repo_id,
                repo_type=args.upload_repo_type,
                path_in_repo=args.upload_path_in_repo,
            )
            logger.info(
                "Uploaded outputs to https://huggingface.co/%s/%s",
                args.upload_repo_type + "s" if args.upload_repo_type != "model" else "",
                args.upload_repo_id,
            )
        except Exception as exc:  # pragma: no cover - depends on external auth/network
            logger.warning("Could not upload outputs: %s", exc)


if __name__ == "__main__":
    main()
