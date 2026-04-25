"""Collect supervised traces for FlowOS without GRPO."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

try:
    from .baseline import fallback_action
    from .tasks import get_scenario
    from .training_utils import (
        TRAIN_SYSTEM_PROMPT,
        build_episode_samples,
        build_turn_prompt,
        coerce_action,
        format_action_text,
        parse_sample_prompt,
        persist_episode_artifacts,
        samples_to_dataset_prompts,
    )
    from .client import DeveloperControlRoomEnv
except ImportError:
    from baseline import fallback_action
    from tasks import get_scenario
    from training_utils import (
        TRAIN_SYSTEM_PROMPT,
        build_episode_samples,
        build_turn_prompt,
        coerce_action,
        format_action_text,
        parse_sample_prompt,
        persist_episode_artifacts,
        samples_to_dataset_prompts,
    )
    from client import DeveloperControlRoomEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect fallback/teacher traces for FlowOS SFT")
    parser.add_argument("--env-url", default="http://localhost:7860", help="FlowOS OpenEnv server URL")
    parser.add_argument("--task-scope", default="all", help="Task scope: all or comma-separated task ids")
    parser.add_argument("--dataset-size", type=int, default=8, help="Number of episodes to collect")
    parser.add_argument("--max-turns", type=int, default=12, help="Max environment turns per episode")
    parser.add_argument("--output-dir", default="outputs/sft_traces", help="Directory for JSONL traces")
    parser.add_argument("--policy", default="fallback", choices=("fallback",), help="Teacher policy to use")
    parser.add_argument(
        "--exclude-curriculum",
        action="store_true",
        help="Do not mix in the easy curriculum tasks when collecting traces for the main simulation workflow",
    )
    return parser.parse_args()


def _expanded_task_scope(task_scope: str, include_curriculum: bool) -> str:
    if not include_curriculum:
        return task_scope
    requested = [task.strip() for task in task_scope.split(",") if task.strip()] if task_scope != "all" else ["all"]
    if "all" in requested:
        return task_scope
    if "simulate_csv_report_workflow" not in requested:
        return task_scope
    expanded = list(requested)
    for extra_task in (
        "simulate_csv_report_curriculum_generate",
        "simulate_csv_report_curriculum_repair",
    ):
        if extra_task not in expanded:
            expanded.append(extra_task)
    return ",".join(expanded)


def _quality_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _compute_trace_rank(
    sample: Any,
    transcript: list[dict[str, Any]],
    observation: Any,
) -> tuple[float, dict[str, float]]:
    scenario = get_scenario(sample.task_id, sample.scenario_index, sample.seed)
    simulation_target = scenario.get("simulation_target", {}) or scenario.get("llm_draft", {}).get("simulation_target", {})
    runtime_status = dict(getattr(observation, "runtime_status", {}))
    checks = dict(runtime_status.get("checks", {}))
    available_validators = list(getattr(observation, "available_validators", []))
    validator_status = dict(getattr(observation, "validator_status", {}))
    editable_targets = list(getattr(observation, "editable_targets", []))
    edited_files = dict(getattr(observation, "edited_files", {}))

    expected_columns = list(simulation_target.get("required_output_columns", []))
    output_schema = list(getattr(observation, "output_schema", []))
    schema_score = 1.0 if expected_columns and output_schema == expected_columns else 0.0

    passed_checks = sum(1 for passed in checks.values() if passed)
    total_checks = len(checks)
    checks_fraction = passed_checks / total_checks if total_checks else 0.0
    errors = list(runtime_status.get("errors", []))
    if runtime_status.get("succeeded", False):
        status_hierarchy = 1.0
    elif checks.get("report_view_check", False):
        status_hierarchy = 0.8
    elif checks.get("duckdb_load_check", False):
        status_hierarchy = 0.6
    elif checks.get("storage_stage_check", False) or runtime_status.get("ran", False):
        status_hierarchy = 0.35 + 0.15 * checks_fraction
    elif errors:
        status_hierarchy = 0.15
    else:
        status_hierarchy = 0.0

    validator_passes = sum(1 for name in available_validators if validator_status.get(name, {}).get("passed"))
    validator_score = validator_passes / len(available_validators) if available_validators else 0.0

    edited_targets = {entry.get("path", "") for entry in transcript if entry.get("action_type") == "edit_file"}
    edited_targets.update(path for path in editable_targets if path in edited_files)
    artifact_score = len(edited_targets & set(editable_targets)) / len(editable_targets) if editable_targets else 0.0

    first_edit_index = next(
        (idx for idx, entry in enumerate(transcript) if entry.get("action_type") == "edit_file"),
        len(transcript),
    )
    investigation_actions = {"read_file", "inspect_schema", "inspect_lineage", "inspect_llm_draft"}
    investigations_before_edit = sum(
        1 for entry in transcript[:first_edit_index] if entry.get("action_type") in investigation_actions
    )
    investigation_score = min(1.0, investigations_before_edit / 2.0)
    validators_after_edit = sum(
        1
        for entry in transcript[first_edit_index:]
        if entry.get("action_type") == "run_validator"
    )
    validator_process_score = min(1.0, validators_after_edit / 2.0)
    duplicate_penalty = 0.0
    recent_signatures: list[tuple[str, str]] = []
    for entry in transcript:
        signature = (entry.get("action_type", ""), entry.get("path", "") or entry.get("validator", "") or "")
        recent_signatures.append(signature)
        if len(recent_signatures) >= 3 and len(set(recent_signatures[-3:])) == 1:
            duplicate_penalty = 0.3
            break
    process_score = max(0.0, 0.55 * investigation_score + 0.45 * validator_process_score - duplicate_penalty)

    cumulative_reward = float(getattr(observation, "cumulative_reward", 0.0))
    reward_signal = min(1.0, max(0.0, (cumulative_reward + 1.25) / 2.25))

    breakdown = {
        "status_hierarchy": round(status_hierarchy, 4),
        "schema_score": round(schema_score, 4),
        "artifact_score": round(artifact_score, 4),
        "validator_score": round(validator_score, 4),
        "process_score": round(process_score, 4),
        "reward_signal": round(reward_signal, 4),
    }
    trace_rank_score = round(
        0.35 * status_hierarchy
        + 0.15 * schema_score
        + 0.15 * artifact_score
        + 0.10 * validator_score
        + 0.15 * process_score
        + 0.10 * reward_signal,
        4,
    )
    return trace_rank_score, breakdown


async def collect_episode(
    env_url: str,
    sample_prompt: str,
    max_turns: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample = parse_sample_prompt(sample_prompt)
    env = DeveloperControlRoomEnv(base_url=env_url)
    await env.connect()
    transcript: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    try:
        result = await env.reset(
            task_id=sample.task_id,
            scenario_index=sample.scenario_index,
            seed=sample.seed,
        )
        observation = result.observation

        for _ in range(max_turns):
            if result.done:
                break
            prompt = build_turn_prompt(observation, transcript)
            action_dict = fallback_action(sample.task_id, observation)
            action = coerce_action(action_dict)
            action_json = json.dumps(action_dict, separators=(",", ":"), ensure_ascii=False)
            examples.append(
                {
                    "task_id": sample.task_id,
                    "scenario_id": observation.scenario_id,
                    "episode_id": observation.episode_id,
                    "step": observation.step_count + 1,
                    "system_prompt": TRAIN_SYSTEM_PROMPT,
                    "user_prompt": prompt,
                    "target_action": action_json,
                    "action_type": action.action_type,
                    "editable_targets": observation.editable_targets,
                    "available_validators": observation.available_validators,
                }
            )

            result = await env.step(action)
            observation = result.observation
            transcript.append(
                {
                    "action_type": action.action_type,
                    "path": action.parameters.path,
                    "validator": action.parameters.validator,
                    "action_text": format_action_text(action),
                    "reward": float(result.reward or 0.0),
                    "feedback": observation.feedback,
                    "error": observation.last_action_error,
                    "raw_model_text": "",
                }
            )
            if result.done:
                break

        trace_rank_score, trace_rank_breakdown = _compute_trace_rank(sample, transcript, observation)
        quality_label = _quality_label(trace_rank_score)
        metrics = {
            "task_id": sample.task_id,
            "scenario_id": observation.scenario_id,
            "episode_id": observation.episode_id,
            "steps": len(transcript),
            "cumulative_reward": float(getattr(observation, "cumulative_reward", 0.0)),
            "trace_rank_score": trace_rank_score,
            "trace_rank_breakdown": trace_rank_breakdown,
            "trace_quality_label": quality_label,
            "validator_status": dict(getattr(observation, "validator_status", {})),
            "available_validators": list(getattr(observation, "available_validators", [])),
            "editable_targets": list(getattr(observation, "editable_targets", [])),
            "edited_files": dict(getattr(observation, "edited_files", {})),
            "active_role": getattr(observation, "active_role", "builder"),
            "materialized_artifacts": dict(getattr(observation, "materialized_artifacts", {})),
            "runtime_status": dict(getattr(observation, "runtime_status", {})),
            "output_schema": list(getattr(observation, "output_schema", [])),
            "report_preview": list(getattr(observation, "report_preview", [])),
        }
        for example in examples:
            example["episode_steps"] = metrics["steps"]
            example["episode_total_reward"] = metrics["cumulative_reward"]
            example["trace_rank_score"] = trace_rank_score
            example["trace_rank_breakdown"] = trace_rank_breakdown
            example["trace_quality_label"] = quality_label
        return examples, metrics
    finally:
        await env.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "traces.jsonl"
    episodes_path = output_dir / "episodes.jsonl"
    artifact_root = output_dir / "artifacts"

    prompts = samples_to_dataset_prompts(
        build_episode_samples(_expanded_task_scope(args.task_scope, not args.exclude_curriculum), args.dataset_size)
    )

    all_examples: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    for prompt in prompts:
        examples, metrics = asyncio.run(collect_episode(args.env_url, prompt, args.max_turns))
        all_examples.extend(examples)
        all_episodes.append(metrics)
        if metrics["materialized_artifacts"] or metrics["runtime_status"]:
            class _Metrics:
                def __init__(self, payload: dict[str, Any]) -> None:
                    self.task_id = payload["task_id"]
                    self.scenario_id = payload["scenario_id"]
                    self.steps = payload["steps"]
                    self.materialized_artifacts = payload["materialized_artifacts"]
                    self.runtime_status = payload["runtime_status"]
                    self.output_schema = payload["output_schema"]
                    self.report_preview = payload["report_preview"]

            persist_episode_artifacts(artifact_root, _Metrics(metrics))

    with traces_path.open("w", encoding="utf-8") as handle:
        for row in all_examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with episodes_path.open("w", encoding="utf-8") as handle:
        for row in all_episodes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_examples)} supervised examples to {traces_path}")
    print(f"Wrote {len(all_episodes)} episode summaries to {episodes_path}")
    print(f"Artifacts stored under {artifact_root}")


if __name__ == "__main__":
    main()
