"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    API_KEY        The injected API key for the provided LLM proxy.
    ENV_URL        The deployed environment URL to connect to.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from typing import Any, List, Optional

from openai import OpenAI

from baseline import action_is_valid, build_action, fallback_action, fetch_grader_result, get_model_action
from client import DeveloperControlRoomEnv
from tasks import list_tasks

DEFAULT_ENV_URL = "https://praanjal-control-room.hf.space"
# DEFAULT_ENV_URL = "http://localhost:7860"
ENV_URL = os.getenv("ENV_URL") or DEFAULT_ENV_URL
API_KEY = os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("DEVELOPER_CONTROL_ROOM_TASK")
BENCHMARK = os.getenv("DEVELOPER_CONTROL_ROOM_BENCHMARK", "developer_control_room")
MAX_STEPS = int(os.getenv("DEVELOPER_CONTROL_ROOM_MAX_STEPS", "14"))
DEFAULT_MODEL_STEPS = 14
MAX_MODEL_STEPS = int(
    os.getenv("DEVELOPER_CONTROL_ROOM_MAX_MODEL_STEPS", str(DEFAULT_MODEL_STEPS))
)
RUN_ALL_SCENARIOS = os.getenv("DEVELOPER_CONTROL_ROOM_RUN_ALL_SCENARIOS", "false").lower() == "true"
SCENARIO_INDEX_RAW = os.getenv("DEVELOPER_CONTROL_ROOM_SCENARIO_INDEX")
SCENARIO_SEED_RAW = os.getenv("DEVELOPER_CONTROL_ROOM_SCENARIO_SEED")
SCENARIO_INDEX = int(SCENARIO_INDEX_RAW) if SCENARIO_INDEX_RAW not in (None, "") else None
SCENARIO_SEED = int(SCENARIO_SEED_RAW) if SCENARIO_SEED_RAW not in (None, "") else None
TEMPERATURE = 0.0
MAX_TOKENS = 250
MODEL_TIMEOUT_SECONDS = float(os.getenv("DEVELOPER_CONTROL_ROOM_MODEL_TIMEOUT_SECONDS", "20"))
DEBUG_LOGGING = os.getenv("DEVELOPER_CONTROL_ROOM_DEBUG", "false").lower() == "true"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = _single_line(error) if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={_single_line(action)} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    bounded_score = min(0.99, max(0.01, score))
    print(
        f"[END] success={str(success).lower()} steps={steps} score={bounded_score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _single_line(value: str | None) -> str:
    return (value or "").replace("\n", "\\n").replace("\r", "\\r")


def debug_log(message: str) -> None:
    if DEBUG_LOGGING:
        print(message, file=sys.stderr, flush=True)


def format_action_str(action_dict: Optional[dict[str, Any]]) -> str:
    if not isinstance(action_dict, dict):
        return "null"
    action_type = str(action_dict.get("action_type") or "").strip() or "unknown"
    params = action_dict.get("parameters", {})
    if not isinstance(params, dict) or not params:
        return f"{action_type}()"

    parts: list[str] = []
    for key in sorted(params):
        value = params[key]
        if value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            text = str(value).replace("\\", "\\\\").replace("'", "\\'")
            text = text.replace("\n", "\\n").replace("\r", "\\r")
            rendered = f"'{text}'"
        parts.append(f"{key}={rendered}")
    return f"{action_type}({','.join(parts)})"


def action_with_provenance(action_dict: dict[str, Any], source: str) -> dict[str, Any]:
    params = dict(action_dict.get("parameters", {}))
    params["agent_source"] = source
    return {
        "action_type": action_dict.get("action_type", ""),
        "parameters": params,
    }


def repair_ready_to_submit(task_name: str, observation: Any) -> bool:
    if not task_name.startswith("repair_"):
        return False
    if not observation.edited_files or not observation.available_validators:
        return False
    return all(
        observation.validator_status.get(validator, {}).get("passed", False)
        for validator in observation.available_validators
    )


def repeated_failed_repair_edit(task_name: str, observation: Any, action_dict: Optional[dict[str, Any]]) -> bool:
    if not task_name.startswith("repair_") or not isinstance(action_dict, dict):
        return False
    if action_dict.get("action_type") != "edit_file":
        return False
    params = action_dict.get("parameters", {})
    if not isinstance(params, dict):
        return False
    path = str(params.get("path") or "").strip()
    content = str(params.get("content") or "")
    if not path or path not in observation.edited_files:
        return False
    if observation.edited_files.get(path, "") != content:
        return False
    return any(not status.get("passed", False) for status in observation.validator_status.values())


async def create_env() -> DeveloperControlRoomEnv:
    if ENV_URL:
        env = DeveloperControlRoomEnv(base_url=ENV_URL)
        await env.connect()
        return env
    raise ValueError("Set ENV_URL before running inference.py")


async def run_task(
    client: OpenAI,
    task_name: str,
    scenario_index: int | None = None,
    scenario_seed: int | None = None,
) -> None:
    env: DeveloperControlRoomEnv | None = None
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    scenario_id: str | None = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await create_env()
        reset_kwargs: dict[str, Any] = {"task_id": task_name}
        if scenario_index is not None:
            reset_kwargs["scenario_index"] = scenario_index
        elif scenario_seed is not None:
            reset_kwargs["seed"] = scenario_seed
        result = await env.reset(**reset_kwargs)
        observation = result.observation
        scenario_id = observation.scenario_id
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_dict: Optional[dict[str, Any]] = None
            action_source = "fallback"
            try:
                if repair_ready_to_submit(task_name, observation):
                    action_dict = fallback_action(task_name, observation)
                elif step <= MAX_MODEL_STEPS:
                    action_dict = get_model_action(
                        client,
                        MODEL_NAME,
                        TEMPERATURE,
                        MAX_TOKENS,
                        MODEL_TIMEOUT_SECONDS,
                        step,
                        observation,
                        history,
                        "",
                    )
                    if action_dict is not None:
                        action_source = "llm"
                        if repeated_failed_repair_edit(task_name, observation, action_dict):
                            action_dict = None
                if not action_is_valid(action_dict, observation):
                    action_source = "fallback_after_llm" if action_dict is not None else "fallback"
                    action_dict = fallback_action(task_name, observation)

                action = build_action(action_with_provenance(action_dict, action_source))
            except Exception as exc:
                debug_log(f"[DEBUG] Step {step} action preparation failed: {exc}")
                debug_log(traceback.format_exc().rstrip())
                action_source = "fallback_after_error"
                action_dict = fallback_action(task_name, observation)
                action = build_action(action_with_provenance(action_dict, action_source))
            result = await env.step(action)
            observation = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = observation.last_action_error

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=format_action_str(action_dict),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(f"Step {step}: {format_action_str(action_dict)} -> reward {reward:+.2f}")

            if done:
                break

        grader_result = fetch_grader_result(env, ENV_URL)
        score = grader_result["total"]
        success = grader_result["solved"]
    except Exception as exc:
        debug_log(f"[DEBUG] task run failed: {exc}")
        debug_log(traceback.format_exc().rstrip())
        success = False
        score = 0.0
        raise

    finally:
        try:
            if env is not None:
                await env.close()
        except Exception as exc:
            debug_log(f"[DEBUG] env.close() error (container cleanup): {exc}")
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = list_tasks()
    selected_tasks = [task for task in tasks if task["id"] == TASK_NAME] if TASK_NAME else tasks

    for task_offset, task in enumerate(selected_tasks):
        task_name = task["id"]
        if RUN_ALL_SCENARIOS:
            for scenario_index in range(task["scenarios"]):
                scenario_seed = None if SCENARIO_SEED is None else SCENARIO_SEED + scenario_index + task_offset
                await run_task(
                    client,
                    task_name,
                    scenario_index=scenario_index,
                    scenario_seed=scenario_seed,
                )
            continue

        scenario_seed = None
        if SCENARIO_INDEX is None and SCENARIO_SEED is not None:
            scenario_seed = SCENARIO_SEED + task_offset

        await run_task(
            client,
            task_name,
            scenario_index=SCENARIO_INDEX,
            scenario_seed=scenario_seed,
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        debug_log(f"[DEBUG] inference failed: {exc}")
        debug_log(traceback.format_exc().rstrip())
        raise SystemExit(1)
