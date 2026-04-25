"""Core environment implementation for Developer Control Room."""

from __future__ import annotations

import os
import sys
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..graders import evaluate_validator, grade
    from ..models import (
        DeveloperControlRoomAction,
        DeveloperControlRoomObservation,
        DeveloperControlRoomState,
    )
    from ..runtime import execute_csv_report_runtime
    from ..tasks import get_scenario, get_task, scenario_count
except ImportError:
    from graders import evaluate_validator, grade
    from models import (
        DeveloperControlRoomAction,
        DeveloperControlRoomObservation,
        DeveloperControlRoomState,
    )
    from runtime import execute_csv_report_runtime
    from tasks import get_scenario, get_task, scenario_count


class DeveloperControlRoomEnvironment(
    Environment[
        DeveloperControlRoomAction,
        DeveloperControlRoomObservation,
        DeveloperControlRoomState,
    ]
):
    """Singleton-friendly environment simulating a developer workspace."""

    SUPPORTS_CONCURRENT_SESSIONS = False
    MAX_PROGRESS_DELTA_REWARD = float(
        os.getenv("DEVELOPER_CONTROL_ROOM_MAX_PROGRESS_DELTA_REWARD", "0.30")
    )
    SOLVED_TERMINAL_REWARD_BONUS = float(
        os.getenv("DEVELOPER_CONTROL_ROOM_SOLVED_TERMINAL_REWARD_BONUS", "0.10")
    )
    SERVER_STEP_LOGS = os.getenv("DEVELOPER_CONTROL_ROOM_SERVER_STEP_LOGS", "true").lower() == "true"

    def __init__(self) -> None:
        self._task_def: dict = {}
        self._scenario: dict = {}
        self._state = DeveloperControlRoomState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=None,
            scenario_id=None,
        )
        self._ready = False

    def _server_log(self, message: str) -> None:
        if self.SERVER_STEP_LOGS:
            print(message, file=sys.stderr, flush=True)

    def _fmt_value(self, value: object) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value).replace("\\", "\\\\").replace("'", "\\'")
        text = text.replace("\n", "\\n").replace("\r", "\\r")
        return f"'{text}'"

    def _format_action(self, action: DeveloperControlRoomAction) -> str:
        payload = action.parameters.model_dump(exclude_none=True)
        payload.pop("agent_source", None)
        if not payload:
            return f"{action.action_type}()"
        parts = [f"{key}={self._fmt_value(payload[key])}" for key in sorted(payload)]
        return f"{action.action_type}({','.join(parts)})"

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = "repair_data_transform",
        scenario_index: int | None = None,
        **_: object,
    ) -> DeveloperControlRoomObservation:
        self._task_def = get_task(task_id)
        if scenario_index is None:
            scenario_index = (seed or 0) % scenario_count(task_id)
        self._scenario = get_scenario(task_id, scenario_index, seed=seed)
        self._state = DeveloperControlRoomState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            scenario_id=self._scenario["scenario_id"],
            max_steps=self._task_def["max_steps"],
            developer_request=self._scenario["developer_request"],
            available_actions=list(self._task_def["available_actions"]),
            action_history=[],
            queried_data={},
            edited_files={},
            validator_status={},
            submission={},
            submitted=False,
            active_role="builder",
            runtime_status={},
            execution_logs=[],
            output_schema=[],
            report_preview=[],
            materialized_artifacts={},
            cumulative_reward=0.0,
            done=False,
            feedback="Episode started.",
            last_action_error=None,
        )
        self._ready = True
        self._server_log(
            f"[SERVER-START] task={task_id} scenario={self._scenario['scenario_id']} max_steps={self._task_def['max_steps']}"
        )
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: DeveloperControlRoomAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> DeveloperControlRoomObservation:
        del timeout_s
        if not self._ready:
            raise RuntimeError("Call reset() before step().")

        if self._state.done:
            return self._build_observation(
                reward=0.0,
                done=True,
                feedback="Episode already completed.",
            )

        before = grade(self._state.task_id or "", self._state.model_dump(), self._scenario)["total"]

        self._state.step_count += 1
        self._state.last_action_error = None
        params = action.parameters
        payload = params.model_dump(exclude_none=True)
        self._state.action_history.append(
            {
                "step": self._state.step_count,
                "action_type": action.action_type,
                "parameters": payload,
            }
        )

        reward = 0.0
        feedback: list[str] = []
        available = set(self._task_def.get("available_actions", []))
        if action.action_type not in available:
            reward -= 0.08
            self._state.last_action_error = f"Action `{action.action_type}` is not available in this task."
            feedback.append("invalid action")
        else:
            reward_delta, step_feedback = self._handle_action(action)
            reward += reward_delta
            if step_feedback:
                feedback.append(step_feedback)

        half = max(1, self._state.max_steps // 2)
        if self._state.step_count > half and not self._state.submitted:
            reward -= 0.01
            feedback.append("late in episode")

        if self._state.step_count >= self._state.max_steps and not self._state.done:
            self._state.done = True
            reward -= 0.1
            feedback.append("max steps reached")

        after_result = grade(self._state.task_id or "", self._state.model_dump(), self._scenario)
        progress_delta = after_result["total"] - before
        progress_delta = max(
            -self.MAX_PROGRESS_DELTA_REWARD,
            min(self.MAX_PROGRESS_DELTA_REWARD, progress_delta),
        )
        reward += progress_delta
        if self._state.done and after_result.get("solved"):
            reward += self.SOLVED_TERMINAL_REWARD_BONUS
        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 4)
        self._state.feedback = " | ".join(part for part in feedback if part) or after_result["feedback"]

        source = payload.get("agent_source", "unknown")
        self._server_log(
            f"[SERVER-STEP] step={self._state.step_count} scenario={self._state.scenario_id} "
            f"source={source} action={self._format_action(action)} reward={round(reward, 4):.2f} "
            f"done={str(self._state.done).lower()} error={self._state.last_action_error or 'null'}"
        )
        if self._state.done:
            self._server_log(
                f"[SERVER-END] task={self._state.task_id} scenario={self._state.scenario_id} "
                f"steps={self._state.step_count} total={after_result['total']:.2f} "
                f"solved={str(after_result.get('solved', False)).lower()}"
            )

        return self._build_observation(
            reward=round(reward, 4),
            done=self._state.done,
            feedback=self._state.feedback,
        )

    def _handle_action(self, action: DeveloperControlRoomAction) -> tuple[float, str]:
        params = action.parameters
        action_type = action.action_type
        queried = self._state.queried_data

        if action_type == "search_workspace":
            query = (params.query or "").strip()
            if not query:
                self._state.last_action_error = "search_workspace requires `query`."
                return -0.05, "missing query"
            results = self._search_workspace(query)
            query_map = queried.setdefault("search_workspace", {})
            first_time = query not in query_map
            query_map[query] = results
            return (0.02 if first_time else -0.01), f"search returned {len(results)} results"

        if action_type == "read_file":
            path = (params.path or "").strip()
            if not path:
                self._state.last_action_error = "read_file requires `path`."
                return -0.05, "missing path"
            content = self._get_file(path)
            if content is None:
                self._state.last_action_error = f"Unknown file: {path}"
                return -0.05, "unknown file"
            file_map = queried.setdefault("read_file", {})
            first_time = path not in file_map
            file_map[path] = content
            return (0.02 if first_time else -0.01), f"read {path}"

        if action_type == "inspect_schema":
            asset = (params.asset or "").strip()
            schema = self._scenario.get("schema_registry", {}).get(asset)
            if not asset or schema is None:
                self._state.last_action_error = f"Unknown asset for inspect_schema: {asset or '<empty>'}"
                return -0.05, "unknown asset"
            schema_map = queried.setdefault("inspect_schema", {})
            first_time = asset not in schema_map
            schema_map[asset] = schema
            return (0.03 if first_time else -0.01), f"inspected schema {asset}"

        if action_type == "inspect_lineage":
            asset = (params.asset or "").strip()
            lineage = self._scenario.get("lineage", {}).get(asset)
            if not asset or lineage is None:
                self._state.last_action_error = f"Unknown asset for inspect_lineage: {asset or '<empty>'}"
                return -0.05, "unknown lineage target"
            lineage_map = queried.setdefault("inspect_lineage", {})
            first_time = asset not in lineage_map
            lineage_map[asset] = lineage
            return (0.03 if first_time else -0.01), f"inspected lineage {asset}"

        if action_type == "inspect_llm_draft":
            draft_id = (params.draft_id or "primary").strip()
            draft = self._scenario.get("llm_draft", {}).get(draft_id)
            if draft is None:
                self._state.last_action_error = f"Unknown draft id: {draft_id}"
                return -0.05, "unknown draft"
            draft_map = queried.setdefault("inspect_llm_draft", {})
            first_time = draft_id not in draft_map
            draft_map[draft_id] = draft
            return (0.03 if first_time else -0.01), f"inspected draft {draft_id}"

        if action_type == "edit_file":
            path = (params.path or "").strip()
            content = params.content or ""
            if not path or not content.strip():
                self._state.last_action_error = "edit_file requires `path` and non-empty `content`."
                return -0.06, "invalid edit"
            allowed = set(self._scenario.get("editable_targets", []))
            if path not in allowed:
                self._state.last_action_error = f"Editing `{path}` is not allowed in this scenario."
                return -0.08, "forbidden edit target"
            current = self._get_file(path) or ""
            if current.strip() == content.strip():
                return -0.03, f"no-op edit on {path}"
            self._state.edited_files[path] = content
            queried.setdefault("edit_file", {})[path] = {
                "preview": content[:200],
                "length": len(content),
            }
            if self._state.task_id == "simulate_csv_report_workflow":
                runtime_status = execute_csv_report_runtime(self._scenario, self._state.model_dump())
                self._state.runtime_status = runtime_status
                self._state.execution_logs = list(runtime_status.get("logs", [])) + list(runtime_status.get("errors", []))
                self._state.output_schema = list(runtime_status.get("output_schema", []))
                self._state.report_preview = list(runtime_status.get("report_preview", []))
                self._state.materialized_artifacts = dict(runtime_status.get("materialized_artifacts", {}))
                if runtime_status.get("errors"):
                    self._state.last_action_error = runtime_status["errors"][-1]
                self._state.active_role = "fixer"
            return -0.01, f"edited {path}"

        if action_type == "run_validator":
            validator = (params.validator or "").strip()
            if not validator:
                self._state.last_action_error = "run_validator requires `validator`."
                return -0.05, "missing validator"
            result = evaluate_validator(self._state.task_id or "", validator, self._state.model_dump(), self._scenario)
            first_time = validator not in self._state.validator_status
            self._state.validator_status[validator] = result
            queried.setdefault("run_validator", {})[validator] = result
            bonus = 0.02 if first_time else -0.01
            if result["passed"]:
                bonus += 0.01
            return bonus, f"{validator}: {'pass' if result['passed'] else 'fail'}"

        if action_type == "submit_repair":
            return self._submit(
                action_type,
                {
                    "root_cause": params.root_cause or "",
                    "fix_path": params.fix_path or "",
                    "summary": params.summary or "",
                },
            )

        if action_type == "submit_review":
            return self._submit(
                action_type,
                {
                    "verdict": params.verdict or "",
                    "issue_type": params.issue_type or "",
                    "summary": params.summary or "",
                },
            )

        if action_type == "submit_workspace":
            return self._submit(action_type, {"summary": params.summary or ""})

        self._state.last_action_error = f"Unhandled action `{action_type}`"
        return -0.08, "unhandled action"

    def _submit(self, action_type: str, submission: dict[str, str]) -> tuple[float, str]:
        expected = self._task_def.get("submission_action")
        if action_type != expected:
            self._state.last_action_error = f"Wrong submission action. Expected `{expected}`."
            return -0.08, "wrong submission action"
        self._state.submission = submission
        self._state.submitted = True
        self._state.done = True
        return 0.02, f"submitted via {action_type}"

    def _search_workspace(self, query: str) -> list[dict[str, str]]:
        tokens = [_token for _token in query.lower().split() if _token]
        matches: list[dict[str, str]] = []
        files = {**self._scenario.get("files", {}), **self._state.edited_files}
        for path, content in files.items():
            haystack = f"{path}\n{content}".lower()
            if all(token in haystack for token in tokens):
                snippet = " ".join(content.strip().split())[:180]
                matches.append({"path": path, "snippet": snippet})
        for asset, content in self._scenario.get("schema_registry", {}).items():
            haystack = f"{asset}\n{content}".lower()
            if all(token in haystack for token in tokens):
                snippet = " ".join(content.strip().split())[:180]
                matches.append({"path": f"schema::{asset}", "snippet": snippet})
        return matches[:5]

    def _get_file(self, path: str) -> str | None:
        if path in self._state.edited_files:
            return self._state.edited_files[path]
        if path in self._scenario.get("files", {}):
            return self._scenario["files"][path]
        if path in self._scenario.get("editable_targets", []):
            return "File does not exist yet. Create it with edit_file."
        return None

    def _build_observation(
        self,
        reward: float,
        done: bool,
        feedback: str | None = None,
    ) -> DeveloperControlRoomObservation:
        return DeveloperControlRoomObservation(
            episode_id=self._state.episode_id or "",
            task_id=self._state.task_id or "",
            scenario_id=self._state.scenario_id or "",
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            developer_request=self._scenario.get("developer_request", ""),
            workspace_summary=self._scenario.get("workspace_summary", ""),
            available_actions=list(self._task_def.get("available_actions", [])),
            known_files=list(self._scenario.get("known_files", [])),
            editable_targets=list(self._scenario.get("editable_targets", [])),
            known_assets=list(self._scenario.get("known_assets", [])),
            available_validators=list(self._scenario.get("available_validators", [])),
            queried_data=self._state.queried_data,
            edited_files=self._state.edited_files,
            validator_status=self._state.validator_status,
            active_role=self._state.active_role,
            runtime_status=self._state.runtime_status,
            execution_logs=self._state.execution_logs,
            output_schema=self._state.output_schema,
            report_preview=self._state.report_preview,
            materialized_artifacts=self._state.materialized_artifacts,
            cumulative_reward=self._state.cumulative_reward,
            feedback=feedback or self._state.feedback,
            last_action_error=self._state.last_action_error,
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> DeveloperControlRoomState:
        if not self._ready:
            return self._state
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="developer_control_room",
            description=(
                "OpenEnv benchmark for AI developer assistants working on "
                "pipeline repair, LLM patch review, and workflow shipping."
            ),
            version="0.1.0",
            author="Codex",
        )

    def close(self) -> None:
        """No-op because the app uses a singleton environment instance."""
