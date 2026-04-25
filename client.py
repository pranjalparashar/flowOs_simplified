"""Client wrapper for the Developer Control Room benchmark."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        DeveloperControlRoomAction,
        DeveloperControlRoomObservation,
        DeveloperControlRoomState,
    )
except ImportError:
    from models import (
        DeveloperControlRoomAction,
        DeveloperControlRoomObservation,
        DeveloperControlRoomState,
    )


class DeveloperControlRoomEnv(
    EnvClient[
        DeveloperControlRoomAction,
        DeveloperControlRoomObservation,
        DeveloperControlRoomState,
    ]
):
    """Convenience client for the Developer Control Room environment."""

    def _step_payload(self, action: DeveloperControlRoomAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self,
        payload: dict[str, Any],
    ) -> StepResult[DeveloperControlRoomObservation]:
        obs = payload.get("observation", {})
        observation = DeveloperControlRoomObservation(
            episode_id=obs.get("episode_id", ""),
            task_id=obs.get("task_id", ""),
            scenario_id=obs.get("scenario_id", ""),
            step_count=obs.get("step_count", 0),
            max_steps=obs.get("max_steps", 0),
            developer_request=obs.get("developer_request", ""),
            workspace_summary=obs.get("workspace_summary", ""),
            available_actions=obs.get("available_actions", []),
            known_files=obs.get("known_files", []),
            editable_targets=obs.get("editable_targets", []),
            known_assets=obs.get("known_assets", []),
            available_validators=obs.get("available_validators", []),
            queried_data=obs.get("queried_data", {}),
            edited_files=obs.get("edited_files", {}),
            validator_status=obs.get("validator_status", {}),
            active_role=obs.get("active_role", "builder"),
            runtime_status=obs.get("runtime_status", {}),
            execution_logs=obs.get("execution_logs", []),
            output_schema=obs.get("output_schema", []),
            report_preview=obs.get("report_preview", []),
            materialized_artifacts=obs.get("materialized_artifacts", {}),
            cumulative_reward=obs.get("cumulative_reward", 0.0),
            feedback=obs.get("feedback", ""),
            last_action_error=obs.get("last_action_error"),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> DeveloperControlRoomState:
        return DeveloperControlRoomState(**payload)
