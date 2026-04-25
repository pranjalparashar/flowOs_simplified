"""Typed models for the Developer Control Room OpenEnv environment."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class ActionParameters(BaseModel):
    """Flexible bag of action-specific parameters."""

    query: str | None = None
    path: str | None = None
    asset: str | None = None
    validator: str | None = None
    content: str | None = None
    root_cause: str | None = None
    fix_path: str | None = None
    verdict: str | None = None
    issue_type: str | None = None
    summary: str | None = None
    draft_id: str | None = None

    model_config = {"extra": "allow"}


class DeveloperControlRoomAction(Action):
    """Action accepted by the environment."""

    action_type: str = Field(..., description="Type of action to execute")
    parameters: ActionParameters = Field(
        default_factory=ActionParameters,
        description="Action-specific parameters",
    )


class DeveloperControlRoomObservation(Observation):
    """Observation returned after reset and step."""

    episode_id: str = Field(..., description="Unique episode identifier")
    task_id: str = Field(..., description="Active task identifier")
    scenario_id: str = Field(..., description="Scenario identifier")
    step_count: int = Field(..., ge=0, description="Steps used in the episode")
    max_steps: int = Field(..., ge=1, description="Maximum steps allowed")
    developer_request: str = Field(..., description="Natural-language request")
    workspace_summary: str = Field(..., description="High-level workspace context")
    available_actions: list[str] = Field(
        default_factory=list,
        description="Valid actions for the current task",
    )
    known_files: list[str] = Field(
        default_factory=list,
        description="Readable workspace files",
    )
    editable_targets: list[str] = Field(
        default_factory=list,
        description="Files the agent may create or modify",
    )
    known_assets: list[str] = Field(
        default_factory=list,
        description="Schemas or logical assets visible to the agent",
    )
    available_validators: list[str] = Field(
        default_factory=list,
        description="Validators available for the current scenario",
    )
    queried_data: dict[str, Any] = Field(
        default_factory=dict,
        description="All tool responses gathered so far",
    )
    edited_files: dict[str, str] = Field(
        default_factory=dict,
        description="Current edited file contents",
    )
    validator_status: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Validator pass/fail status accumulated so far",
    )
    active_role: str = Field(default="builder", description="Current active role in shared-workspace tasks")
    runtime_status: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution/runtime status for simulation-backed tasks",
    )
    execution_logs: list[str] = Field(
        default_factory=list,
        description="Recent execution log lines from the runtime",
    )
    output_schema: list[str] = Field(
        default_factory=list,
        description="Current output schema produced by the runtime",
    )
    report_preview: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Preview rows from the final report output",
    )
    materialized_artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Generated artifact contents returned by simulation-backed tasks",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Running reward across the episode",
    )
    feedback: str = Field(default="", description="Per-step feedback")
    last_action_error: str | None = Field(
        default=None,
        description="Error from the previous action, if any",
    )


class DeveloperControlRoomState(State):
    """Internal state returned by GET /state."""

    task_id: str | None = Field(default=None, description="Current task identifier")
    scenario_id: str | None = Field(
        default=None,
        description="Current scenario identifier",
    )
    max_steps: int = Field(default=0, ge=0, description="Maximum allowed steps")
    developer_request: str = Field(default="", description="Request under evaluation")
    available_actions: list[str] = Field(default_factory=list)
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    queried_data: dict[str, Any] = Field(default_factory=dict)
    edited_files: dict[str, str] = Field(default_factory=dict)
    validator_status: dict[str, dict[str, Any]] = Field(default_factory=dict)
    submission: dict[str, Any] = Field(default_factory=dict)
    submitted: bool = Field(default=False)
    active_role: str = Field(default="builder")
    runtime_status: dict[str, Any] = Field(default_factory=dict)
    execution_logs: list[str] = Field(default_factory=list)
    output_schema: list[str] = Field(default_factory=list)
    report_preview: list[dict[str, Any]] = Field(default_factory=list)
    materialized_artifacts: dict[str, str] = Field(default_factory=dict)
    cumulative_reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    feedback: str = Field(default="")
    last_action_error: str | None = Field(default=None)
