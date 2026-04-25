"""FastAPI app for the Developer Control Room benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import HTTPException

from openenv.core.env_server.web_interface import create_web_interface_app

try:
    from ..graders import grade
    from ..models import DeveloperControlRoomAction, DeveloperControlRoomObservation
    from ..tasks import ALL_TASKS, list_tasks
    from .developer_control_room_environment import DeveloperControlRoomEnvironment
    from .ui import build_developer_control_room_ui
except ImportError:
    from graders import grade
    from models import DeveloperControlRoomAction, DeveloperControlRoomObservation
    from tasks import ALL_TASKS, list_tasks
    from server.developer_control_room_environment import DeveloperControlRoomEnvironment
    from server.ui import build_developer_control_room_ui

ROOT = Path(__file__).resolve().parents[1]
ENV = DeveloperControlRoomEnvironment()


def _get_env() -> DeveloperControlRoomEnvironment:
    return ENV


app = create_web_interface_app(
    _get_env,
    DeveloperControlRoomAction,
    DeveloperControlRoomObservation,
    env_name="developer_control_room",
    max_concurrent_envs=1,
    gradio_builder=build_developer_control_room_ui,
)


@app.get("/tasks")
def tasks() -> dict:
    return {
        "tasks": list_tasks(),
        "total": len(ALL_TASKS),
        "action_schema": {
            "search_workspace": {"query": "string"},
            "read_file": {"path": "string"},
            "inspect_schema": {"asset": "string"},
            "inspect_lineage": {"asset": "string"},
            "inspect_llm_draft": {"draft_id": "string"},
            "edit_file": {"path": "string", "content": "string"},
            "run_validator": {"validator": "string"},
            "submit_repair": {
                "root_cause": "string",
                "fix_path": "string",
                "summary": "string",
            },
            "submit_review": {
                "verdict": "approve|reject",
                "issue_type": "string",
                "summary": "string",
            },
            "submit_workspace": {"summary": "string"},
        },
    }


@app.get("/grader")
def grader() -> dict:
    state = ENV.state.model_dump()
    if not state.get("task_id"):
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    result = grade(state["task_id"], state, ENV._scenario)
    return {
        "task_id": state["task_id"],
        "scenario_id": state.get("scenario_id"),
        "steps_used": state.get("step_count", 0),
        "done": state.get("done", False),
        "total": result["total"],
        "solved": result.get("solved", False),
        "breakdown": result["breakdown"],
        "feedback": result["feedback"],
    }


@app.post("/baseline")
def baseline() -> dict:
    script = ROOT / "inference.py"
    if not script.exists():
        raise HTTPException(status_code=500, detail="inference.py not found")
    env = {
        **os.environ,
        "ENV_URL": os.environ.get("ENV_URL", "http://localhost:7860"),
    }
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1200,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=500, detail=f"inference.py timed out: {exc}") from exc

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "returncode": result.returncode,
                "stdout_tail": result.stdout.splitlines()[-20:],
                "stderr_tail": result.stderr.splitlines()[-20:],
            },
        )

    return {
        "status": "completed",
        "stdout_tail": result.stdout.splitlines()[-20:],
        "stderr_tail": result.stderr.splitlines()[-20:],
    }


@app.get("/api/info")
def api_info() -> dict:
    return {
        "name": "developer_control_room",
        "version": "0.1.0",
        "description": "FlowOS end-to-end benchmark for production AI agents",
        "tasks": [task["id"] for task in list_tasks()],
        "docs": "/docs",
    }


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
