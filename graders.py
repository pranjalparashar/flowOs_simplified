"""Deterministic graders and validator helpers for Developer Control Room."""

from __future__ import annotations

from typing import Any

try:
    from .tasks import get_grader_family
except ImportError:
    from tasks import get_grader_family


def _norm(text: str) -> str:
    return text.lower().strip()


def _strict_score(value: float) -> float:
    return round(min(0.99, max(0.01, value)), 4)


def _workspace_files(state: dict, scenario: dict) -> dict[str, str]:
    files = dict(scenario.get("files", {}))
    files.update(state.get("edited_files", {}))
    return files


def _token_score(text: str, groups: list[list[str]]) -> tuple[float, int, int]:
    if not groups:
        return 1.0, 0, 0
    lowered = _norm(text)
    matched = 0
    for group in groups:
        if any(_norm(option) in lowered for option in group):
            matched += 1
    return matched / len(groups), matched, len(groups)


def _contains_any(text: str, snippets: list[str]) -> bool:
    lowered = _norm(text)
    return any(_norm(snippet) in lowered for snippet in snippets)


def _source_text(source: dict, state: dict, scenario: dict) -> str:
    source_type = source.get("type")
    if source_type == "file":
        return _workspace_files(state, scenario).get(source.get("path", ""), "")
    if source_type == "draft":
        return scenario.get("llm_draft", {}).get(source.get("draft_id", "primary"), "")
    if source_type == "schema":
        return scenario.get("schema_registry", {}).get(source.get("asset", ""), "")
    return ""


def _investigation_score(state: dict, targets: set[tuple[str, str]]) -> float:
    history = state.get("action_history", [])
    seen: set[tuple[str, str]] = set()
    for item in history:
        action_type = item.get("action_type", "")
        params = item.get("parameters", {})
        key = (
            action_type,
            _norm(
                params.get("path")
                or params.get("asset")
                or params.get("validator")
                or params.get("draft_id")
                or params.get("query")
                or ""
            ),
        )
        if key in targets:
            seen.add(key)
    if not targets:
        return 0.0
    return min(1.0, len(seen) / len(targets))


def _groups_score(text: str, groups: list[list[str]], forbidden_terms: list[str] | None = None) -> float:
    score, _, _ = _token_score(text, groups)
    if forbidden_terms and _contains_any(text, forbidden_terms):
        return 0.0
    return score


def _summary_score(text: str, groups: list[list[str]]) -> float:
    return _groups_score(text, groups)


def _validator_usage_score(state: dict, validator_targets: list[str]) -> float:
    if not validator_targets:
        return 0.0
    seen = state.get("validator_status", {})
    matched = sum(1 for name in validator_targets if name in seen)
    return matched / len(validator_targets)


def evaluate_validator(
    task_id: str,
    validator_name: str,
    state: dict,
    scenario: dict,
) -> dict[str, Any]:
    del task_id
    validators = scenario.get("validators", {})
    spec = validators.get(validator_name)
    if spec is None:
        return {"passed": False, "message": f"Unknown validator '{validator_name}'"}

    kind = spec.get("kind")

    if kind == "file_groups":
        text = _source_text({"type": "file", "path": spec.get("path", "")}, state, scenario)
        score, matched, total = _token_score(text, spec.get("required_groups", []))
        has_forbidden = _contains_any(text, spec.get("forbidden_terms", []))
        ok = score == 1.0 and not has_forbidden
        return {
            "passed": ok,
            "message": (
                f"PASS: {validator_name} satisfied."
                if ok
                else f"FAIL: matched {matched}/{total} groups; forbidden={has_forbidden}."
            ),
        }

    if kind == "draft_groups":
        text = _source_text({"type": "draft", "draft_id": spec.get("draft_id", "primary")}, state, scenario)
        score, matched, total = _token_score(text, spec.get("required_groups", []))
        has_forbidden = _contains_any(text, spec.get("forbidden_terms", []))
        ok = score == 1.0 and not has_forbidden
        return {
            "passed": ok,
            "message": (
                f"PASS: {validator_name} satisfied."
                if ok
                else f"FAIL: matched {matched}/{total} groups; forbidden={has_forbidden}."
            ),
        }

    if kind == "workflow_artifact":
        target = scenario.get("workflow_target", {})
        artifact_key = spec.get("artifact", "")
        artifact = target.get("artifacts", {}).get(artifact_key, {})
        text = _source_text({"type": "file", "path": artifact.get("path", "")}, state, scenario)
        score, matched, total = _token_score(text, artifact.get("required_groups", []))
        has_forbidden = _contains_any(text, artifact.get("forbidden_terms", []))
        ok = score == 1.0 and not has_forbidden
        return {
            "passed": ok,
            "message": (
                f"PASS: {artifact_key} artifact satisfied validator."
                if ok
                else f"FAIL: matched {matched}/{total} groups; forbidden={has_forbidden}."
            ),
        }

    if kind == "runtime_check":
        runtime_status = state.get("runtime_status", {})
        checks = runtime_status.get("checks", {})
        check_key = spec.get("check_key", "")
        passed = bool(checks.get(check_key, False))
        return {
            "passed": passed,
            "message": (
                f"PASS: runtime check {check_key} satisfied."
                if passed
                else f"FAIL: runtime check {check_key} not satisfied."
            ),
        }

    return {"passed": False, "message": f"Unhandled validator kind '{kind}'"}


def _grade_pipeline_repair(state: dict, scenario: dict) -> dict[str, Any]:
    files = _workspace_files(state, scenario)
    target = scenario["repair_target"]
    path = target["path"]
    text = files.get(path, "")

    investigation = 0.15 * _investigation_score(
        state,
        {(action_type, _norm(name)) for action_type, name in target.get("investigation_targets", [])},
    )

    edit_raw = _groups_score(text, target.get("required_groups", []), target.get("forbidden_terms", []))
    edit_score = 0.45 * edit_raw

    validator_status = state.get("validator_status", {})
    validator_targets = scenario.get("available_validators", [])
    validator_hits = sum(1 for name in validator_targets if validator_status.get(name, {}).get("passed"))
    validator_score = 0.2 * (validator_hits / len(validator_targets) if validator_targets else 0.0)

    submission = state.get("submission", {})
    root_cause_score = _summary_score(submission.get("root_cause", ""), target.get("root_cause_groups", []))
    summary_score_raw = _summary_score(submission.get("summary", ""), target.get("summary_groups", []))
    fix_path_ok = 1.0 if submission.get("fix_path", "") == target.get("fix_path", "") else 0.0
    submission_score = 0.2 * ((root_cause_score + summary_score_raw + fix_path_ok) / 3.0)
    solved = (
        state.get("submitted", False)
        and edit_raw == 1.0
        and validator_hits >= max(1, len(validator_targets) - 1)
        and root_cause_score >= (2 / 3)
        and fix_path_ok == 1.0
    )

    total = _strict_score(investigation + edit_score + validator_score + submission_score)
    return {
        "total": total,
        "solved": solved,
        "breakdown": {
            "investigation": round(investigation, 4),
            "edit_score": round(edit_score, 4),
            "validator_score": round(validator_score, 4),
            "submission_score": round(submission_score, 4),
        },
        "feedback": (
            f"investigation={investigation:.2f} "
            f"edit={edit_score:.2f} validators={validator_score:.2f} "
            f"submission={submission_score:.2f}"
        ),
    }


def _grade_llm_patch_review(state: dict, scenario: dict) -> dict[str, Any]:
    target = scenario["review_target"]
    submission = state.get("submission", {})

    investigation = 0.15 * _investigation_score(
        state,
        {(action_type, _norm(name)) for action_type, name in target.get("investigation_targets", [])},
    )

    verdict_score = 0.3 if _norm(submission.get("verdict", "")) == target.get("correct_verdict", "") else 0.0
    issue_score = 0.25 if _norm(submission.get("issue_type", "")) == target.get("correct_issue_type", "") else 0.0
    summary_raw = _summary_score(submission.get("summary", ""), target.get("summary_groups", []))
    summary_score = 0.12 * summary_raw
    validator_score = 0.08 * _validator_usage_score(state, target.get("validator_targets", []))
    solved = (
        state.get("submitted", False)
        and verdict_score > 0.0
        and issue_score > 0.0
        and summary_raw >= (2 / 3)
        and _investigation_score(
            state,
            {(action_type, _norm(name)) for action_type, name in target.get("investigation_targets", [])},
        )
        >= (2 / 3)
    )

    total = _strict_score(investigation + verdict_score + issue_score + summary_score + validator_score)
    return {
        "total": total,
        "solved": solved,
        "breakdown": {
            "investigation": round(investigation, 4),
            "verdict_score": round(verdict_score, 4),
            "issue_score": round(issue_score, 4),
            "summary_score": round(summary_score, 4),
            "validator_score": round(validator_score, 4),
        },
        "feedback": (
            f"investigation={investigation:.2f} verdict={verdict_score:.2f} "
            f"issue={issue_score:.2f} summary={summary_score:.2f}"
        ),
    }


def _artifact_score(files: dict[str, str], target: dict) -> float:
    required_artifacts = target.get("required_artifacts", [])
    artifacts = target.get("artifacts", {})
    if not required_artifacts:
        return 0.0

    total_weight = sum(artifacts[name].get("weight", 1.0) for name in required_artifacts)
    if total_weight <= 0:
        return 0.0

    weighted_score = 0.0
    for name in required_artifacts:
        artifact = artifacts.get(name, {})
        path = artifact.get("path", "")
        text = files.get(path, "")
        raw = _groups_score(text, artifact.get("required_groups", []), artifact.get("forbidden_terms", []))
        weighted_score += artifact.get("weight", 1.0) * raw
    return weighted_score / total_weight


def _governance_score(files: dict[str, str], target: dict) -> float:
    forbidden_terms = target.get("forbidden_terms", [])
    required_artifacts = target.get("required_artifacts", [])
    artifacts = target.get("artifacts", {})
    if not forbidden_terms:
        return 1.0

    for name in required_artifacts:
        artifact = artifacts.get(name, {})
        text = files.get(artifact.get("path", ""), "")
        artifact_forbidden = list(forbidden_terms) + artifact.get("forbidden_terms", [])
        if _contains_any(text, artifact_forbidden):
            return 0.0
    return 1.0


def _grade_workflow_shipping(state: dict, scenario: dict) -> dict[str, Any]:
    files = _workspace_files(state, scenario)
    target = scenario["workflow_target"]

    investigation_raw = _investigation_score(
        state,
        {(action_type, _norm(name)) for action_type, name in target.get("investigation_targets", [])},
    )
    investigation = 0.2 * investigation_raw

    artifact_raw = _artifact_score(files, target)
    artifact_score = 0.3 * artifact_raw

    validator_targets = target.get("validator_targets", [])
    solve_validator_targets = target.get("solve_validator_targets", validator_targets)
    validator_status = state.get("validator_status", {})
    validator_hits = sum(1 for name in validator_targets if validator_status.get(name, {}).get("passed"))
    solve_validator_hits = sum(1 for name in solve_validator_targets if validator_status.get(name, {}).get("passed"))
    validator_score = 0.3 * (validator_hits / len(validator_targets) if validator_targets else 0.0)

    governance_raw = _governance_score(files, target)
    governance_score = 0.1 * governance_raw

    summary = state.get("submission", {}).get("summary", "")
    summary_raw = _summary_score(summary, target.get("summary_groups", []))
    summary_score = 0.1 * summary_raw
    solved = (
        state.get("submitted", False)
        and investigation_raw >= (2 / 3)
        and artifact_raw == 1.0
        and (solve_validator_hits == len(solve_validator_targets) if solve_validator_targets else True)
        and governance_raw == 1.0
        and summary_raw == 1.0
    )

    total = _strict_score(investigation + artifact_score + validator_score + governance_score + summary_score)
    return {
        "total": total,
        "solved": solved,
        "breakdown": {
            "investigation": round(investigation, 4),
            "artifact_score": round(artifact_score, 4),
            "validator_score": round(validator_score, 4),
            "governance_score": round(governance_score, 4),
            "summary_score": round(summary_score, 4),
        },
        "feedback": (
            f"investigation={investigation:.2f} artifact={artifact_score:.2f} "
            f"validators={validator_score:.2f} governance={governance_score:.2f}"
        ),
    }


def _grade_simulation_workflow(state: dict, scenario: dict) -> dict[str, Any]:
    files = _workspace_files(state, scenario)
    target = scenario["workflow_target"]
    runtime_status = state.get("runtime_status", {})

    investigation_raw = _investigation_score(
        state,
        {(action_type, _norm(name)) for action_type, name in target.get("investigation_targets", [])},
    )
    investigation = 0.15 * investigation_raw

    artifact_raw = _artifact_score(files, target)
    artifact_score = 0.25 * artifact_raw

    checks = runtime_status.get("checks", {})
    runtime_targets = target.get("solve_validator_targets", [])
    runtime_hits = sum(1 for name in runtime_targets if checks.get(name, False))
    runtime_score_raw = runtime_hits / len(runtime_targets) if runtime_targets else 0.0
    runtime_score = 0.35 * runtime_score_raw

    summary_raw = _summary_score(state.get("submission", {}).get("summary", ""), target.get("summary_groups", []))
    summary_score = 0.1 * summary_raw

    role_bonus = 0.05 if state.get("active_role") == "fixer" and state.get("runtime_status", {}).get("ran") else 0.0

    expected_columns = (
        scenario.get("simulation_target", {})
        or scenario.get("llm_draft", {}).get("simulation_target", {})
    ).get("required_output_columns", [])
    output_schema = state.get("output_schema", [])
    schema_score = 0.1 if list(output_schema) == list(expected_columns) else 0.0

    solved = (
        state.get("submitted", False)
        and artifact_raw == 1.0
        and runtime_score_raw == 1.0
        and summary_raw >= (2 / 3)
        and schema_score > 0.0
        and runtime_status.get("succeeded", False)
    )

    total = _strict_score(investigation + artifact_score + runtime_score + summary_score + role_bonus + schema_score)
    return {
        "total": total,
        "solved": solved,
        "breakdown": {
            "investigation": round(investigation, 4),
            "artifact_score": round(artifact_score, 4),
            "runtime_score": round(runtime_score, 4),
            "schema_score": round(schema_score, 4),
            "summary_score": round(summary_score, 4),
            "role_bonus": round(role_bonus, 4),
        },
        "feedback": (
            f"investigation={investigation:.2f} artifact={artifact_score:.2f} "
            f"runtime={runtime_score:.2f} schema={schema_score:.2f}"
        ),
    }


def grade(task_id: str, state: dict, scenario: dict) -> dict[str, Any]:
    graders = {
        "repair": _grade_pipeline_repair,
        "review": _grade_llm_patch_review,
        "workflow": _grade_workflow_shipping,
        "simulation": _grade_simulation_workflow,
    }
    try:
        grader_family = get_grader_family(task_id)
    except ValueError:
        grader_family = task_id
    fn = graders.get(grader_family)
    if fn is None:
        return {
            "total": 0.01,
            "solved": False,
            "breakdown": {},
            "feedback": f"Unknown task or grader family '{task_id}'",
        }
    return fn(state, scenario)
