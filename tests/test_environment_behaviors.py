from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline import action_is_valid, build_action, fallback_action, get_model_action
from graders import grade
from server.developer_control_room_environment import DeveloperControlRoomEnvironment
from tasks import get_scenario


SIMULATION_PIPELINE_YAML = """name: customer_daily_report_job
storage_path: mock_s3/staged/sales_orders.csv
raw_table: raw_sales_orders
load_sql: sql/load_raw.sql
build_sql: sql/build_table.sql
report_sql: sql/report_view.sql
final_view: customer_daily_report
"""

SIMULATION_LOAD_SQL = """create or replace view staged_sales_orders as
select
  order_id,
  customer_id,
  cast(order_date as date) as order_date,
  quantity,
  unit_price
from raw_sales_orders;
"""

SIMULATION_BUILD_SQL = """create or replace table customer_summary as
select
  customer_id,
  order_date,
  count(*) as order_count,
  sum(quantity * unit_price) as gross_revenue_usd
from staged_sales_orders
group by 1, 2;
"""

SIMULATION_REPORT_SQL = """create or replace view customer_daily_report as
select
  customer_id,
  order_date,
  order_count,
  gross_revenue_usd
from customer_summary;
"""


def run_episode_with_fallback(task_id: str, scenario_index: int) -> dict:
    env = DeveloperControlRoomEnvironment()
    observation = env.reset(task_id=task_id, scenario_index=scenario_index)
    while not env._state.done:
        action_dict = fallback_action(task_id, observation)
        observation = env.step(build_action(action_dict))
    return {
        "state": env._state.model_dump(),
        "scenario": env._scenario,
        "grade": grade(task_id, env._state.model_dump(), env._scenario),
    }


def _run_simulation_sequence(report_sql: str = SIMULATION_REPORT_SQL) -> dict:
    env = DeveloperControlRoomEnvironment()
    observation = env.reset(task_id="simulate_csv_report_workflow", scenario_index=0)

    for action in (
        {
            "action_type": "edit_file",
            "parameters": {"path": "pipelines/report_job.yaml", "content": SIMULATION_PIPELINE_YAML},
        },
        {
            "action_type": "edit_file",
            "parameters": {"path": "sql/load_raw.sql", "content": SIMULATION_LOAD_SQL},
        },
        {
            "action_type": "edit_file",
            "parameters": {"path": "sql/build_table.sql", "content": SIMULATION_BUILD_SQL},
        },
        {
            "action_type": "edit_file",
            "parameters": {"path": "sql/report_view.sql", "content": report_sql},
        },
    ):
        observation = env.step(build_action(action))

    return {"env": env, "observation": observation}


def test_repeated_file_reads_are_penalized() -> None:
    env = DeveloperControlRoomEnvironment()
    env.reset(task_id="repair_pipeline_execution", scenario_index=0)

    first = env.step(
        build_action(
            {
                "action_type": "read_file",
                "parameters": {"path": "pipelines/customer_events_ingest.yaml"},
            }
        )
    )
    second = env.step(
        build_action(
            {
                "action_type": "read_file",
                "parameters": {"path": "pipelines/customer_events_ingest.yaml"},
            }
        )
    )

    assert first.reward > 0
    assert second.reward < first.reward


def test_repeated_validator_spam_gets_worse() -> None:
    env = DeveloperControlRoomEnvironment()
    env.reset(task_id="repair_pipeline_execution", scenario_index=1)

    env.step(
        build_action(
            {
                "action_type": "read_file",
                "parameters": {"path": "ci/jenkins/orders_daily.groovy"},
            }
        )
    )
    env.step(
        build_action(
            {
                "action_type": "edit_file",
                "parameters": {
                    "path": "ci/jenkins/orders_daily.groovy",
                    "content": """pipeline {
  agent any
  stages {
    stage('Run') {
      steps {
        sh './scripts/run_orders_daily.sh'
      }
    }
  }
}
""",
                },
            }
        )
    )

    first = env.step(
        build_action(
            {
                "action_type": "run_validator",
                "parameters": {"validator": "jenkins_lint"},
            }
        )
    )
    second = env.step(
        build_action(
            {
                "action_type": "run_validator",
                "parameters": {"validator": "jenkins_lint"},
            }
        )
    )

    assert first.reward > 0
    assert second.reward <= 0
    assert second.reward < first.reward


def test_invalid_early_edits_and_validators_are_blocked_by_hybrid_guardrails() -> None:
    env = DeveloperControlRoomEnvironment()

    reporting_obs = env.reset(task_id="synthesize_data_product", scenario_index=1)
    assert not action_is_valid(
        {
            "action_type": "edit_file",
            "parameters": {"path": "pipelines/customer_margin_mart.yaml", "content": "x"},
        },
        reporting_obs,
    )

    repair_obs = env.reset(task_id="repair_pipeline_execution", scenario_index=4)
    assert not action_is_valid(
        {
            "action_type": "run_validator",
            "parameters": {"validator": "column_mapping_guard"},
        },
        repair_obs,
    )


def test_same_trajectory_gives_same_score() -> None:
    first = run_episode_with_fallback("repair_pipeline_execution", 0)
    second = run_episode_with_fallback("repair_pipeline_execution", 0)

    assert first["state"]["action_history"] == second["state"]["action_history"]
    assert first["state"]["cumulative_reward"] == second["state"]["cumulative_reward"]
    assert first["grade"]["total"] == second["grade"]["total"]
    assert first["grade"]["solved"] == second["grade"]["solved"]


def test_hard_tasks_mix_solved_and_unsolved_scenarios() -> None:
    solved_reporting = run_episode_with_fallback("synthesize_reporting_asset", 0)["grade"]
    unsolved_reporting = run_episode_with_fallback("synthesize_reporting_asset", 1)["grade"]
    unsolved_product = run_episode_with_fallback("synthesize_data_product", 0)["grade"]
    solved_product = run_episode_with_fallback("synthesize_data_product", 1)["grade"]

    assert solved_reporting["solved"] is True
    assert unsolved_reporting["solved"] is False
    assert unsolved_product["solved"] is False
    assert solved_product["solved"] is True
    assert solved_reporting["total"] > unsolved_reporting["total"]
    assert solved_product["total"] > unsolved_product["total"]


def test_reset_produces_clean_state() -> None:
    env = DeveloperControlRoomEnvironment()
    env.reset(task_id="repair_data_transform", scenario_index=0)
    env.step(
        build_action(
            {
                "action_type": "read_file",
                "parameters": {"path": "transforms/orders_daily.sql"},
            }
        )
    )
    env.step(
        build_action(
            {
                "action_type": "inspect_schema",
                "parameters": {"asset": "raw.orders_events"},
            }
        )
    )

    fresh = env.reset(task_id="review_ai_patch_safety", scenario_index=0)

    assert fresh.step_count == 0
    assert fresh.cumulative_reward == 0.0
    assert fresh.queried_data == {}
    assert fresh.edited_files == {}
    assert fresh.validator_status == {}
    assert fresh.last_action_error is None


def test_forced_model_timeout_returns_none(monkeypatch) -> None:
    env = DeveloperControlRoomEnvironment()
    observation = env.reset(task_id="repair_data_transform", scenario_index=0)

    monkeypatch.setenv("DEVELOPER_CONTROL_ROOM_FORCE_MODEL_TIMEOUT", "true")
    result = get_model_action(
        client=None,  # type: ignore[arg-type]
        model_name="demo-model",
        temperature=0.2,
        max_tokens=32,
        timeout_seconds=0.01,
        step=1,
        observation=observation,
        history=[],
    )
    assert result is None


def test_seeded_variants_are_deterministic_and_change_literals() -> None:
    first = get_scenario("review_ai_patch_correctness", 1, seed=1)
    second = get_scenario("review_ai_patch_correctness", 1, seed=1)
    third = get_scenario("review_ai_patch_correctness", 1, seed=2)

    assert first["developer_request"] == second["developer_request"]
    assert first["files"]["schemas/subscription_mrr_daily.json"] == second["files"]["schemas/subscription_mrr_daily.json"]
    assert first["developer_request"] != third["developer_request"]


def test_review_submission_reward_is_capped() -> None:
    env = DeveloperControlRoomEnvironment()
    observation = env.reset(task_id="review_ai_patch_safety", scenario_index=0)

    for _ in range(3):
        action_dict = fallback_action("review_ai_patch_safety", observation)
        observation = env.step(build_action(action_dict))

    submit_action = fallback_action("review_ai_patch_safety", observation)
    result = env.step(build_action(submit_action))

    assert submit_action["action_type"] == "submit_review"
    assert result.reward <= (
        DeveloperControlRoomEnvironment.MAX_PROGRESS_DELTA_REWARD
        + DeveloperControlRoomEnvironment.SOLVED_TERMINAL_REWARD_BONUS
        + 0.02
    )


def test_repair_edit_progress_reward_is_capped() -> None:
    env = DeveloperControlRoomEnvironment()
    observation = env.reset(task_id="repair_data_transform", scenario_index=0)

    for _ in range(2):
        action_dict = fallback_action("repair_data_transform", observation)
        observation = env.step(build_action(action_dict))

    edit_action = fallback_action("repair_data_transform", observation)
    result = env.step(build_action(edit_action))

    assert edit_action["action_type"] == "edit_file"
    assert result.reward <= DeveloperControlRoomEnvironment.MAX_PROGRESS_DELTA_REWARD


def test_solved_episode_gets_terminal_bonus() -> None:
    env = DeveloperControlRoomEnvironment()
    observation = env.reset(task_id="review_ai_patch_safety", scenario_index=0)

    while not env._state.done:
        action_dict = fallback_action("review_ai_patch_safety", observation)
        observation = env.step(build_action(action_dict))

    assert observation.done is True
    assert observation.reward >= DeveloperControlRoomEnvironment.SOLVED_TERMINAL_REWARD_BONUS


def test_simulation_reset_exposes_runtime_context() -> None:
    env = DeveloperControlRoomEnvironment()
    observation = env.reset(task_id="simulate_csv_report_workflow", scenario_index=0)

    assert observation.active_role == "builder"
    assert "data/sales_orders.csv" in observation.known_files
    assert "pipelines/report_job.yaml" in observation.editable_targets
    assert "storage_stage_check" in observation.available_validators
    assert observation.runtime_status == {}


def test_simulation_runtime_happy_path_executes_end_to_end() -> None:
    pytest.importorskip("duckdb")
    run = _run_simulation_sequence()
    env = run["env"]
    observation = run["observation"]

    assert observation.active_role == "fixer"
    assert observation.runtime_status.get("succeeded") is True
    assert observation.output_schema == [
        "customer_id",
        "order_date",
        "order_count",
        "gross_revenue_usd",
    ]
    assert observation.report_preview

    for validator in (
        "storage_stage_check",
        "duckdb_load_check",
        "report_view_check",
        "output_schema_check",
    ):
        observation = env.step(
            build_action(
                {
                    "action_type": "run_validator",
                    "parameters": {"validator": validator},
                }
            )
        )
        assert observation.validator_status[validator]["passed"] is True

    observation = env.step(
        build_action(
            {
                "action_type": "submit_workspace",
                "parameters": {
                    "summary": "Built a DuckDB pipeline from sales_orders.csv and published customer_daily_report."
                },
            }
        )
    )
    final_grade = grade("simulate_csv_report_workflow", env._state.model_dump(), env._scenario)

    assert observation.done is True
    assert final_grade["solved"] is True


def test_simulation_fixer_can_repair_broken_report_view() -> None:
    pytest.importorskip("duckdb")
    broken_report_sql = """create or replace view customer_daily_report as
select
  customer_id,
  order_date,
  order_count,
  gross_revenue_usd as gross_revenue
from customer_summary;
"""

    run = _run_simulation_sequence(report_sql=broken_report_sql)
    env = run["env"]
    observation = run["observation"]

    assert observation.active_role == "fixer"
    assert observation.runtime_status.get("checks", {}).get("report_view_check") is True
    assert observation.runtime_status.get("checks", {}).get("output_schema_check") is False

    observation = env.step(
        build_action(
            {
                "action_type": "edit_file",
                "parameters": {"path": "sql/report_view.sql", "content": SIMULATION_REPORT_SQL},
            }
        )
    )

    assert observation.runtime_status.get("succeeded") is True
    assert observation.output_schema[-1] == "gross_revenue_usd"


def test_simulation_sequence_replays_deterministically() -> None:
    pytest.importorskip("duckdb")
    first = _run_simulation_sequence()
    second = _run_simulation_sequence()

    first_env = first["env"]
    second_env = second["env"]

    assert first_env._state.action_history == second_env._state.action_history
    assert first_env._state.runtime_status == second_env._state.runtime_status
    assert first_env._state.report_preview == second_env._state.report_preview
    assert grade("simulate_csv_report_workflow", first_env._state.model_dump(), first_env._scenario) == grade(
        "simulate_csv_report_workflow",
        second_env._state.model_dump(),
        second_env._scenario,
    )
