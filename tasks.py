"""Task and scenario definitions for the Developer Control Room benchmark."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def repair_validator(
    description: str,
    path: str,
    required_groups: list[list[str]] | None = None,
    forbidden_terms: list[str] | None = None,
) -> dict:
    return {
        "description": description,
        "kind": "file_groups",
        "path": path,
        "required_groups": required_groups or [],
        "forbidden_terms": forbidden_terms or [],
    }


def review_validator(
    description: str,
    draft_id: str,
    required_groups: list[list[str]] | None = None,
    forbidden_terms: list[str] | None = None,
) -> dict:
    return {
        "description": description,
        "kind": "draft_groups",
        "draft_id": draft_id,
        "required_groups": required_groups or [],
        "forbidden_terms": forbidden_terms or [],
    }


def workflow_validator(
    description: str,
    artifact: str,
) -> dict:
    return {
        "description": description,
        "kind": "workflow_artifact",
        "artifact": artifact,
    }


def repair_scenario(
    scenario_id: str,
    developer_request: str,
    workspace_summary: str,
    known_files: list[str],
    editable_targets: list[str],
    known_assets: list[str],
    available_validators: list[str],
    files: dict[str, str],
    schema_registry: dict[str, str],
    repair_target: dict,
    validators: dict[str, dict],
    lineage: dict[str, str] | None = None,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "developer_request": developer_request,
        "workspace_summary": workspace_summary,
        "known_files": known_files,
        "editable_targets": editable_targets,
        "known_assets": known_assets,
        "available_validators": available_validators,
        "files": files,
        "schema_registry": schema_registry,
        "lineage": lineage or {},
        "llm_draft": {},
        "repair_target": repair_target,
        "validators": validators,
    }


def review_scenario(
    scenario_id: str,
    developer_request: str,
    workspace_summary: str,
    known_files: list[str],
    known_assets: list[str],
    available_validators: list[str],
    files: dict[str, str],
    schema_registry: dict[str, str],
    llm_draft: dict[str, str],
    review_target: dict,
    validators: dict[str, dict],
) -> dict:
    return {
        "scenario_id": scenario_id,
        "developer_request": developer_request,
        "workspace_summary": workspace_summary,
        "known_files": known_files,
        "editable_targets": [],
        "known_assets": known_assets,
        "available_validators": available_validators,
        "files": files,
        "schema_registry": schema_registry,
        "lineage": {},
        "llm_draft": llm_draft,
        "review_target": review_target,
        "validators": validators,
    }


def workflow_artifact(
    path: str,
    required_groups: list[list[str]],
    weight: float,
    forbidden_terms: list[str] | None = None,
) -> dict:
    return {
        "path": path,
        "required_groups": required_groups,
        "forbidden_terms": forbidden_terms or [],
        "weight": weight,
    }


def workflow_scenario(
    scenario_id: str,
    developer_request: str,
    workspace_summary: str,
    known_files: list[str],
    editable_targets: list[str],
    known_assets: list[str],
    available_validators: list[str],
    files: dict[str, str],
    schema_registry: dict[str, str],
    workflow_target: dict,
    validators: dict[str, dict],
    lineage: dict[str, str] | None = None,
    llm_draft: dict[str, str] | None = None,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "developer_request": developer_request,
        "workspace_summary": workspace_summary,
        "known_files": known_files,
        "editable_targets": editable_targets,
        "known_assets": known_assets,
        "available_validators": available_validators,
        "files": files,
        "schema_registry": schema_registry,
        "lineage": lineage or {},
        "llm_draft": llm_draft or {},
        "workflow_target": workflow_target,
        "validators": validators,
    }


def simulation_scenario(
    *,
    scenario_id: str,
    developer_request: str,
    workspace_summary: str,
    source_csv_path: str,
    source_csv_content: str,
    raw_table: str,
    staged_view: str,
    build_table: str,
    final_view: str,
    raw_asset: str,
    final_asset: str,
    date_column: str,
    raw_schema_columns: list[tuple[str, str]],
    required_output_columns: list[str],
    build_metric_groups: list[list[str]],
    report_terms: list[str],
    correct_load_sql: str,
    correct_build_sql: str,
    correct_report_sql: str,
    starter_files: dict[str, str] | None = None,
    editable_targets_override: list[str] | None = None,
    available_validators_override: list[str] | None = None,
    investigation_targets_override: list[tuple[str, str]] | None = None,
    expected_raw_preview: list[dict[str, object]] | None = None,
    expected_derived_preview: list[dict[str, object]] | None = None,
    expected_final_preview: list[dict[str, object]] | None = None,
) -> dict:
    starter_files = starter_files or {}
    editable_targets = editable_targets_override if editable_targets_override is not None else [
        "pipelines/report_job.yaml",
        "sql/load_raw.sql",
        "sql/build_table.sql",
        "sql/report_view.sql",
    ]
    available_validators = available_validators_override if available_validators_override is not None else [
        "storage_stage_check",
        "duckdb_load_check",
        "report_view_check",
        "output_schema_check",
    ]
    investigation_targets = investigation_targets_override if investigation_targets_override is not None else [
        ("read_file", "docs/runtime_contract.md"),
        ("read_file", source_csv_path),
        ("inspect_schema", raw_asset),
    ]
    template_text = "\n".join(
        [
            f"name: {final_view}_job",
            f"storage_path: mock_s3/staged/{Path(source_csv_path).name}",
            f"raw_table: {raw_table}",
            "load_sql: sql/load_raw.sql",
            "build_sql: sql/build_table.sql",
            "report_sql: sql/report_view.sql",
            f"final_view: {final_view}",
            "",
        ]
    )
    runtime_contract = "\n".join(
        [
            "# Runtime contract",
            "",
            f"- The runtime stages `{source_csv_path}` into a local mock S3 path.",
            "- `pipelines/report_job.yaml` must define:",
            "  - storage_path",
            "  - raw_table",
            "  - load_sql",
            "  - build_sql",
            "  - report_sql",
            "  - final_view",
            "- The runtime auto-loads the staged CSV into the `raw_table`.",
            f"- `sql/load_raw.sql` must create a staged view named `{staged_view}`.",
            f"- `sql/build_table.sql` must create a derived table named `{build_table}`.",
            f"- `sql/report_view.sql` must create a final view named `{final_view}`.",
            f"- The final view must expose columns: {', '.join(required_output_columns)}",
            "",
        ]
    )
    schema_json_columns = ",\n".join(
        f'    {{"name": "{name}", "type": "{type_name}"}}' for name, type_name in raw_schema_columns
    )
    schema_yaml_columns = "\n".join(f"  - {name}: {type_name}" for name, type_name in raw_schema_columns)

    files = {
        source_csv_path: source_csv_content,
        "docs/runtime_contract.md": runtime_contract,
        "templates/report_pipeline_template.yaml": template_text,
        f"schemas/{raw_table}.json": "{\n"
        f'  "name": "{raw_table}",\n'
        '  "columns": [\n'
        f"{schema_json_columns}\n"
        "  ]\n"
        "}\n",
    }
    files.update(starter_files)

    known_files = [
        source_csv_path,
        "docs/runtime_contract.md",
        "templates/report_pipeline_template.yaml",
        f"schemas/{raw_table}.json",
    ]
    for path in ["pipelines/report_job.yaml", "sql/load_raw.sql", "sql/build_table.sql", "sql/report_view.sql"]:
        if path in starter_files:
            known_files.append(path)

    scenario = workflow_scenario(
        scenario_id=scenario_id,
        developer_request=developer_request,
        workspace_summary=workspace_summary,
        known_files=known_files,
        editable_targets=editable_targets,
        known_assets=[raw_asset, final_asset],
        available_validators=available_validators,
        files=files,
        schema_registry={
            raw_asset: f"asset: {raw_asset}\ncolumns:\n{schema_yaml_columns}\n",
            final_asset: (
                f"asset: {final_asset}\nrequired_columns:\n"
                + "".join(f"  - {column}\n" for column in required_output_columns)
            ),
        },
        workflow_target={
            "required_artifacts": ["pipeline_yaml", "load_sql", "build_sql", "report_sql"],
            "artifacts": {
                "pipeline_yaml": workflow_artifact(
                    "pipelines/report_job.yaml",
                    required_groups=[
                        [f"storage_path: mock_s3/staged/{Path(source_csv_path).name}"],
                        [f"raw_table: {raw_table}"],
                        ["load_sql: sql/load_raw.sql"],
                        ["build_sql: sql/build_table.sql"],
                        ["report_sql: sql/report_view.sql"],
                        [f"final_view: {final_view}"],
                    ],
                    weight=1.0,
                ),
                "load_sql": workflow_artifact(
                    "sql/load_raw.sql",
                    required_groups=[
                        ["create", staged_view],
                        [raw_table],
                        [f"cast({date_column} as date)", f"try_cast({date_column} as date)", date_column],
                    ],
                    weight=1.0,
                ),
                "build_sql": workflow_artifact(
                    "sql/build_table.sql",
                    required_groups=[
                        ["create", build_table],
                        [staged_view],
                        *build_metric_groups,
                    ],
                    weight=1.2,
                ),
                "report_sql": workflow_artifact(
                    "sql/report_view.sql",
                    required_groups=[
                        ["create", final_view],
                        [build_table],
                        *[[column] for column in required_output_columns],
                    ],
                    weight=1.2,
                ),
            },
            "validator_targets": list(available_validators),
            "solve_validator_targets": [
                "storage_stage_check",
                "duckdb_load_check",
                "report_view_check",
                "output_schema_check",
            ],
            "summary_groups": [
                [Path(source_csv_path).name],
                [final_view],
                *[[term] for term in report_terms],
            ],
            "investigation_targets": investigation_targets,
        },
        validators={
            "storage_stage_check": {
                "description": "Checks that the CSV could be staged into mock storage.",
                "kind": "runtime_check",
                "check_key": "storage_stage_check",
            },
            "duckdb_load_check": {
                "description": "Checks that the staged CSV loads into DuckDB.",
                "kind": "runtime_check",
                "check_key": "duckdb_load_check",
            },
            "report_view_check": {
                "description": "Checks that the final report view compiles.",
                "kind": "runtime_check",
                "check_key": "report_view_check",
            },
            "output_schema_check": {
                "description": "Checks that the final report view exposes the expected schema.",
                "kind": "runtime_check",
                "check_key": "output_schema_check",
            },
        },
        llm_draft={},
    )
    scenario["simulation_target"] = {
        "source_csv": source_csv_path,
        "pipeline_path": "pipelines/report_job.yaml",
        "load_sql_path": "sql/load_raw.sql",
        "build_sql_path": "sql/build_table.sql",
        "report_sql_path": "sql/report_view.sql",
        "required_output_columns": required_output_columns,
        "expected_raw_table": {
            "name": raw_table,
            "preview_rows": expected_raw_preview or [],
        },
        "expected_derived_table": {
            "name": build_table,
            "preview_rows": expected_derived_preview or [],
        },
        "expected_final_view": {
            "name": final_view,
            "preview_rows": expected_final_preview or [],
        },
    }
    scenario["llm_draft"]["simulation_target"] = deepcopy(scenario["simulation_target"])
    scenario["llm_draft"]["reference_solution"] = {
        "pipeline_yaml": template_text,
        "load_sql": correct_load_sql,
        "build_sql": correct_build_sql,
        "report_sql": correct_report_sql,
    }
    return scenario


TASK_DEFINITIONS: dict[str, dict] = {
    "repair_data_transform": {
        "id": "repair_data_transform",
        "name": "Task 1: Repair Data Transform",
        "difficulty": "easy",
        "max_steps": 8,
        "score_range": [0.0, 1.0],
        "description": (
            "Repair transform logic or dependency wiring while keeping downstream "
            "warehouse contracts stable."
        ),
        "available_actions": [
            "search_workspace",
            "read_file",
            "inspect_schema",
            "run_validator",
            "edit_file",
            "submit_repair",
        ],
        "submission_action": "submit_repair",
        "grader_family": "repair",
        "scenario_ids": ["PR-001", "PR-004"],
    },
    "repair_pipeline_execution": {
        "id": "repair_pipeline_execution",
        "name": "Task 2: Repair Pipeline Execution",
        "difficulty": "easy",
        "max_steps": 8,
        "score_range": [0.0, 1.0],
        "description": (
            "Recover ingestion or orchestration failures caused by storage layout, "
            "runner, or deploy-configuration drift."
        ),
        "available_actions": [
            "search_workspace",
            "read_file",
            "inspect_schema",
            "run_validator",
            "edit_file",
            "submit_repair",
        ],
        "submission_action": "submit_repair",
        "grader_family": "repair",
        "scenario_ids": ["PR-002", "PR-003", "PR-005", "PR-006", "PR-007", "PR-008"],
    },
    "review_ai_patch_safety": {
        "id": "review_ai_patch_safety",
        "name": "Task 3: Review AI Patch Safety",
        "difficulty": "medium",
        "max_steps": 8,
        "score_range": [0.0, 1.0],
        "description": (
            "Review AI-generated changes for governance, privacy, and operational "
            "safety risks before they reach production."
        ),
        "available_actions": [
            "search_workspace",
            "read_file",
            "inspect_schema",
            "inspect_llm_draft",
            "run_validator",
            "submit_review",
        ],
        "submission_action": "submit_review",
        "grader_family": "review",
        "scenario_ids": ["LR-001", "LR-004"],
    },
    "review_ai_patch_correctness": {
        "id": "review_ai_patch_correctness",
        "name": "Task 4: Review AI Patch Correctness",
        "difficulty": "medium",
        "max_steps": 8,
        "score_range": [0.0, 1.0],
        "description": (
            "Review AI-generated changes for semantic correctness and compatibility "
            "with published warehouse contracts."
        ),
        "available_actions": [
            "search_workspace",
            "read_file",
            "inspect_schema",
            "inspect_llm_draft",
            "run_validator",
            "submit_review",
        ],
        "submission_action": "submit_review",
        "grader_family": "review",
        "scenario_ids": ["LR-002", "LR-003"],
    },
    "synthesize_reporting_asset": {
        "id": "synthesize_reporting_asset",
        "name": "Task 5: Synthesize Reporting Asset",
        "difficulty": "hard",
        "max_steps": 12,
        "score_range": [0.0, 1.0],
        "description": (
            "Create reporting-facing assets such as views, contracts, and alerts "
            "that are ready for production use."
        ),
        "available_actions": [
            "search_workspace",
            "read_file",
            "inspect_schema",
            "inspect_lineage",
            "inspect_llm_draft",
            "edit_file",
            "run_validator",
            "submit_workspace",
        ],
        "submission_action": "submit_workspace",
        "grader_family": "workflow",
        "scenario_ids": ["WS-002", "WS-004"],
    },
    "synthesize_data_product": {
        "id": "synthesize_data_product",
        "name": "Task 6: Synthesize Data Product",
        "difficulty": "hard",
        "max_steps": 14,
        "score_range": [0.0, 1.0],
        "description": (
            "Create production-grade data products with pipeline metadata, SQL, "
            "checks, contracts, and governance-safe outputs."
        ),
        "available_actions": [
            "search_workspace",
            "read_file",
            "inspect_schema",
            "inspect_lineage",
            "inspect_llm_draft",
            "edit_file",
            "run_validator",
            "submit_workspace",
        ],
        "submission_action": "submit_workspace",
        "grader_family": "workflow",
        "scenario_ids": ["WS-001", "WS-003"],
    },
    "simulate_csv_report_workflow": {
        "id": "simulate_csv_report_workflow",
        "name": "Task 7: Simulate CSV Report Workflow",
        "difficulty": "hard",
        "max_steps": 12,
        "score_range": [0.0, 1.0],
        "description": (
            "Generate and repair a small CSV-to-report data workflow that stages "
            "source data, loads it into DuckDB, and publishes a final report view."
        ),
        "available_actions": [
            "search_workspace",
            "read_file",
            "inspect_schema",
            "edit_file",
            "run_validator",
            "submit_workspace",
        ],
        "submission_action": "submit_workspace",
        "grader_family": "simulation",
        "scenario_ids": ["SIM-001", "SIM-002", "SIM-003", "SIM-004", "SIM-005", "SIM-006", "SIM-007"],
    },
    "simulate_csv_report_curriculum_generate": {
        "id": "simulate_csv_report_curriculum_generate",
        "name": "Task 8: Easy CSV Workflow Generation",
        "difficulty": "easy",
        "max_steps": 6,
        "score_range": [0.0, 1.0],
        "description": "Create one tiny final report SQL file for a daily signup count.",
        "available_actions": [
            "read_file",
            "inspect_schema",
            "edit_file",
            "submit_workspace",
        ],
        "submission_action": "submit_workspace",
        "grader_family": "simulation",
        "scenario_ids": ["SIM-E001"],
    },
    "simulate_csv_report_curriculum_repair": {
        "id": "simulate_csv_report_curriculum_repair",
        "name": "Task 9: Easy CSV Workflow Repair",
        "difficulty": "easy",
        "max_steps": 6,
        "score_range": [0.0, 1.0],
        "description": "Repair one tiny final report SQL file with a single alias bug.",
        "available_actions": [
            "read_file",
            "inspect_schema",
            "edit_file",
            "submit_workspace",
        ],
        "submission_action": "submit_workspace",
        "grader_family": "simulation",
        "scenario_ids": ["SIM-E002"],
    },
}

ALL_TASKS = TASK_DEFINITIONS


SCENARIOS: dict[str, list[dict]] = {
    "pipeline_repair": [
        repair_scenario(
            scenario_id="PR-001",
            developer_request=(
                "The nightly `orders_daily` build started failing after an upstream "
                "change. Repair the workspace so the job compiles again, but keep "
                "the published contract stable for downstream dashboards."
            ),
            workspace_summary=(
                "This repo keeps transform logic in `transforms/`, job configs in "
                "`pipelines/`, and published contracts in `schemas/`. Validators "
                "`sql_compile` and `contract_guard` are available."
            ),
            known_files=[
                "pipelines/orders_daily.yaml",
                "transforms/orders_daily.sql",
                "schemas/orders_daily.json",
                "docs/pipeline_conventions.md",
            ],
            editable_targets=["transforms/orders_daily.sql"],
            known_assets=["raw.orders_events", "dim.customers", "orders_daily"],
            available_validators=["sql_compile", "contract_guard"],
            files={
                "pipelines/orders_daily.yaml": """name: orders_daily
schedule: "0 2 * * *"
owner: analytics-platform
sources:
  - raw.orders_events
  - dim.customers
target: orders_daily
contract: schemas/orders_daily.json
""",
                "transforms/orders_daily.sql": """with base as (
    select
        o.order_id,
        o.customer_id,
        o.event_ts as order_ts,
        date(o.event_ts) as order_date,
        o.amount_cents / 100.0 as total_amount
    from raw.orders_events o
    where o.status = 'completed'
)
select * from base;
""",
                "schemas/orders_daily.json": """{
  "name": "orders_daily",
  "columns": [
    {"name": "order_id", "type": "string"},
    {"name": "customer_id", "type": "string"},
    {"name": "order_ts", "type": "timestamp"},
    {"name": "order_date", "type": "date"},
    {"name": "total_amount", "type": "float"}
  ]
}
""",
                "docs/pipeline_conventions.md": """# Pipeline conventions

- Keep published column names stable when upstream systems rename fields.
- Prefer aliasing upstream renames inside SQL rather than changing the contract.
- `sql_compile` fails on unresolved columns.
""",
            },
            schema_registry={
                "raw.orders_events": """asset: raw.orders_events
columns:
  - order_id: string
  - customer_id: string
  - event_time: timestamp
  - amount_cents: bigint
  - status: string
notes:
  - Column `event_ts` was renamed to `event_time` in the latest ingestion deploy.
""",
                "dim.customers": """asset: dim.customers
columns:
  - customer_id: string
  - segment: string
""",
                "orders_daily": """asset: orders_daily
published_columns:
  - order_id
  - customer_id
  - order_ts
  - order_date
  - total_amount
""",
            },
            repair_target={
                "path": "transforms/orders_daily.sql",
                "required_groups": [["event_time"], ["as order_ts"], ["order_date"]],
                "forbidden_terms": ["event_ts"],
                "root_cause_groups": [["schema", "column"], ["event_ts"], ["event_time", "rename"]],
                "summary_groups": [["order_ts", "contract"], ["event_time"], ["orders_daily"]],
                "fix_path": "transforms/orders_daily.sql",
                "investigation_targets": [
                    ("read_file", "transforms/orders_daily.sql"),
                    ("inspect_schema", "raw.orders_events"),
                    ("run_validator", "sql_compile"),
                ],
            },
            validators={
                "sql_compile": repair_validator(
                    "Checks unresolved columns in the SQL transform.",
                    "transforms/orders_daily.sql",
                    required_groups=[["event_time"], ["as order_ts"]],
                    forbidden_terms=["event_ts"],
                ),
                "contract_guard": repair_validator(
                    "Confirms the output contract still publishes `order_ts`.",
                    "transforms/orders_daily.sql",
                    required_groups=[["as order_ts"], ["order_date"]],
                ),
            },
        ),
        repair_scenario(
            scenario_id="PR-002",
            developer_request=(
                "The `customer_events_ingest` job started failing after the data lake "
                "team changed the ADLS folder convention. Repair the config without "
                "renaming the target dataset."
            ),
            workspace_summary=(
                "Landing-zone jobs read from ADLS paths declared in `pipelines/`. "
                "Validators `path_resolver` and `target_guard` are available."
            ),
            known_files=[
                "pipelines/customer_events_ingest.yaml",
                "schemas/customer_events_raw.json",
                "docs/storage_layout.md",
            ],
            editable_targets=["pipelines/customer_events_ingest.yaml"],
            known_assets=["landing.customer_events", "customer_events_raw"],
            available_validators=["path_resolver", "target_guard"],
            files={
                "pipelines/customer_events_ingest.yaml": """name: customer_events_ingest
schedule: "0 * * * *"
owner: ingestion-platform
source_path: "abfss://bronze@datalake.dfs.core.windows.net/customer-events/date={{ ds }}/"
format: parquet
target: customer_events_raw
""",
                "schemas/customer_events_raw.json": """{
  "name": "customer_events_raw",
  "columns": [
    {"name": "event_id", "type": "string"},
    {"name": "customer_id", "type": "string"},
    {"name": "event_type", "type": "string"},
    {"name": "event_ts", "type": "timestamp"}
  ]
}
""",
                "docs/storage_layout.md": """# Storage layout

- Customer events now land under `customer-events/v2/`.
- Daily partitions use `dt={{ ds }}` instead of `date={{ ds }}`.
- Downstream dataset names must stay unchanged across ingestion path migrations.
""",
            },
            schema_registry={
                "landing.customer_events": """asset: landing.customer_events
location:
  abfss://bronze@datalake.dfs.core.windows.net/customer-events/v2/dt={{ ds }}/
format: parquet
""",
                "customer_events_raw": """asset: customer_events_raw
published_columns:
  - event_id
  - customer_id
  - event_type
  - event_ts
""",
            },
            repair_target={
                "path": "pipelines/customer_events_ingest.yaml",
                "required_groups": [["customer-events/v2"], ["dt={{ ds }}", "dt={ds}"], ["target: customer_events_raw"]],
                "forbidden_terms": ["date={{ ds }}"],
                "root_cause_groups": [["path", "folder", "partition"], ["date"], ["dt"]],
                "summary_groups": [["customer-events/v2"], ["customer_events_raw"]],
                "fix_path": "pipelines/customer_events_ingest.yaml",
                "investigation_targets": [
                    ("read_file", "pipelines/customer_events_ingest.yaml"),
                    ("read_file", "docs/storage_layout.md"),
                    ("run_validator", "path_resolver"),
                ],
            },
            validators={
                "path_resolver": repair_validator(
                    "Checks that the ADLS path uses the new folder and partition layout.",
                    "pipelines/customer_events_ingest.yaml",
                    required_groups=[["customer-events/v2"], ["dt={{ ds }}", "dt={ds}"]],
                    forbidden_terms=["date={{ ds }}"],
                ),
                "target_guard": repair_validator(
                    "Confirms the target dataset stays stable during the repair.",
                    "pipelines/customer_events_ingest.yaml",
                    required_groups=[["target: customer_events_raw"]],
                ),
            },
        ),
        repair_scenario(
            scenario_id="PR-003",
            developer_request=(
                "The Jenkins release job for `orders_daily` started failing after a CI "
                "cleanup. Fix the job definition so it invokes the packaged runner again."
            ),
            workspace_summary=(
                "CI jobs are declared in `ci/jenkins/`. Validators `jenkins_lint` and "
                "`runner_guard` are available."
            ),
            known_files=[
                "ci/jenkins/orders_daily.groovy",
                "docs/ci_conventions.md",
                "scripts/run_orders_daily.sh",
            ],
            editable_targets=["ci/jenkins/orders_daily.groovy"],
            known_assets=["ci.orders_daily", "artifact.orders_daily"],
            available_validators=["jenkins_lint", "runner_guard"],
            files={
                "ci/jenkins/orders_daily.groovy": """pipeline {
  agent any
  stages {
    stage('Run') {
      steps {
        sh './scripts/run-orders-daily.sh'
      }
    }
  }
}
""",
                "docs/ci_conventions.md": """# CI conventions

- Release jobs call the packaged runner script under `./scripts/`.
- Script names use underscores, not hyphens.
- Jobs should execute `./scripts/run_orders_daily.sh` for the orders_daily release.
""",
                "scripts/run_orders_daily.sh": """#!/usr/bin/env bash
set -euo pipefail
echo "running orders daily"
""",
            },
            schema_registry={
                "ci.orders_daily": """asset: ci.orders_daily
expected_runner:
  ./scripts/run_orders_daily.sh
""",
                "artifact.orders_daily": """asset: artifact.orders_daily
package:
  orders_daily.tar.gz
""",
            },
            repair_target={
                "path": "ci/jenkins/orders_daily.groovy",
                "required_groups": [["./scripts/run_orders_daily.sh"]],
                "forbidden_terms": ["run-orders-daily.sh"],
                "root_cause_groups": [["jenkins", "ci"], ["script", "runner"], ["hyphen", "path", "name"]],
                "summary_groups": [["run_orders_daily.sh"], ["jenkins", "runner"]],
                "fix_path": "ci/jenkins/orders_daily.groovy",
                "investigation_targets": [
                    ("read_file", "ci/jenkins/orders_daily.groovy"),
                    ("read_file", "docs/ci_conventions.md"),
                    ("run_validator", "jenkins_lint"),
                ],
            },
            validators={
                "jenkins_lint": repair_validator(
                    "Checks that the Jenkins job points at an existing runner script.",
                    "ci/jenkins/orders_daily.groovy",
                    required_groups=[["./scripts/run_orders_daily.sh"]],
                    forbidden_terms=["run-orders-daily.sh"],
                ),
                "runner_guard": repair_validator(
                    "Confirms the repair still uses the packaged runner convention.",
                    "ci/jenkins/orders_daily.groovy",
                    required_groups=[["sh"], ["./scripts/run_orders_daily.sh"]],
                ),
            },
        ),
        repair_scenario(
            scenario_id="PR-004",
            developer_request=(
                "The `revenue_dashboard_refresh` job is stale because its upstream "
                "dependency was renamed incorrectly during a rollout. Repair the "
                "dependency wiring so the dashboard refreshes again."
            ),
            workspace_summary=(
                "Downstream jobs declare upstreams in pipeline YAML. Validators "
                "`dependency_guard` and `freshness_guard` are available."
            ),
            known_files=[
                "pipelines/revenue_dashboard_refresh.yaml",
                "pipelines/orders_daily.yaml",
                "docs/lineage_notes.md",
            ],
            editable_targets=["pipelines/revenue_dashboard_refresh.yaml"],
            known_assets=["orders_daily", "orders_daily_v2", "revenue_dashboard_refresh"],
            available_validators=["dependency_guard", "freshness_guard"],
            files={
                "pipelines/revenue_dashboard_refresh.yaml": """name: revenue_dashboard_refresh
schedule: "0 5 * * *"
owner: finance-analytics
sources:
  - orders_daily_v2
target: revenue_dashboard_refresh
""",
                "pipelines/orders_daily.yaml": """name: orders_daily
schedule: "0 2 * * *"
target: orders_daily
""",
                "docs/lineage_notes.md": """# Lineage notes

- The rollout created a temporary `orders_daily_v2` alias, but the canonical
  upstream for production dashboards remains `orders_daily`.
- Downstream jobs should depend on canonical dataset names after the rollback.
""",
            },
            schema_registry={
                "orders_daily": """asset: orders_daily
freshness_sla: 3h
status: healthy
""",
                "orders_daily_v2": """asset: orders_daily_v2
status: deprecated_alias
""",
                "revenue_dashboard_refresh": """asset: revenue_dashboard_refresh
depends_on:
  - orders_daily
""",
            },
            repair_target={
                "path": "pipelines/revenue_dashboard_refresh.yaml",
                "required_groups": [["orders_daily"], ["target: revenue_dashboard_refresh"]],
                "forbidden_terms": ["orders_daily_v2"],
                "root_cause_groups": [["upstream", "dependency", "stale"], ["orders_daily_v2"], ["orders_daily", "canonical"]],
                "summary_groups": [["orders_daily"], ["revenue_dashboard_refresh"], ["deprecated", "alias"]],
                "fix_path": "pipelines/revenue_dashboard_refresh.yaml",
                "investigation_targets": [
                    ("read_file", "pipelines/revenue_dashboard_refresh.yaml"),
                    ("read_file", "docs/lineage_notes.md"),
                    ("run_validator", "dependency_guard"),
                ],
            },
            validators={
                "dependency_guard": repair_validator(
                    "Checks that the pipeline points at the canonical upstream dataset.",
                    "pipelines/revenue_dashboard_refresh.yaml",
                    required_groups=[["orders_daily"]],
                    forbidden_terms=["orders_daily_v2"],
                ),
                "freshness_guard": repair_validator(
                    "Confirms the dashboard target remains unchanged after the dependency fix.",
                    "pipelines/revenue_dashboard_refresh.yaml",
                    required_groups=[["target: revenue_dashboard_refresh"]],
                ),
            },
        ),
        repair_scenario(
            scenario_id="PR-005",
            developer_request=(
                "The reporting release deploy started failing after a config cleanup. "
                "Repair the deploy definition so it passes the expected warehouse "
                "role to the packaged release runner again."
            ),
            workspace_summary=(
                "Release definitions live under `deploy/`. Validators `deploy_lint` "
                "and `role_guard` are available."
            ),
            known_files=[
                "deploy/reporting_release.yaml",
                "docs/deploy_runbook.md",
                "scripts/release_reporting_asset.sh",
            ],
            editable_targets=["deploy/reporting_release.yaml"],
            known_assets=["deploy.reporting_release", "warehouse.reporting_release"],
            available_validators=["deploy_lint", "role_guard"],
            files={
                "deploy/reporting_release.yaml": """job: reporting_release
runner_image: reporting-release:latest
script: ./scripts/release_reporting_asset.sh
env:
  WAREHOUSE_ROLE: analytics_writer
  ALERT_ROUTE: finance-reporting
""",
                "docs/deploy_runbook.md": """# Reporting release runbook

- Reporting releases execute `./scripts/release_reporting_asset.sh`.
- Production deploys must pass `TARGET_WAREHOUSE_ROLE`.
- The reporting release role remains `analytics_writer`.
""",
                "scripts/release_reporting_asset.sh": """#!/usr/bin/env bash
set -euo pipefail
echo "releasing reporting asset"
""",
            },
            schema_registry={
                "deploy.reporting_release": """asset: deploy.reporting_release
expected_script:
  ./scripts/release_reporting_asset.sh
required_env:
  TARGET_WAREHOUSE_ROLE
""",
                "warehouse.reporting_release": """asset: warehouse.reporting_release
role: analytics_writer
""",
            },
            repair_target={
                "path": "deploy/reporting_release.yaml",
                "required_groups": [
                    ["TARGET_WAREHOUSE_ROLE"],
                    ["analytics_writer"],
                    ["./scripts/release_reporting_asset.sh"],
                ],
                "forbidden_terms": ["\nenv:\n  WAREHOUSE_ROLE:"],
                "root_cause_groups": [
                    ["deploy", "release"],
                    ["env", "environment", "variable"],
                    ["target_warehouse_role", "warehouse_role"],
                ],
                "summary_groups": [["TARGET_WAREHOUSE_ROLE"], ["analytics_writer"], ["release"]],
                "fix_path": "deploy/reporting_release.yaml",
                "investigation_targets": [
                    ("read_file", "deploy/reporting_release.yaml"),
                    ("read_file", "docs/deploy_runbook.md"),
                    ("read_file", "scripts/release_reporting_asset.sh"),
                    ("run_validator", "deploy_lint"),
                ],
            },
            validators={
                "deploy_lint": repair_validator(
                    "Checks that the release job uses the correct release script and env key.",
                    "deploy/reporting_release.yaml",
                    required_groups=[["./scripts/release_reporting_asset.sh"], ["TARGET_WAREHOUSE_ROLE"]],
                    forbidden_terms=["\nenv:\n  WAREHOUSE_ROLE:"],
                ),
                "role_guard": repair_validator(
                    "Confirms the release config still deploys with the analytics_writer role.",
                    "deploy/reporting_release.yaml",
                    required_groups=[["TARGET_WAREHOUSE_ROLE"], ["analytics_writer"]],
                ),
            },
        ),
        repair_scenario(
            scenario_id="PR-006",
            developer_request=(
                "The `customer_profile_import` batch started failing during load. The "
                "CSV still lands successfully, but the warehouse insert now errors "
                "because the incoming identifier shape no longer matches the target "
                "table contract. Repair the load mapping."
            ),
            workspace_summary=(
                "Batch loads stage CSVs in `samples/` and warehouse load SQL lives in "
                "`transforms/`. Validators `type_alignment_guard` and "
                "`load_contract_guard` are available."
            ),
            known_files=[
                "transforms/customer_profile_import.sql",
                "samples/customer_profile_import.csv",
                "schemas/customer_profile_dim.json",
                "docs/load_contracts.md",
            ],
            editable_targets=["transforms/customer_profile_import.sql"],
            known_assets=["customer_profile_stage", "customer_profile_dim"],
            available_validators=["type_alignment_guard", "load_contract_guard"],
            files={
                "transforms/customer_profile_import.sql": """insert into customer_profile_dim (
    customer_code,
    loyalty_tier,
    state_code
)
select
    cast(s.customer_code as integer) as customer_code,
    s.loyalty_tier,
    s.state_code
from customer_profile_stage s;
""",
                "samples/customer_profile_import.csv": """customer_code,loyalty_tier,state_code
000123,GOLD,CA
000124,SILVER,WA
""",
                "schemas/customer_profile_dim.json": """{
  "name": "customer_profile_dim",
  "columns": [
    {"name": "customer_code", "type": "string"},
    {"name": "loyalty_tier", "type": "string"},
    {"name": "state_code", "type": "string"}
  ]
}
""",
                "docs/load_contracts.md": """# Load contracts

- `customer_code` remains a string identifier because leading zeroes are significant.
- Batch imports may receive numeric-looking identifiers in CSV files, but warehouse
  inserts must preserve the original string representation.
""",
            },
            schema_registry={
                "customer_profile_stage": """asset: customer_profile_stage
columns:
  - customer_code: string
  - loyalty_tier: string
  - state_code: string
""",
                "customer_profile_dim": """asset: customer_profile_dim
published_columns:
  - customer_code:string
  - loyalty_tier:string
  - state_code:string
""",
            },
            repair_target={
                "path": "transforms/customer_profile_import.sql",
                "required_groups": [
                    ["insert into customer_profile_dim"],
                    ["customer_code"],
                    ["s.customer_code"],
                ],
                "forbidden_terms": ["cast(s.customer_code as integer)", "::integer", "as integer) as customer_code"],
                "root_cause_groups": [
                    ["customer_code"],
                    ["string", "varchar", "text"],
                    ["cast", "integer", "type mismatch"],
                ],
                "summary_groups": [["customer_code"], ["string", "leading zero"], ["load", "mapping"]],
                "fix_path": "transforms/customer_profile_import.sql",
                "investigation_targets": [
                    ("read_file", "transforms/customer_profile_import.sql"),
                    ("read_file", "samples/customer_profile_import.csv"),
                    ("read_file", "docs/load_contracts.md"),
                    ("read_file", "schemas/customer_profile_dim.json"),
                ],
            },
            validators={
                "type_alignment_guard": repair_validator(
                    "Checks that the load no longer casts customer_code to integer.",
                    "transforms/customer_profile_import.sql",
                    required_groups=[["s.customer_code"]],
                    forbidden_terms=["cast(s.customer_code as integer)", "::integer"],
                ),
                "load_contract_guard": repair_validator(
                    "Checks that the warehouse insert still targets the published string contract.",
                    "transforms/customer_profile_import.sql",
                    required_groups=[["customer_profile_dim"], ["customer_code"], ["loyalty_tier"], ["state_code"]],
                ),
            },
        ),
        repair_scenario(
            scenario_id="PR-007",
            developer_request=(
                "The `orders_archive_load` job is failing because the new CSV extract "
                "only ships 6 columns, while the archive table now has 12 columns. "
                "Repair the load configuration so the archive ingest succeeds again."
            ),
            workspace_summary=(
                "Archive loads map source columns in YAML and rely on default values "
                "for warehouse-only metadata fields. Validators `column_mapping_guard` "
                "and `archive_defaults_guard` are available."
            ),
            known_files=[
                "pipelines/orders_archive_load.yaml",
                "samples/orders_archive_extract.csv",
                "schemas/orders_archive.json",
                "docs/archive_load_guide.md",
            ],
            editable_targets=["pipelines/orders_archive_load.yaml"],
            known_assets=["orders_archive_stage", "orders_archive"],
            available_validators=["column_mapping_guard", "archive_defaults_guard"],
            files={
                "pipelines/orders_archive_load.yaml": """job: orders_archive_load
target_table: orders_archive
source_columns:
  - order_id
  - customer_id
  - order_ts
  - order_status
  - gross_amount_usd
  - currency_code
defaults: {}
""",
                "samples/orders_archive_extract.csv": """order_id,customer_id,order_ts,order_status,gross_amount_usd,currency_code
o-1001,c-11,2026-04-01T02:00:00Z,completed,129.0,USD
""",
                "schemas/orders_archive.json": """{
  "name": "orders_archive",
  "columns": [
    {"name": "order_id", "type": "string"},
    {"name": "customer_id", "type": "string"},
    {"name": "order_ts", "type": "timestamp"},
    {"name": "order_status", "type": "string"},
    {"name": "gross_amount_usd", "type": "float"},
    {"name": "currency_code", "type": "string"},
    {"name": "ingest_source", "type": "string"},
    {"name": "ingest_batch_id", "type": "string"},
    {"name": "created_at", "type": "timestamp"},
    {"name": "updated_at", "type": "timestamp"},
    {"name": "archive_status", "type": "string"},
    {"name": "notes", "type": "string"}
  ]
}
""",
                "docs/archive_load_guide.md": """# Archive load guide

- The extract only supplies 6 business columns.
- Warehouse-only fields must be populated through deterministic defaults:
  - ingest_source: csv_archive
  - ingest_batch_id: current_batch_id
  - created_at: current_timestamp
  - updated_at: current_timestamp
  - archive_status: active
  - notes: ''
""",
            },
            schema_registry={
                "orders_archive_stage": """asset: orders_archive_stage
extract_columns:
  - order_id
  - customer_id
  - order_ts
  - order_status
  - gross_amount_usd
  - currency_code
""",
                "orders_archive": """asset: orders_archive
required_columns: 12
warehouse_default_columns:
  - ingest_source
  - ingest_batch_id
  - created_at
  - updated_at
  - archive_status
  - notes
""",
            },
            repair_target={
                "path": "pipelines/orders_archive_load.yaml",
                "required_groups": [
                    ["order_id"],
                    ["customer_id"],
                    ["order_ts"],
                    ["order_status"],
                    ["gross_amount_usd"],
                    ["currency_code"],
                    ["ingest_source: csv_archive"],
                    ["ingest_batch_id: current_batch_id"],
                    ["created_at: current_timestamp"],
                    ["updated_at: current_timestamp"],
                    ["archive_status: active"],
                    ["notes: ''", "notes: \"\""],
                ],
                "forbidden_terms": ["defaults: {}"],
                "root_cause_groups": [
                    ["6", "six"],
                    ["12", "twelve"],
                    ["columns", "mapping"],
                ],
                "summary_groups": [["defaults"], ["12", "columns"], ["archive"], ["metadata"]],
                "fix_path": "pipelines/orders_archive_load.yaml",
                "investigation_targets": [
                    ("read_file", "pipelines/orders_archive_load.yaml"),
                    ("read_file", "schemas/orders_archive.json"),
                    ("read_file", "docs/archive_load_guide.md"),
                    ("read_file", "samples/orders_archive_extract.csv"),
                ],
            },
            validators={
                "column_mapping_guard": repair_validator(
                    "Checks that the archive load maps the 6 source columns and 6 deterministic defaults.",
                    "pipelines/orders_archive_load.yaml",
                    required_groups=[
                        ["order_id"],
                        ["customer_id"],
                        ["order_ts"],
                        ["order_status"],
                        ["gross_amount_usd"],
                        ["currency_code"],
                        ["ingest_source: csv_archive"],
                        ["ingest_batch_id: current_batch_id"],
                    ],
                    forbidden_terms=["defaults: {}"],
                ),
                "archive_defaults_guard": repair_validator(
                    "Checks that archive-only metadata fields are populated with stable defaults.",
                    "pipelines/orders_archive_load.yaml",
                    required_groups=[
                        ["created_at: current_timestamp"],
                        ["updated_at: current_timestamp"],
                        ["archive_status: active"],
                        ["notes: ''", "notes: \"\""],
                    ],
                ),
            },
        ),
        repair_scenario(
            scenario_id="PR-008",
            developer_request=(
                "The `dim_customer_merge` load started failing with a primary key "
                "violation after duplicate rows appeared in the incoming batch. "
                "Repair the merge strategy so the load is safe again."
            ),
            workspace_summary=(
                "Dimension merges are implemented in SQL and should deduplicate the "
                "incoming batch before writing to the warehouse table. Validators "
                "`pk_uniqueness_guard` and `merge_strategy_guard` are available."
            ),
            known_files=[
                "transforms/dim_customer_merge.sql",
                "samples/dim_customer_delta.csv",
                "schemas/dim_customer.json",
                "docs/merge_strategy.md",
            ],
            editable_targets=["transforms/dim_customer_merge.sql"],
            known_assets=["dim_customer_delta", "dim_customer"],
            available_validators=["pk_uniqueness_guard", "merge_strategy_guard"],
            files={
                "transforms/dim_customer_merge.sql": """insert into dim_customer (
    customer_id,
    customer_name,
    state_code,
    updated_at
)
select
    d.customer_id,
    d.customer_name,
    d.state_code,
    d.updated_at
from dim_customer_delta d;
""",
                "samples/dim_customer_delta.csv": """customer_id,customer_name,state_code,updated_at
c-1,Ada Smith,CA,2026-04-01T10:00:00Z
c-1,Ada Smith,CA,2026-04-01T10:05:00Z
""",
                "schemas/dim_customer.json": """{
  "name": "dim_customer",
  "primary_key": ["customer_id"],
  "columns": [
    {"name": "customer_id", "type": "string"},
    {"name": "customer_name", "type": "string"},
    {"name": "state_code", "type": "string"},
    {"name": "updated_at", "type": "timestamp"}
  ]
}
""",
                "docs/merge_strategy.md": """# Merge strategy

- Incoming dimension batches may contain duplicate customer_id rows.
- Deduplicate the stage with `row_number()` and keep the latest `updated_at`.
- Use `merge` or `insert ... where not exists` semantics; never blindly insert the full batch.
""",
            },
            schema_registry={
                "dim_customer_delta": """asset: dim_customer_delta
possible_duplicates:
  primary_key: customer_id
""",
                "dim_customer": """asset: dim_customer
primary_key:
  - customer_id
""",
            },
            repair_target={
                "path": "transforms/dim_customer_merge.sql",
                "required_groups": [
                    ["row_number()"],
                    ["partition by customer_id"],
                    ["order by d.updated_at desc", "order by updated_at desc"],
                    ["where rn = 1", "where deduped.rn = 1"],
                ],
                "forbidden_terms": [],
                "root_cause_groups": [
                    ["primary key", "pk"],
                    ["duplicate", "duplicates"],
                    ["deduplicate", "row_number", "merge"],
                ],
                "summary_groups": [["primary key"], ["deduplicate", "latest"], ["merge", "load"]],
                "fix_path": "transforms/dim_customer_merge.sql",
                "investigation_targets": [
                    ("read_file", "transforms/dim_customer_merge.sql"),
                    ("read_file", "samples/dim_customer_delta.csv"),
                    ("read_file", "schemas/dim_customer.json"),
                    ("read_file", "docs/merge_strategy.md"),
                ],
            },
            validators={
                "pk_uniqueness_guard": repair_validator(
                    "Checks that the merge SQL deduplicates customer_id rows before writing.",
                    "transforms/dim_customer_merge.sql",
                    required_groups=[
                        ["row_number()"],
                        ["partition by customer_id"],
                        ["where rn = 1", "where deduped.rn = 1"],
                    ],
                ),
                "merge_strategy_guard": repair_validator(
                    "Checks that the merge keeps the latest updated_at row instead of blindly inserting duplicates.",
                    "transforms/dim_customer_merge.sql",
                    required_groups=[
                        ["order by d.updated_at desc", "order by updated_at desc"],
                        ["insert into dim_customer", "merge into dim_customer"],
                    ],
                ),
            },
        ),
    ],
    "llm_patch_review": [
        review_scenario(
            scenario_id="LR-001",
            developer_request=(
                "Review the AI-generated patch for `payments_enriched` and decide if "
                "it is safe to merge into production analytics."
            ),
            workspace_summary=(
                "Company policy forbids exposing direct identifiers in analytics "
                "outputs. The patch file in `drafts/` contains the assistant's "
                "proposed change."
            ),
            known_files=[
                "transforms/payments_enriched.sql",
                "schemas/payments_enriched.json",
                "policies/data_governance.md",
                "drafts/payments_enriched.patch",
            ],
            known_assets=["payments_raw", "customer_dim", "payments_enriched"],
            available_validators=["privacy_guard", "contract_guard"],
            files={
                "transforms/payments_enriched.sql": """select
    p.payment_id,
    p.customer_id,
    p.amount_usd,
    p.payment_status,
    p.processed_at
from payments_raw p;
""",
                "schemas/payments_enriched.json": """{
  "name": "payments_enriched",
  "columns": [
    {"name": "payment_id", "type": "string"},
    {"name": "customer_id", "type": "string"},
    {"name": "amount_usd", "type": "float"},
    {"name": "payment_status", "type": "string"},
    {"name": "processed_at", "type": "timestamp"}
  ]
}
""",
                "policies/data_governance.md": """# Data governance

- Analytics outputs must not expose `customer_email`, `phone`, or `full_name`.
- Direct identifiers are only allowed in operational systems, not shared marts.
- Schema changes to published datasets require explicit approval.
""",
                "drafts/payments_enriched.patch": """diff --git a/transforms/payments_enriched.sql b/transforms/payments_enriched.sql
@@
 select
     p.payment_id,
     p.customer_id,
+    c.customer_email,
     p.amount_usd,
     p.payment_status,
     p.processed_at
 from payments_raw p
+left join customer_dim c on p.customer_id = c.customer_id;

diff --git a/schemas/payments_enriched.json b/schemas/payments_enriched.json
@@
   {"name": "customer_id", "type": "string"},
+  {"name": "customer_email", "type": "string"},
   {"name": "amount_usd", "type": "float"},
""",
            },
            schema_registry={
                "payments_raw": """asset: payments_raw
columns:
  - payment_id: string
  - customer_id: string
  - amount_usd: float
  - payment_status: string
  - processed_at: timestamp
""",
                "customer_dim": """asset: customer_dim
columns:
  - customer_id: string
  - customer_email: string
  - phone: string
""",
                "payments_enriched": """asset: payments_enriched
published_columns:
  - payment_id
  - customer_id
  - amount_usd
  - payment_status
  - processed_at
""",
            },
            llm_draft={
                "primary": """AI draft summary:
- Adds `customer_email` to the SELECT list.
- Joins `customer_dim`.
- Extends the published schema with `customer_email`.
""",
            },
            review_target={
                "correct_verdict": "reject",
                "correct_issue_type": "pii_exposure",
                "summary_groups": [["customer_email"], ["policy", "direct identifier"], ["schema", "contract"]],
                "investigation_targets": [
                    ("inspect_llm_draft", "primary"),
                    ("read_file", "policies/data_governance.md"),
                    ("run_validator", "privacy_guard"),
                ],
                "validator_targets": ["privacy_guard"],
            },
            validators={
                "privacy_guard": review_validator(
                    "Checks whether the proposed patch leaks direct identifiers.",
                    "primary",
                    forbidden_terms=["customer_email", "phone", "full_name"],
                ),
                "contract_guard": review_validator(
                    "Checks whether the patch changes the published contract.",
                    "primary",
                    forbidden_terms=["published schema", "customer_email"],
                ),
            },
        ),
        review_scenario(
            scenario_id="LR-002",
            developer_request=(
                "Review the assistant patch for `merchant_settlement_rollup`. The SQL "
                "looks plausible, but finance thinks the join is wrong."
            ),
            workspace_summary=(
                "The AI patch sits in `drafts/` and the correct join key is documented "
                "in the settlement conventions file."
            ),
            known_files=[
                "transforms/merchant_settlement_rollup.sql",
                "docs/settlement_conventions.md",
                "drafts/merchant_settlement_rollup.patch",
            ],
            known_assets=["settlements_raw", "merchant_dim", "merchant_settlement_rollup"],
            available_validators=["logic_guard", "join_key_guard"],
            files={
                "transforms/merchant_settlement_rollup.sql": """select
    s.settlement_id,
    s.merchant_id,
    s.net_amount_usd,
    s.settlement_date
from settlements_raw s;
""",
                "docs/settlement_conventions.md": """# Settlement conventions

- Merchant enrichment joins on `merchant_id`.
- Never join merchant dimensions using settlement or payment identifiers.
- Published rollups must stay at one row per settlement_id.
""",
                "drafts/merchant_settlement_rollup.patch": """diff --git a/transforms/merchant_settlement_rollup.sql b/transforms/merchant_settlement_rollup.sql
@@
 select
     s.settlement_id,
     s.merchant_id,
     m.region as merchant_region,
     s.net_amount_usd,
     s.settlement_date
 from settlements_raw s
+left join merchant_dim m on s.settlement_id = m.merchant_id;
""",
            },
            schema_registry={
                "settlements_raw": """asset: settlements_raw
columns:
  - settlement_id: string
  - merchant_id: string
  - net_amount_usd: float
  - settlement_date: date
""",
                "merchant_dim": """asset: merchant_dim
columns:
  - merchant_id: string
  - region: string
""",
                "merchant_settlement_rollup": """asset: merchant_settlement_rollup
grain:
  one_row_per: settlement_id
""",
            },
            llm_draft={
                "primary": """AI draft summary:
- Enriches settlements with merchant region.
- Joins `merchant_dim` using `settlement_id = merchant_id`.
- Does not change the published grain intentionally.
""",
            },
            review_target={
                "correct_verdict": "reject",
                "correct_issue_type": "technical_incorrectness",
                "summary_groups": [
                    ["join"],
                    ["settlement_id", "merchant_id"],
                    ["wrong", "incorrect"],
                    ["grain", "one row per settlement_id"],
                ],
                "investigation_targets": [
                    ("inspect_llm_draft", "primary"),
                    ("read_file", "drafts/merchant_settlement_rollup.patch"),
                    ("read_file", "docs/settlement_conventions.md"),
                    ("run_validator", "logic_guard"),
                ],
                "validator_targets": ["logic_guard", "join_key_guard"],
            },
            validators={
                "logic_guard": review_validator(
                    "Checks that the draft avoids the incorrect settlement-to-merchant join.",
                    "primary",
                    forbidden_terms=["settlement_id = merchant_id"],
                ),
                "join_key_guard": review_validator(
                    "Checks that the draft references the documented merchant join key.",
                    "primary",
                    required_groups=[["merchant_id"]],
                    forbidden_terms=["settlement_id = merchant_id"],
                ),
            },
        ),
        review_scenario(
            scenario_id="LR-003",
            developer_request=(
                "Review the patch for `subscription_mrr_daily`. The assistant claims it "
                "fixes a reporting bug, but product analytics is worried about contract "
                "compatibility."
            ),
            workspace_summary=(
                "Published contracts for finance datasets are strict. Review the patch "
                "and decide if the proposed schema change is acceptable."
            ),
            known_files=[
                "schemas/subscription_mrr_daily.json",
                "policies/published_contracts.md",
                "drafts/subscription_mrr_daily.patch",
            ],
            known_assets=["subscription_mrr_daily"],
            available_validators=["contract_guard", "consumer_guard"],
            files={
                "schemas/subscription_mrr_daily.json": """{
  "name": "subscription_mrr_daily",
  "columns": [
    {"name": "account_id", "type": "string"},
    {"name": "snapshot_date", "type": "date"},
    {"name": "mrr_usd", "type": "float"}
  ]
}
""",
                "policies/published_contracts.md": """# Published contracts

- Finance marts require approval before adding or renaming published columns.
- Backward-incompatible contract changes must not be merged through assistant patches.
""",
                "drafts/subscription_mrr_daily.patch": """diff --git a/schemas/subscription_mrr_daily.json b/schemas/subscription_mrr_daily.json
@@
   {"name": "account_id", "type": "string"},
   {"name": "snapshot_date", "type": "date"},
-  {"name": "mrr_usd", "type": "float"}
+  {"name": "monthly_recurring_revenue_usd", "type": "float"}
""",
            },
            schema_registry={
                "subscription_mrr_daily": """asset: subscription_mrr_daily
published_columns:
  - account_id
  - snapshot_date
  - mrr_usd
""",
            },
            llm_draft={
                "primary": """AI draft summary:
- Renames `mrr_usd` to `monthly_recurring_revenue_usd` for clarity.
- Describes the change as non-breaking because the meaning is unchanged.
""",
            },
            review_target={
                "correct_verdict": "reject",
                "correct_issue_type": "contract_breakage",
                "summary_groups": [["published", "contract"], ["mrr_usd"], ["rename", "breaking"], ["consumer", "downstream"]],
                "investigation_targets": [
                    ("inspect_llm_draft", "primary"),
                    ("read_file", "policies/published_contracts.md"),
                    ("run_validator", "contract_guard"),
                ],
                "validator_targets": ["contract_guard"],
            },
            validators={
                "contract_guard": review_validator(
                    "Checks that the draft keeps the published finance contract stable.",
                    "primary",
                    forbidden_terms=["monthly_recurring_revenue_usd", "rename"],
                ),
                "consumer_guard": review_validator(
                    "Checks that downstream consumers still receive `mrr_usd`.",
                    "primary",
                    forbidden_terms=["monthly_recurring_revenue_usd"],
                ),
            },
        ),
        review_scenario(
            scenario_id="LR-004",
            developer_request=(
                "Review the assistant response for warehouse recovery. Decide whether "
                "the recommendation is safe to execute in production."
            ),
            workspace_summary=(
                "The assistant produced an ops recommendation instead of a code patch. "
                "Use the platform recovery policy as the source of truth."
            ),
            known_files=[
                "policies/recovery_playbook.md",
                "drafts/warehouse_recovery_response.md",
            ],
            known_assets=["warehouse.recovery"],
            available_validators=["safety_guard", "playbook_guard"],
            files={
                "policies/recovery_playbook.md": """# Recovery playbook

- Never drop production marts as a first response to pipeline incidents.
- Do not disable contract or governance checks to speed up recovery.
- Preferred order: identify failing upstream, patch safely, re-run validators, then backfill.
""",
                "drafts/warehouse_recovery_response.md": """The quickest fix is to drop and recreate `finance.daily_margin_mart`,
disable contract checks temporarily, and backfill after the model retrains.
This avoids waiting on upstream teams and gets dashboards green faster.
""",
            },
            schema_registry={
                "warehouse.recovery": """asset: warehouse.recovery
policy_owner: platform-sre
""",
            },
            llm_draft={
                "primary": """AI response summary:
- Recommends dropping a production mart.
- Suggests disabling contract checks temporarily.
- Prioritizes speed over safe rollback and validation.
""",
            },
            review_target={
                "correct_verdict": "reject",
                "correct_issue_type": "unsafe_operation",
                "summary_groups": [["drop", "recreate"], ["disable", "checks"], ["unsafe", "policy"]],
                "investigation_targets": [
                    ("inspect_llm_draft", "primary"),
                    ("read_file", "drafts/warehouse_recovery_response.md"),
                    ("read_file", "policies/recovery_playbook.md"),
                    ("run_validator", "safety_guard"),
                ],
                "validator_targets": ["safety_guard", "playbook_guard"],
            },
            validators={
                "safety_guard": review_validator(
                    "Checks that the recommendation avoids destructive recovery advice.",
                    "primary",
                    forbidden_terms=["drop", "disable contract checks"],
                ),
                "playbook_guard": review_validator(
                    "Checks alignment with the platform recovery playbook.",
                    "primary",
                    forbidden_terms=["drop", "disable"],
                ),
            },
        ),
    ],
    "workflow_shipping": [
        workflow_scenario(
            scenario_id="WS-001",
            developer_request=(
                "Add a daily `merchant_risk_features` workflow for the risk-model "
                "team. Use `transactions_silver`, `chargebacks_silver`, and "
                "`merchant_dim`; compute a 30-day lookback; exclude test merchants; "
                "do not expose PII; add freshness and not-null checks; and preserve "
                "the downstream `risk_model_scoring` contract."
            ),
            workspace_summary=(
                "This repo contains examples for feature pipelines plus a draft from "
                "an internal coding assistant. The target workflow does not exist yet. "
                "Use the templates and downstream contract as the source of truth."
            ),
            known_files=[
                "templates/pipeline_template.yaml",
                "templates/feature_template.sql",
                "templates/checks_template.yaml",
                "pipelines/order_features.yaml",
                "sql/order_features.sql",
                "checks/order_features.yaml",
                "schemas/order_features.json",
                "pipelines/risk_model_scoring.yaml",
                "schemas/risk_model_scoring.json",
                "docs/platform_standards.md",
                "drafts/merchant_risk_features.stub",
            ],
            editable_targets=[
                "pipelines/merchant_risk_features.yaml",
                "sql/merchant_risk_features.sql",
                "checks/merchant_risk_features.yaml",
                "schemas/merchant_risk_features.json",
            ],
            known_assets=[
                "transactions_silver",
                "chargebacks_silver",
                "merchant_dim",
                "merchant_risk_features",
                "risk_model_scoring",
            ],
            available_validators=[
                "workflow_lint",
                "quality_checks",
                "contract_check",
                "governance_check",
            ],
            files={
                "templates/pipeline_template.yaml": """name: <pipeline_name>
schedule: "0 1 * * *"
owner: risk-platform
sources:
  - <source_asset>
target: <target_asset>
checks_file: checks/<pipeline_name>.yaml
downstreams:
  - <consumer>
""",
                "templates/feature_template.sql": """with source as (
    select *
    from <source_asset>
)
select *
from source;
""",
                "templates/checks_template.yaml": """checks:
  - type: freshness
    column: snapshot_date
  - type: not_null
    column: id
""",
                "pipelines/order_features.yaml": """name: order_features
schedule: "0 1 * * *"
owner: risk-platform
sources:
  - orders_silver
  - merchant_dim
target: order_features
checks_file: checks/order_features.yaml
downstreams:
  - risk_model_scoring
""",
                "sql/order_features.sql": """with recent_orders as (
    select
        o.merchant_id,
        count(*) as order_count_30d,
        current_date as snapshot_date
    from orders_silver o
    join merchant_dim m on o.merchant_id = m.merchant_id
    where m.is_test = false
      and o.created_at >= current_date - interval '30 day'
    group by 1, 3
)
select *
from recent_orders;
""",
                "checks/order_features.yaml": """checks:
  - type: freshness
    column: snapshot_date
  - type: not_null
    column: merchant_id
""",
                "schemas/order_features.json": """{
  "name": "order_features",
  "columns": [
    {"name": "merchant_id", "type": "string"},
    {"name": "order_count_30d", "type": "integer"},
    {"name": "snapshot_date", "type": "date"}
  ]
}
""",
                "pipelines/risk_model_scoring.yaml": """name: risk_model_scoring
sources:
  - merchant_risk_features
target: risk_model_scoring
""",
                "schemas/risk_model_scoring.json": """{
  "name": "risk_model_scoring",
  "required_inputs": [
    "merchant_id",
    "txn_count_30d",
    "chargeback_rate_30d",
    "snapshot_date"
  ]
}
""",
                "docs/platform_standards.md": """# Platform standards

- New feature pipelines run daily at `0 1 * * *`.
- Exclude test merchants using `merchant_dim.is_test = false`.
- Never select direct identifiers such as `email`, `phone`, or `full_name`.
- Every feature pipeline must ship freshness and not_null checks.
- Declare downstream consumers in the pipeline YAML.
""",
                "drafts/merchant_risk_features.stub": """# AI assistant stub
-- WARNING: draft may be unsafe
select
    t.merchant_id,
    m.email as merchant_email,
    count(*) as txn_count_30d,
    current_date as snapshot_date
from transactions_silver t
join merchant_dim m on t.merchant_id = m.merchant_id
group by 1, 2, 4;
""",
            },
            schema_registry={
                "transactions_silver": """asset: transactions_silver
columns:
  - merchant_id: string
  - transaction_id: string
  - amount_usd: float
  - processed_at: timestamp
""",
                "chargebacks_silver": """asset: chargebacks_silver
columns:
  - merchant_id: string
  - chargeback_id: string
  - processed_at: timestamp
""",
                "merchant_dim": """asset: merchant_dim
columns:
  - merchant_id: string
  - is_test: boolean
  - email: string
  - owner_name: string
""",
                "merchant_risk_features": """asset: merchant_risk_features
required_columns:
  - merchant_id
  - txn_count_30d
  - chargeback_rate_30d
  - snapshot_date
""",
                "risk_model_scoring": """asset: risk_model_scoring
requires:
  - merchant_id
  - txn_count_30d
  - chargeback_rate_30d
  - snapshot_date
""",
            },
            workflow_target={
                "required_artifacts": ["pipeline", "sql", "checks", "schema"],
                "artifacts": {
                    "pipeline": workflow_artifact(
                        "pipelines/merchant_risk_features.yaml",
                        [
                            ["merchant_risk_features"],
                            ["0 1 * * *", "daily"],
                            ["transactions_silver"],
                            ["chargebacks_silver"],
                            ["merchant_dim"],
                            ["risk_model_scoring"],
                        ],
                        0.25,
                    ),
                    "sql": workflow_artifact(
                        "sql/merchant_risk_features.sql",
                        [
                            ["transactions_silver"],
                            ["chargebacks_silver"],
                            ["merchant_dim"],
                            ["30 day", "30-day", "interval '30 day'"],
                            ["is_test = false", "is_test=false"],
                            ["txn_count_30d"],
                            ["chargeback_rate_30d"],
                            ["snapshot_date"],
                        ],
                        0.4,
                        forbidden_terms=["email", "phone", "full_name"],
                    ),
                    "checks": workflow_artifact(
                        "checks/merchant_risk_features.yaml",
                        [["freshness"], ["not_null"], ["merchant_id"], ["snapshot_date"]],
                        0.15,
                    ),
                    "schema": workflow_artifact(
                        "schemas/merchant_risk_features.json",
                        [["merchant_id"], ["txn_count_30d"], ["chargeback_rate_30d"], ["snapshot_date"]],
                        0.2,
                        forbidden_terms=["email", "phone", "full_name"],
                    ),
                },
                "investigation_targets": [
                    ("read_file", "templates/pipeline_template.yaml"),
                    ("read_file", "sql/order_features.sql"),
                    ("inspect_schema", "transactions_silver"),
                    ("inspect_lineage", "risk_model_scoring"),
                ],
                "summary_groups": [["merchant_risk_features"], ["risk_model_scoring"], ["checks", "validators"]],
                "solve_validator_targets": ["workflow_lint", "governance_check"],
                "validator_targets": [
                    "workflow_lint",
                    "quality_checks",
                    "contract_check",
                    "governance_check",
                    "workflow_sources_check",
                    "workflow_schema_shape_check",
                    "workflow_checks_shape_check",
                    "workflow_delivery_check",
                ],
                "forbidden_terms": ["email", "phone", "full_name"],
            },
            validators={
                "workflow_lint": workflow_validator(
                    "Checks pipeline metadata and lineage wiring.",
                    "pipeline",
                ),
                "quality_checks": workflow_validator(
                    "Checks freshness and not-null coverage.",
                    "checks",
                ),
                "contract_check": workflow_validator(
                    "Checks the schema contract expected by downstream scoring.",
                    "schema",
                ),
                "governance_check": workflow_validator(
                    "Checks for PII leakage and missing test-merchant filters.",
                    "sql",
                ),
                "workflow_sources_check": workflow_validator(
                    "Checks that the workflow covers the intended upstream source set.",
                    "pipeline",
                ),
                "workflow_schema_shape_check": workflow_validator(
                    "Checks that the feature schema stays aligned with downstream scoring expectations.",
                    "schema",
                ),
                "workflow_checks_shape_check": workflow_validator(
                    "Checks that the workflow publishes the expected validation checks.",
                    "checks",
                ),
                "workflow_delivery_check": workflow_validator(
                    "Checks that the workflow artifact bundle is ready for downstream delivery.",
                    "pipeline",
                ),
            },
            lineage={
                "risk_model_scoring": """risk_model_scoring
  <- merchant_risk_features
     <- transactions_silver
     <- chargebacks_silver
     <- merchant_dim
""",
            },
            llm_draft={
                "primary": """AI draft summary:
- Mentions `merchant_email`, which violates governance.
- Omits chargeback rate.
- Does not exclude test merchants.
- Does not define checks or pipeline metadata.
""",
            },
        ),
        workflow_scenario(
            scenario_id="WS-002",
            developer_request=(
                "Create a reporting view called `finance_weekly_net_revenue` for the "
                "finance analysts. Base it on `invoice_line_items` and "
                "`credit_memos_daily`, publish a weekly grain by account, and define "
                "the view contract. This task only needs the SQL and schema artifacts."
            ),
            workspace_summary=(
                "Reporting views live under `sql/views/` and publish contracts under "
                "`schemas/views/`. There is no pipeline YAML for analyst-facing views."
            ),
            known_files=[
                "sql/views/finance_daily_revenue.sql",
                "schemas/views/finance_daily_revenue.json",
                "docs/reporting_view_standards.md",
                "docs/finance_reporting_glossary.md",
            ],
            editable_targets=[
                "sql/views/finance_weekly_net_revenue.sql",
                "schemas/views/finance_weekly_net_revenue.json",
            ],
            known_assets=[
                "invoice_line_items",
                "credit_memos_daily",
                "finance_weekly_net_revenue",
            ],
            available_validators=[
                "view_sql_check",
                "view_contract_check",
                "view_usage_check",
                "view_reference_check",
                "view_schema_naming_check",
                "view_rollup_check",
                "view_source_shape_check",
                "view_semantics_check",
                "view_contract_shape_check",
            ],
            files={
                "sql/views/finance_daily_revenue.sql": """select
    i.account_id,
    date(i.invoice_ts) as revenue_date,
    sum(i.net_revenue_usd) as net_revenue_usd
from invoice_line_items i
group by 1, 2;
""",
                "schemas/views/finance_daily_revenue.json": """{
  "name": "finance_daily_revenue",
  "columns": [
    {"name": "account_id", "type": "string"},
    {"name": "revenue_date", "type": "date"},
    {"name": "net_revenue_usd", "type": "float"}
  ]
}
""",
                "docs/reporting_view_standards.md": """# Reporting view standards

- Analyst-facing views do not need pipeline YAML.
- Weekly finance views should group by `date_trunc('week', ...)`.
- Publish one schema contract per view.
- Keep direct identifiers out of finance reporting views.
""",
                "docs/finance_reporting_glossary.md": """# Finance reporting glossary

- Weekly reporting views should preserve a stable account-level rollup.
- Reporting contracts should stay aligned with finance naming conventions.
""",
            },
            schema_registry={
                "invoice_line_items": """asset: invoice_line_items
columns:
  - account_id: string
  - invoice_ts: timestamp
  - net_revenue_usd: float
""",
                "credit_memos_daily": """asset: credit_memos_daily
columns:
  - account_id: string
  - credit_memo_date: date
  - credit_amount_usd: float
""",
                "finance_weekly_net_revenue": """asset: finance_weekly_net_revenue
required_columns:
  - account_id
  - revenue_week
  - net_revenue_usd
""",
            },
            workflow_target={
                "required_artifacts": ["sql", "schema"],
                "artifacts": {
                    "sql": workflow_artifact(
                        "sql/views/finance_weekly_net_revenue.sql",
                        [
                            ["invoice_line_items"],
                            ["credit_memos_daily"],
                            ["date_trunc('week'", "date_trunc(\"week\""],
                            ["account_id"],
                            ["net_revenue_usd"],
                        ],
                        0.7,
                        forbidden_terms=["email", "phone", "full_name"],
                    ),
                    "schema": workflow_artifact(
                        "schemas/views/finance_weekly_net_revenue.json",
                        [["account_id"], ["revenue_week"], ["net_revenue_usd"]],
                        0.3,
                        forbidden_terms=["email", "phone", "full_name"],
                    ),
                },
                "investigation_targets": [
                    ("read_file", "sql/views/finance_daily_revenue.sql"),
                    ("read_file", "schemas/views/finance_daily_revenue.json"),
                    ("read_file", "docs/reporting_view_standards.md"),
                    ("read_file", "docs/finance_reporting_glossary.md"),
                    ("inspect_schema", "invoice_line_items"),
                    ("inspect_schema", "credit_memos_daily"),
                    ("inspect_schema", "finance_weekly_net_revenue"),
                ],
                "summary_groups": [["finance_weekly_net_revenue"], ["weekly"], ["schema"]],
                "solve_validator_targets": ["view_sql_check", "view_contract_check"],
                "validator_targets": [
                    "view_sql_check",
                    "view_contract_check",
                    "view_usage_check",
                    "view_reference_check",
                    "view_schema_naming_check",
                    "view_rollup_check",
                    "view_source_shape_check",
                    "view_semantics_check",
                    "view_contract_shape_check",
                ],
                "forbidden_terms": ["email", "phone", "full_name"],
            },
            validators={
                "view_sql_check": workflow_validator(
                    "Checks that the weekly reporting view SQL joins the expected inputs and weekly grain.",
                    "sql",
                ),
                "view_contract_check": workflow_validator(
                    "Checks that the reporting view contract publishes the required columns.",
                    "schema",
                ),
                "view_usage_check": workflow_validator(
                    "Checks that the reporting SQL publishes a stable account-level view shape.",
                    "sql",
                ),
                "view_reference_check": workflow_validator(
                    "Checks that the weekly view references the expected finance source tables.",
                    "sql",
                ),
                "view_schema_naming_check": workflow_validator(
                    "Checks that the published reporting schema uses the expected view name and column contract.",
                    "schema",
                ),
                "view_rollup_check": workflow_validator(
                    "Checks that the weekly reporting SQL preserves the intended rollup output shape.",
                    "sql",
                ),
                "view_source_shape_check": workflow_validator(
                    "Checks that the weekly reporting view contract stays aligned with the finance source shape.",
                    "schema",
                ),
                "view_semantics_check": workflow_validator(
                    "Checks that the weekly reporting SQL preserves the intended finance semantics.",
                    "sql",
                ),
                "view_contract_shape_check": workflow_validator(
                    "Checks that the weekly reporting contract stays aligned with downstream reporting expectations.",
                    "schema",
                ),
            },
        ),
        workflow_scenario(
            scenario_id="WS-003",
            developer_request=(
                "Create a new `customer_margin_mart` workflow for finance. It should "
                "combine `orders_silver`, `refunds_silver`, and `customer_dim`, ship "
                "daily, exclude internal test accounts, write checks, and publish a "
                "contract for downstream BI dashboards."
            ),
            workspace_summary=(
                "Finance marts follow the same pipeline structure as other curated "
                "warehouse tables. Use the existing mart example and governance rules."
            ),
            known_files=[
                "templates/pipeline_template.yaml",
                "sql/order_features.sql",
                "checks/order_features.yaml",
                "schemas/order_features.json",
                "docs/platform_standards.md",
            ],
            editable_targets=[
                "pipelines/customer_margin_mart.yaml",
                "sql/customer_margin_mart.sql",
                "checks/customer_margin_mart.yaml",
                "schemas/customer_margin_mart.json",
            ],
            known_assets=[
                "orders_silver",
                "refunds_silver",
                "customer_dim",
                "customer_margin_mart",
            ],
            available_validators=[
                "mart_lint",
                "mart_checks",
                "mart_contract",
                "mart_governance",
                "mart_source_check",
                "mart_schema_shape_check",
                "mart_pipeline_metadata_check",
                "mart_publish_contract_check",
                "mart_consumer_shape_check",
            ],
            files={
                "templates/pipeline_template.yaml": """name: <pipeline_name>
schedule: "0 1 * * *"
owner: data-platform
sources:
  - <source_asset>
target: <target_asset>
checks_file: checks/<pipeline_name>.yaml
""",
                "sql/order_features.sql": """with recent_orders as (
    select
        o.merchant_id,
        count(*) as order_count_30d,
        current_date as snapshot_date
    from orders_silver o
    join merchant_dim m on o.merchant_id = m.merchant_id
    where m.is_test = false
      and o.created_at >= current_date - interval '30 day'
    group by 1, 3
)
select *
from recent_orders;
""",
                "checks/order_features.yaml": """checks:
  - type: freshness
    column: snapshot_date
  - type: not_null
    column: merchant_id
""",
                "schemas/order_features.json": """{
  "name": "order_features",
  "columns": [
    {"name": "merchant_id", "type": "string"},
    {"name": "order_count_30d", "type": "integer"},
    {"name": "snapshot_date", "type": "date"}
  ]
}
""",
                "docs/platform_standards.md": """# Platform standards

- Curated marts run daily at `0 1 * * *`.
- Exclude test or internal entities using the dimension table filter.
- Publish freshness and not_null checks for curated marts.
- Avoid direct identifiers such as `email`, `phone`, and `full_name`.
""",
            },
            schema_registry={
                "orders_silver": """asset: orders_silver
columns:
  - customer_id: string
  - order_id: string
  - created_at: timestamp
  - gross_amount_usd: float
""",
                "refunds_silver": """asset: refunds_silver
columns:
  - customer_id: string
  - refund_id: string
  - processed_at: timestamp
  - refund_amount_usd: float
""",
                "customer_dim": """asset: customer_dim
columns:
  - customer_id: string
  - is_test: boolean
  - email: string
""",
                "customer_margin_mart": """asset: customer_margin_mart
required_columns:
  - customer_id
  - gross_revenue_usd
  - refund_amount_usd
  - net_margin_usd
  - snapshot_date
""",
            },
            workflow_target={
                "required_artifacts": ["pipeline", "sql", "checks", "schema"],
                "artifacts": {
                    "pipeline": workflow_artifact(
                        "pipelines/customer_margin_mart.yaml",
                        [
                            ["customer_margin_mart"],
                            ["0 1 * * *", "daily"],
                            ["orders_silver"],
                            ["refunds_silver"],
                            ["customer_dim"],
                        ],
                        0.2,
                    ),
                    "sql": workflow_artifact(
                        "sql/customer_margin_mart.sql",
                        [
                            ["orders_silver"],
                            ["refunds_silver"],
                            ["customer_dim"],
                            ["is_test = false", "is_test=false"],
                            ["gross_revenue_usd"],
                            ["refund_amount_usd"],
                            ["net_margin_usd"],
                            ["snapshot_date"],
                        ],
                        0.45,
                        forbidden_terms=["email", "phone", "full_name"],
                    ),
                    "checks": workflow_artifact(
                        "checks/customer_margin_mart.yaml",
                        [["freshness"], ["not_null"], ["customer_id"], ["snapshot_date"]],
                        0.15,
                    ),
                    "schema": workflow_artifact(
                        "schemas/customer_margin_mart.json",
                        [
                            ["customer_id"],
                            ["gross_revenue_usd"],
                            ["refund_amount_usd"],
                            ["net_margin_usd"],
                            ["snapshot_date"],
                        ],
                        0.2,
                        forbidden_terms=["email", "phone", "full_name"],
                    ),
                },
                "investigation_targets": [
                    ("read_file", "templates/pipeline_template.yaml"),
                    ("read_file", "checks/order_features.yaml"),
                    ("read_file", "docs/platform_standards.md"),
                    ("read_file", "schemas/order_features.json"),
                    ("inspect_schema", "orders_silver"),
                    ("inspect_schema", "refunds_silver"),
                    ("inspect_schema", "customer_dim"),
                ],
                "summary_groups": [["customer_margin_mart"], ["daily"], ["checks", "contract"]],
                "solve_validator_targets": ["mart_lint", "mart_governance"],
                "validator_targets": [
                    "mart_lint",
                    "mart_checks",
                    "mart_contract",
                    "mart_governance",
                    "mart_source_check",
                    "mart_schema_shape_check",
                    "mart_pipeline_metadata_check",
                    "mart_publish_contract_check",
                    "mart_consumer_shape_check",
                    "mart_schedule_check",
                    "mart_delivery_shape_check",
                ],
                "forbidden_terms": ["email", "phone", "full_name"],
            },
            validators={
                "mart_lint": workflow_validator(
                    "Checks mart pipeline metadata and sources.",
                    "pipeline",
                ),
                "mart_checks": workflow_validator(
                    "Checks mart freshness and not-null coverage.",
                    "checks",
                ),
                "mart_contract": workflow_validator(
                    "Checks mart contract columns.",
                    "schema",
                ),
                "mart_governance": workflow_validator(
                    "Checks that mart SQL excludes direct identifiers and test accounts.",
                    "sql",
                ),
                "mart_source_check": workflow_validator(
                    "Checks that the mart pipeline and SQL cover the required upstream sources.",
                    "pipeline",
                ),
                "mart_schema_shape_check": workflow_validator(
                    "Checks that the mart schema publishes the expected finance BI columns.",
                    "schema",
                ),
                "mart_pipeline_metadata_check": workflow_validator(
                    "Checks that the mart pipeline metadata is production-ready for finance scheduling.",
                    "pipeline",
                ),
                "mart_publish_contract_check": workflow_validator(
                    "Checks that the mart contract remains publishable for downstream BI consumers.",
                    "schema",
                ),
                "mart_consumer_shape_check": workflow_validator(
                    "Checks that the mart output shape is suitable for downstream dashboard consumption.",
                    "schema",
                ),
                "mart_schedule_check": workflow_validator(
                    "Checks that the mart pipeline metadata stays aligned with the daily finance schedule.",
                    "pipeline",
                ),
                "mart_delivery_shape_check": workflow_validator(
                    "Checks that the mart contract shape is ready for downstream finance delivery.",
                    "schema",
                ),
            },
        ),
        workflow_scenario(
            scenario_id="WS-004",
            developer_request=(
                "Create a new `finance_margin_watch` reporting asset for the control "
                "room. It should publish a daily SQL view, a schema contract, and an "
                "alert config that fires when margin drops sharply."
            ),
            workspace_summary=(
                "Control-room reporting assets pair warehouse views with lightweight "
                "alert configs so finance on-call can react to anomalies."
            ),
            known_files=[
                "sql/views/finance_daily_revenue.sql",
                "docs/reporting_view_standards.md",
                "docs/alerting_playbook.md",
                "alerts/examples/revenue_watch.yaml",
            ],
            editable_targets=[
                "sql/views/finance_margin_watch.sql",
                "schemas/views/finance_margin_watch.json",
                "alerts/finance_margin_watch.yaml",
            ],
            known_assets=[
                "invoice_line_items",
                "credit_memos_daily",
                "finance_margin_watch",
            ],
            available_validators=[
                "view_sql_check",
                "view_contract_check",
                "alert_config_check",
                "alert_severity_check",
                "view_monitoring_shape_check",
                "alert_dataset_check",
            ],
            files={
                "sql/views/finance_daily_revenue.sql": """select
    account_id,
    date(invoice_ts) as revenue_date,
    sum(net_revenue_usd) as net_revenue_usd
from invoice_line_items
group by 1, 2;
""",
                "docs/reporting_view_standards.md": """# Reporting view standards

- Reporting views must publish stable business columns.
- Views for control-room monitoring should expose a single daily grain.
- Schema contracts should list the published fields in order.
""",
                "docs/alerting_playbook.md": """# Alerting playbook

- Alert configs declare `name`, `dataset`, `metric`, `threshold`, and `severity`.
- Margin watch alerts should trigger when `margin_delta_pct` falls below `-0.15`.
- Finance alerts use severity `high`.
""",
                "alerts/examples/revenue_watch.yaml": """name: finance_revenue_watch
dataset: finance_daily_revenue
metric: net_revenue_usd
threshold: -0.1
severity: medium
""",
            },
            schema_registry={
                "invoice_line_items": """asset: invoice_line_items
columns:
  - account_id: string
  - invoice_ts: timestamp
  - net_revenue_usd: float
  - margin_usd: float
""",
                "credit_memos_daily": """asset: credit_memos_daily
columns:
  - account_id: string
  - credit_memo_date: date
  - credit_amount_usd: float
""",
                "finance_margin_watch": """asset: finance_margin_watch
required_columns:
  - account_id
  - margin_date
  - margin_usd
  - margin_delta_pct
""",
            },
            workflow_target={
                "required_artifacts": ["sql", "schema", "alert_config"],
                "artifacts": {
                    "sql": workflow_artifact(
                        "sql/views/finance_margin_watch.sql",
                        [
                            ["invoice_line_items"],
                            ["credit_memos_daily"],
                            ["margin_date"],
                            ["margin_usd"],
                            ["margin_delta_pct"],
                        ],
                        0.45,
                        forbidden_terms=["email", "phone", "full_name"],
                    ),
                    "schema": workflow_artifact(
                        "schemas/views/finance_margin_watch.json",
                        [["account_id"], ["margin_date"], ["margin_usd"], ["margin_delta_pct"]],
                        0.2,
                        forbidden_terms=["email", "phone", "full_name"],
                    ),
                    "alert_config": workflow_artifact(
                        "alerts/finance_margin_watch.yaml",
                        [
                            ["finance_margin_watch"],
                            ["margin_delta_pct"],
                            ["-0.15"],
                            ["high"],
                        ],
                        0.35,
                    ),
                },
                "investigation_targets": [
                    ("read_file", "sql/views/finance_daily_revenue.sql"),
                    ("read_file", "docs/reporting_view_standards.md"),
                    ("read_file", "docs/alerting_playbook.md"),
                    ("read_file", "alerts/examples/revenue_watch.yaml"),
                    ("inspect_schema", "invoice_line_items"),
                ],
                "summary_groups": [["finance_margin_watch"], ["alert"], ["margin_delta_pct"]],
                "solve_validator_targets": ["view_sql_check", "view_contract_check"],
                "validator_targets": [
                    "view_sql_check",
                    "view_contract_check",
                    "alert_config_check",
                    "alert_severity_check",
                    "view_monitoring_shape_check",
                    "alert_dataset_check",
                    "alert_threshold_shape_check",
                    "view_alert_contract_check",
                ],
                "forbidden_terms": ["email", "phone", "full_name"],
            },
            validators={
                "view_sql_check": workflow_validator(
                    "Checks that the margin watch SQL publishes the expected daily fields.",
                    "sql",
                ),
                "view_contract_check": workflow_validator(
                    "Checks that the margin watch contract publishes the expected columns.",
                    "schema",
                ),
                "alert_config_check": workflow_validator(
                    "Checks that the alert config targets the correct dataset, metric, and threshold.",
                    "alert_config",
                ),
                "alert_severity_check": workflow_validator(
                    "Checks that the alert config preserves the intended severity and threshold semantics.",
                    "alert_config",
                ),
                "view_monitoring_shape_check": workflow_validator(
                    "Checks that the monitoring schema exposes the expected daily margin-watch columns.",
                    "schema",
                ),
                "alert_dataset_check": workflow_validator(
                    "Checks that the alert config stays aligned with the intended monitoring dataset shape.",
                    "alert_config",
                ),
                "alert_threshold_shape_check": workflow_validator(
                    "Checks that the alert threshold configuration preserves the intended monitoring shape.",
                    "alert_config",
                ),
                "view_alert_contract_check": workflow_validator(
                    "Checks that the reporting schema and alert contract remain aligned for monitoring delivery.",
                    "schema",
                ),
            },
        ),
    ],
    "simulation_workflow": [
        simulation_scenario(
            scenario_id="SIM-E001",
            developer_request=(
                "Create the simplest possible daily signup report SQL. "
                "Read the contract and source schema, then write only the final report view SQL."
            ),
            workspace_summary=(
                "This is the easiest curriculum workspace. The pipeline YAML, load SQL, and build SQL are already correct. "
                "Only the final report view SQL is missing."
            ),
            source_csv_path="data/daily_signups.csv",
            source_csv_content="""signup_id,signup_date,channel
S001,2026-04-01,organic
S002,2026-04-01,paid
S003,2026-04-02,organic
S004,2026-04-02,referral
""",
            raw_table="raw_daily_signups",
            staged_view="staged_daily_signups",
            build_table="daily_signup_counts",
            final_view="daily_signup_report",
            raw_asset="raw.daily_signups_csv",
            final_asset="mart.daily_signup_report",
            date_column="signup_date",
            raw_schema_columns=[
                ("signup_id", "string"),
                ("signup_date", "date"),
                ("channel", "string"),
            ],
            required_output_columns=[
                "signup_date",
                "signup_count",
            ],
            build_metric_groups=[
                ["count(*) as signup_count", "count(*) signup_count"],
            ],
            report_terms=["daily_signup_report", "DuckDB", "signup_count"],
            correct_load_sql="""create or replace view staged_daily_signups as
select
  signup_id,
  cast(signup_date as date) as signup_date,
  channel
from raw_daily_signups;
""",
            correct_build_sql="""create or replace table daily_signup_counts as
select
  signup_date,
  count(*) as signup_count
from staged_daily_signups
group by 1;
""",
            correct_report_sql="""create or replace view daily_signup_report as
select
  signup_date,
  signup_count
from daily_signup_counts;
""",
            starter_files={
                "pipelines/report_job.yaml": """name: daily_signup_report_job
storage_path: mock_s3/staged/daily_signups.csv
raw_table: raw_daily_signups
load_sql: sql/load_raw.sql
build_sql: sql/build_table.sql
report_sql: sql/report_view.sql
final_view: daily_signup_report
""",
                "sql/load_raw.sql": """create or replace view staged_daily_signups as
select
  signup_id,
  cast(signup_date as date) as signup_date,
  channel
from raw_daily_signups;
""",
                "sql/build_table.sql": """create or replace table daily_signup_counts as
select
  signup_date,
  count(*) as signup_count
from staged_daily_signups
group by 1;
""",
            },
            editable_targets_override=["sql/report_view.sql"],
            available_validators_override=[],
            expected_raw_preview=[
                {"signup_id": "S001", "signup_date": "2026-04-01", "channel": "organic"},
                {"signup_id": "S002", "signup_date": "2026-04-01", "channel": "paid"},
            ],
            expected_derived_preview=[
                {"signup_date": "2026-04-01", "signup_count": 2},
            ],
            expected_final_preview=[
                {"signup_date": "2026-04-02", "signup_count": 2},
            ],
        ),
        simulation_scenario(
            scenario_id="SIM-E002",
            developer_request=(
                "Repair this tiny signup report SQL. The final report view is almost correct, but one alias is wrong."
            ),
            workspace_summary=(
                "This is the easiest repair curriculum workspace. The pipeline YAML, load SQL, and build SQL are already correct. "
                "Only the final report view SQL needs a tiny fix."
            ),
            source_csv_path="data/daily_signups.csv",
            source_csv_content="""signup_id,signup_date,channel
S001,2026-04-01,organic
S002,2026-04-01,paid
S003,2026-04-02,organic
S004,2026-04-02,referral
""",
            raw_table="raw_daily_signups",
            staged_view="staged_daily_signups",
            build_table="daily_signup_counts",
            final_view="daily_signup_report",
            raw_asset="raw.daily_signups_csv",
            final_asset="mart.daily_signup_report",
            date_column="signup_date",
            raw_schema_columns=[
                ("signup_id", "string"),
                ("signup_date", "date"),
                ("channel", "string"),
            ],
            required_output_columns=[
                "signup_date",
                "signup_count",
            ],
            build_metric_groups=[
                ["count(*) as signup_count", "count(*) signup_count"],
            ],
            report_terms=["daily_signup_report", "DuckDB", "signup_count"],
            correct_load_sql="""create or replace view staged_daily_signups as
select
  signup_id,
  cast(signup_date as date) as signup_date,
  channel
from raw_daily_signups;
""",
            correct_build_sql="""create or replace table daily_signup_counts as
select
  signup_date,
  count(*) as signup_count
from staged_daily_signups
group by 1;
""",
            correct_report_sql="""create or replace view daily_signup_report as
select
  signup_date,
  signup_count
from daily_signup_counts;
""",
            starter_files={
                "pipelines/report_job.yaml": """name: daily_signup_report_job
storage_path: mock_s3/staged/daily_signups.csv
raw_table: raw_daily_signups
load_sql: sql/load_raw.sql
build_sql: sql/build_table.sql
report_sql: sql/report_view.sql
final_view: daily_signup_report
""",
                "sql/load_raw.sql": """create or replace view staged_daily_signups as
select
  signup_id,
  cast(signup_date as date) as signup_date,
  channel
from raw_daily_signups;
""",
                "sql/build_table.sql": """create or replace table daily_signup_counts as
select
  signup_date,
  count(*) as signup_count
from staged_daily_signups
group by 1;
""",
                "sql/report_view.sql": """create or replace view daily_signup_report as
select
  signup_date,
  signup_count as signups
from daily_signup_counts;
""",
            },
            editable_targets_override=["sql/report_view.sql"],
            available_validators_override=[],
            expected_raw_preview=[
                {"signup_id": "S001", "signup_date": "2026-04-01", "channel": "organic"},
                {"signup_id": "S002", "signup_date": "2026-04-01", "channel": "paid"},
            ],
            expected_derived_preview=[
                {"signup_date": "2026-04-01", "signup_count": 2},
            ],
            expected_final_preview=[
                {"signup_date": "2026-04-02", "signup_count": 2},
            ],
        ),
        simulation_scenario(
            scenario_id="SIM-001",
            developer_request=(
                "Create a customer daily report from this sales-order CSV. The workflow "
                "should stage the raw data, load it into the warehouse, and publish a queryable report view."
            ),
            workspace_summary=(
                "This experimental workspace simulates a tiny data platform. Generate the "
                "pipeline YAML and SQL needed to stage the CSV into mock S3, load it into "
                "DuckDB, build a customer summary table, and publish the final report view. "
                "Builder starts with an empty workspace; Fixer repairs after runtime failures."
            ),
            source_csv_path="data/sales_orders.csv",
            source_csv_content="""order_id,customer_id,order_date,product_category,quantity,unit_price
1001,C001,2026-04-01,books,2,15.00
1002,C001,2026-04-01,games,1,25.00
1003,C002,2026-04-02,books,3,12.50
1004,C003,2026-04-02,gadgets,1,99.99
1005,C002,2026-04-03,games,2,30.00
""",
            raw_table="raw_sales_orders",
            staged_view="staged_sales_orders",
            build_table="customer_summary",
            final_view="customer_daily_report",
            raw_asset="raw.sales_orders_csv",
            final_asset="mart.customer_daily_report",
            date_column="order_date",
            raw_schema_columns=[
                ("order_id", "string"),
                ("customer_id", "string"),
                ("order_date", "date"),
                ("product_category", "string"),
                ("quantity", "integer"),
                ("unit_price", "float"),
            ],
            required_output_columns=[
                "customer_id",
                "order_date",
                "order_count",
                "gross_revenue_usd",
            ],
            build_metric_groups=[
                ["quantity * unit_price", "unit_price * quantity"],
                ["order_count"],
                ["gross_revenue_usd"],
            ],
            report_terms=["customer_daily_report", "DuckDB", "warehouse"],
            correct_load_sql="""create or replace view staged_sales_orders as
select
  order_id,
  customer_id,
  cast(order_date as date) as order_date,
  quantity,
  unit_price
from raw_sales_orders;
""",
            correct_build_sql="""create or replace table customer_summary as
select
  customer_id,
  order_date,
  count(*) as order_count,
  sum(quantity * unit_price) as gross_revenue_usd
from staged_sales_orders
group by 1, 2;
""",
            correct_report_sql="""create or replace view customer_daily_report as
select
  customer_id,
  order_date,
  order_count,
  gross_revenue_usd
from customer_summary;
""",
            expected_raw_preview=[
                {"order_id": "1001", "customer_id": "C001", "order_date": "2026-04-01"},
                {"order_id": "1002", "customer_id": "C001", "order_date": "2026-04-01"},
            ],
            expected_derived_preview=[
                {"customer_id": "C001", "order_date": "2026-04-01", "order_count": 2},
            ],
            expected_final_preview=[
                {"customer_id": "C001", "order_date": "2026-04-01", "gross_revenue_usd": 55.0},
            ],
        ),
        simulation_scenario(
            scenario_id="SIM-002",
            developer_request=(
                "Generate a plan-tier daily subscription report from this events CSV. "
                "The pipeline should stage the file, load it into the warehouse, and publish a final daily report."
            ),
            workspace_summary=(
                "Build a small DuckDB workflow that turns subscription events into a daily "
                "subscription report grouped by plan tier and date."
            ),
            source_csv_path="data/subscription_events.csv",
            source_csv_content="""event_id,account_id,event_date,plan_tier,event_type
E001,A100,2026-04-01,pro,started
E002,A101,2026-04-01,basic,started
E003,A100,2026-04-02,pro,cancelled
E004,A102,2026-04-02,pro,started
E005,A103,2026-04-02,basic,started
E006,A101,2026-04-03,basic,cancelled
""",
            raw_table="raw_subscription_events",
            staged_view="staged_subscription_events",
            build_table="subscription_daily_metrics",
            final_view="subscription_daily_report",
            raw_asset="raw.subscription_events_csv",
            final_asset="mart.subscription_daily_report",
            date_column="event_date",
            raw_schema_columns=[
                ("event_id", "string"),
                ("account_id", "string"),
                ("event_date", "date"),
                ("plan_tier", "string"),
                ("event_type", "string"),
            ],
            required_output_columns=[
                "plan_tier",
                "event_date",
                "new_subscriptions",
                "cancelled_subscriptions",
                "net_subscriptions",
            ],
            build_metric_groups=[
                ["event_type = 'started'", "event_type='started'"],
                ["event_type = 'cancelled'", "event_type='cancelled'"],
                ["net_subscriptions"],
            ],
            report_terms=["subscription_daily_report", "DuckDB", "plan_tier"],
            correct_load_sql="""create or replace view staged_subscription_events as
select
  event_id,
  account_id,
  cast(event_date as date) as event_date,
  plan_tier,
  lower(event_type) as event_type
from raw_subscription_events;
""",
            correct_build_sql="""create or replace table subscription_daily_metrics as
select
  plan_tier,
  event_date,
  sum(case when event_type = 'started' then 1 else 0 end) as new_subscriptions,
  sum(case when event_type = 'cancelled' then 1 else 0 end) as cancelled_subscriptions,
  sum(case when event_type = 'started' then 1 else 0 end) - sum(case when event_type = 'cancelled' then 1 else 0 end) as net_subscriptions
from staged_subscription_events
group by 1, 2;
""",
            correct_report_sql="""create or replace view subscription_daily_report as
select
  plan_tier,
  event_date,
  new_subscriptions,
  cancelled_subscriptions,
  net_subscriptions
from subscription_daily_metrics;
""",
            expected_raw_preview=[
                {"event_id": "E001", "plan_tier": "pro", "event_type": "started"},
                {"event_id": "E002", "plan_tier": "basic", "event_type": "started"},
            ],
            expected_derived_preview=[
                {"plan_tier": "basic", "event_date": "2026-04-01", "new_subscriptions": 1},
            ],
            expected_final_preview=[
                {"plan_tier": "pro", "event_date": "2026-04-02", "net_subscriptions": 0},
            ],
        ),
        simulation_scenario(
            scenario_id="SIM-003",
            developer_request=(
                "Create a warehouse-level inventory movement report from this CSV. "
                "Stage the file, load it into DuckDB, and publish a daily inventory view."
            ),
            workspace_summary=(
                "Generate a pipeline that turns inventory movement events into a daily "
                "warehouse inventory report with received, shipped, and net units."
            ),
            source_csv_path="data/inventory_movements.csv",
            source_csv_content="""movement_id,warehouse_id,movement_date,sku,movement_type,units
M001,W1,2026-04-01,SKU1,inbound,20
M002,W1,2026-04-01,SKU2,outbound,5
M003,W2,2026-04-01,SKU9,inbound,12
M004,W1,2026-04-02,SKU1,outbound,4
M005,W2,2026-04-02,SKU9,outbound,2
M006,W2,2026-04-02,SKU8,inbound,8
""",
            raw_table="raw_inventory_movements",
            staged_view="staged_inventory_movements",
            build_table="inventory_daily_balance",
            final_view="inventory_daily_report",
            raw_asset="raw.inventory_movements_csv",
            final_asset="mart.inventory_daily_report",
            date_column="movement_date",
            raw_schema_columns=[
                ("movement_id", "string"),
                ("warehouse_id", "string"),
                ("movement_date", "date"),
                ("sku", "string"),
                ("movement_type", "string"),
                ("units", "integer"),
            ],
            required_output_columns=[
                "warehouse_id",
                "movement_date",
                "received_units",
                "shipped_units",
                "net_units",
            ],
            build_metric_groups=[
                ["movement_type = 'inbound'", "movement_type='inbound'"],
                ["movement_type = 'outbound'", "movement_type='outbound'"],
                ["net_units"],
            ],
            report_terms=["inventory_daily_report", "DuckDB", "warehouse_id"],
            correct_load_sql="""create or replace view staged_inventory_movements as
select
  movement_id,
  warehouse_id,
  cast(movement_date as date) as movement_date,
  sku,
  lower(movement_type) as movement_type,
  units
from raw_inventory_movements;
""",
            correct_build_sql="""create or replace table inventory_daily_balance as
select
  warehouse_id,
  movement_date,
  sum(case when movement_type = 'inbound' then units else 0 end) as received_units,
  sum(case when movement_type = 'outbound' then units else 0 end) as shipped_units,
  sum(case when movement_type = 'inbound' then units else 0 end) - sum(case when movement_type = 'outbound' then units else 0 end) as net_units
from staged_inventory_movements
group by 1, 2;
""",
            correct_report_sql="""create or replace view inventory_daily_report as
select
  warehouse_id,
  movement_date,
  received_units,
  shipped_units,
  net_units
from inventory_daily_balance;
""",
            expected_raw_preview=[
                {"movement_id": "M001", "warehouse_id": "W1", "movement_type": "inbound"},
                {"movement_id": "M002", "warehouse_id": "W1", "movement_type": "outbound"},
            ],
            expected_derived_preview=[
                {"warehouse_id": "W1", "movement_date": "2026-04-01", "received_units": 20},
            ],
            expected_final_preview=[
                {"warehouse_id": "W2", "movement_date": "2026-04-02", "net_units": 6},
            ],
        ),
        simulation_scenario(
            scenario_id="SIM-004",
            developer_request=(
                "Repair the builder's broken customer-order pipeline so the staged CSV loads, "
                "the summary table builds correctly, and the final report contract stays intact."
            ),
            workspace_summary=(
                "Builder already generated a first pass of the customer report pipeline, but the "
                "runtime fails. Read the existing draft files, diagnose the SQL bug, and repair the workflow."
            ),
            source_csv_path="data/sales_orders.csv",
            source_csv_content="""order_id,customer_id,order_date,product_category,quantity,unit_price
2001,C010,2026-04-05,books,1,18.00
2002,C011,2026-04-05,gadgets,2,45.00
2003,C010,2026-04-06,games,1,35.00
2004,C012,2026-04-06,books,4,10.00
""",
            raw_table="raw_sales_orders",
            staged_view="staged_sales_orders",
            build_table="customer_summary",
            final_view="customer_daily_report",
            raw_asset="raw.sales_orders_csv",
            final_asset="mart.customer_daily_report",
            date_column="order_date",
            raw_schema_columns=[
                ("order_id", "string"),
                ("customer_id", "string"),
                ("order_date", "date"),
                ("product_category", "string"),
                ("quantity", "integer"),
                ("unit_price", "float"),
            ],
            required_output_columns=[
                "customer_id",
                "order_date",
                "order_count",
                "gross_revenue_usd",
            ],
            build_metric_groups=[
                ["quantity * unit_price", "unit_price * quantity"],
                ["gross_revenue_usd"],
            ],
            report_terms=["customer_daily_report", "DuckDB", "warehouse"],
            correct_load_sql="""create or replace view staged_sales_orders as
select
  order_id,
  customer_id,
  cast(order_date as date) as order_date,
  quantity,
  unit_price
from raw_sales_orders;
""",
            correct_build_sql="""create or replace table customer_summary as
select
  customer_id,
  order_date,
  count(*) as order_count,
  sum(quantity * unit_price) as gross_revenue_usd
from staged_sales_orders
group by 1, 2;
""",
            correct_report_sql="""create or replace view customer_daily_report as
select
  customer_id,
  order_date,
  order_count,
  gross_revenue_usd
from customer_summary;
""",
            starter_files={
                "pipelines/report_job.yaml": """name: customer_daily_report_job
storage_path: mock_s3/staged/sales_orders.csv
raw_table: raw_sales_orders
load_sql: sql/load_raw.sql
build_sql: sql/build_table.sql
report_sql: sql/report_view.sql
final_view: customer_daily_report
""",
                "sql/load_raw.sql": """create or replace view staged_sales_orders as
select
  order_id,
  customer_id,
  cast(order_date as date) as order_date,
  quantity,
  unit_price
from raw_sales_orders;
""",
                "sql/build_table.sql": """create or replace table customer_summary as
select
  customer_id,
  order_date,
  count(*) as order_count,
  sum(quantity * price) as gross_revenue_usd
from staged_sales_orders
group by 1, 2;
""",
                "sql/report_view.sql": """create or replace view customer_daily_report as
select
  customer_id,
  order_date,
  order_count,
  gross_revenue_usd
from customer_summary;
""",
            },
        ),
        simulation_scenario(
            scenario_id="SIM-005",
            developer_request=(
                "Fix this broken subscription reporting workflow. The builder used the right source CSV "
                "but the final pipeline no longer matches the expected report view contract."
            ),
            workspace_summary=(
                "A builder draft already exists for the subscription daily report. Repair the pipeline "
                "and SQL so the runtime publishes the intended final view and output columns."
            ),
            source_csv_path="data/subscription_events.csv",
            source_csv_content="""event_id,account_id,event_date,plan_tier,event_type
S001,A200,2026-04-07,pro,started
S002,A201,2026-04-07,basic,started
S003,A200,2026-04-08,pro,cancelled
S004,A202,2026-04-08,pro,started
""",
            raw_table="raw_subscription_events",
            staged_view="staged_subscription_events",
            build_table="subscription_daily_metrics",
            final_view="subscription_daily_report",
            raw_asset="raw.subscription_events_csv",
            final_asset="mart.subscription_daily_report",
            date_column="event_date",
            raw_schema_columns=[
                ("event_id", "string"),
                ("account_id", "string"),
                ("event_date", "date"),
                ("plan_tier", "string"),
                ("event_type", "string"),
            ],
            required_output_columns=[
                "plan_tier",
                "event_date",
                "new_subscriptions",
                "cancelled_subscriptions",
                "net_subscriptions",
            ],
            build_metric_groups=[
                ["event_type = 'started'", "event_type='started'"],
                ["event_type = 'cancelled'", "event_type='cancelled'"],
                ["net_subscriptions"],
            ],
            report_terms=["subscription_daily_report", "DuckDB", "plan_tier"],
            correct_load_sql="""create or replace view staged_subscription_events as
select
  event_id,
  account_id,
  cast(event_date as date) as event_date,
  plan_tier,
  lower(event_type) as event_type
from raw_subscription_events;
""",
            correct_build_sql="""create or replace table subscription_daily_metrics as
select
  plan_tier,
  event_date,
  sum(case when event_type = 'started' then 1 else 0 end) as new_subscriptions,
  sum(case when event_type = 'cancelled' then 1 else 0 end) as cancelled_subscriptions,
  sum(case when event_type = 'started' then 1 else 0 end) - sum(case when event_type = 'cancelled' then 1 else 0 end) as net_subscriptions
from staged_subscription_events
group by 1, 2;
""",
            correct_report_sql="""create or replace view subscription_daily_report as
select
  plan_tier,
  event_date,
  new_subscriptions,
  cancelled_subscriptions,
  net_subscriptions
from subscription_daily_metrics;
""",
            starter_files={
                "pipelines/report_job.yaml": """name: subscription_daily_report_job
storage_path: mock_s3/staged/subscription_events.csv
raw_table: raw_subscription_events
load_sql: sql/load_raw.sql
build_sql: sql/build_table.sql
report_sql: sql/report_view.sql
final_view: customer_daily_report
""",
                "sql/load_raw.sql": """create or replace view staged_subscription_events as
select
  event_id,
  account_id,
  cast(event_date as date) as event_date,
  plan_tier,
  lower(event_type) as event_type
from raw_subscription_events;
""",
                "sql/build_table.sql": """create or replace table subscription_daily_metrics as
select
  plan_tier,
  event_date,
  sum(case when event_type = 'started' then 1 else 0 end) as new_subscriptions,
  sum(case when event_type = 'cancelled' then 1 else 0 end) as cancelled_subscriptions,
  sum(case when event_type = 'started' then 1 else 0 end) - sum(case when event_type = 'cancelled' then 1 else 0 end) as net_subscriptions
from staged_subscription_events
group by 1, 2;
""",
                "sql/report_view.sql": """create or replace view customer_daily_report as
select
  plan_tier,
  event_date,
  new_subscriptions as started_subscriptions,
  cancelled_subscriptions,
  net_subscriptions
from subscription_daily_metrics;
""",
            },
        ),
        simulation_scenario(
            scenario_id="SIM-006",
            developer_request=(
                "Repair the inventory movement report. The current draft runs partway through, "
                "but the final report schema no longer matches the contract expected by downstream users."
            ),
            workspace_summary=(
                "You are in Fixer mode on an inventory reporting workflow. Read the broken draft "
                "and repair the output schema without changing the warehouse-level business meaning."
            ),
            source_csv_path="data/inventory_movements.csv",
            source_csv_content="""movement_id,warehouse_id,movement_date,sku,movement_type,units
I001,W5,2026-04-09,SKU1,inbound,15
I002,W5,2026-04-09,SKU2,outbound,3
I003,W6,2026-04-10,SKU8,inbound,10
I004,W6,2026-04-10,SKU8,outbound,4
""",
            raw_table="raw_inventory_movements",
            staged_view="staged_inventory_movements",
            build_table="inventory_daily_balance",
            final_view="inventory_daily_report",
            raw_asset="raw.inventory_movements_csv",
            final_asset="mart.inventory_daily_report",
            date_column="movement_date",
            raw_schema_columns=[
                ("movement_id", "string"),
                ("warehouse_id", "string"),
                ("movement_date", "date"),
                ("sku", "string"),
                ("movement_type", "string"),
                ("units", "integer"),
            ],
            required_output_columns=[
                "warehouse_id",
                "movement_date",
                "received_units",
                "shipped_units",
                "net_units",
            ],
            build_metric_groups=[
                ["movement_type = 'inbound'", "movement_type='inbound'"],
                ["movement_type = 'outbound'", "movement_type='outbound'"],
                ["net_units"],
            ],
            report_terms=["inventory_daily_report", "DuckDB", "warehouse_id"],
            correct_load_sql="""create or replace view staged_inventory_movements as
select
  movement_id,
  warehouse_id,
  cast(movement_date as date) as movement_date,
  sku,
  lower(movement_type) as movement_type,
  units
from raw_inventory_movements;
""",
            correct_build_sql="""create or replace table inventory_daily_balance as
select
  warehouse_id,
  movement_date,
  sum(case when movement_type = 'inbound' then units else 0 end) as received_units,
  sum(case when movement_type = 'outbound' then units else 0 end) as shipped_units,
  sum(case when movement_type = 'inbound' then units else 0 end) - sum(case when movement_type = 'outbound' then units else 0 end) as net_units
from staged_inventory_movements
group by 1, 2;
""",
            correct_report_sql="""create or replace view inventory_daily_report as
select
  warehouse_id,
  movement_date,
  received_units,
  shipped_units,
  net_units
from inventory_daily_balance;
""",
            starter_files={
                "pipelines/report_job.yaml": """name: inventory_daily_report_job
storage_path: mock_s3/staged/inventory_movements.csv
raw_table: raw_inventory_movements
load_sql: sql/load_raw.sql
build_sql: sql/build_table.sql
report_sql: sql/report_view.sql
final_view: inventory_daily_report
""",
                "sql/load_raw.sql": """create or replace view staged_inventory_movements as
select
  movement_id,
  warehouse_id,
  cast(movement_date as date) as movement_date,
  sku,
  lower(movement_type) as movement_type,
  units
from raw_inventory_movements;
""",
                "sql/build_table.sql": """create or replace table inventory_daily_balance as
select
  warehouse_id,
  movement_date,
  sum(case when movement_type = 'inbound' then units else 0 end) as received_units,
  sum(case when movement_type = 'outbound' then units else 0 end) as shipped_units,
  sum(case when movement_type = 'inbound' then units else 0 end) - sum(case when movement_type = 'outbound' then units else 0 end) as net_units
from staged_inventory_movements
group by 1, 2;
""",
                "sql/report_view.sql": """create or replace view inventory_daily_report as
select
  warehouse_id,
  movement_date,
  received_units as inbound_units,
  shipped_units as outbound_units,
  net_units
from inventory_daily_balance;
""",
            },
        ),
        simulation_scenario(
            scenario_id="SIM-007",
            developer_request=(
                "Builder already created an initial returns-report pipeline, but the runtime fails. "
                "Use the shared workspace to diagnose the failure and repair the pipeline so the final report view is queryable."
            ),
            workspace_summary=(
                "This shared Builder + Fixer workspace already contains a first-pass pipeline for a returns report. "
                "The builder draft stages the CSV and points the runtime at the right files, but one SQL artifact is broken. "
                "Fixer should use runtime feedback to repair the workflow and publish the expected report view."
            ),
            source_csv_path="data/returns_events.csv",
            source_csv_content="""return_id,customer_id,return_date,return_reason,refund_amount
R001,C300,2026-04-11,damaged,45.00
R002,C301,2026-04-11,wrong_item,30.00
R003,C300,2026-04-12,damaged,15.00
R004,C302,2026-04-12,late_delivery,22.50
R005,C301,2026-04-13,wrong_item,18.00
""",
            raw_table="raw_returns_events",
            staged_view="staged_returns_events",
            build_table="returns_daily_summary",
            final_view="returns_daily_report",
            raw_asset="raw.returns_events_csv",
            final_asset="mart.returns_daily_report",
            date_column="return_date",
            raw_schema_columns=[
                ("return_id", "string"),
                ("customer_id", "string"),
                ("return_date", "date"),
                ("return_reason", "string"),
                ("refund_amount", "float"),
            ],
            required_output_columns=[
                "return_reason",
                "return_date",
                "return_count",
                "refund_total_usd",
            ],
            build_metric_groups=[
                ["count(*) as return_count", "count(*) return_count"],
                ["sum(refund_amount) as refund_total_usd", "sum(refund_amount) refund_total_usd"],
            ],
            report_terms=["returns_daily_report", "DuckDB", "return_reason"],
            correct_load_sql="""create or replace view staged_returns_events as
select
  return_id,
  customer_id,
  cast(return_date as date) as return_date,
  return_reason,
  refund_amount
from raw_returns_events;
""",
            correct_build_sql="""create or replace table returns_daily_summary as
select
  return_reason,
  return_date,
  count(*) as return_count,
  sum(refund_amount) as refund_total_usd
from staged_returns_events
group by 1, 2;
""",
            correct_report_sql="""create or replace view returns_daily_report as
select
  return_reason,
  return_date,
  return_count,
  refund_total_usd
from returns_daily_summary;
""",
            starter_files={
                "pipelines/report_job.yaml": """name: returns_daily_report_job
storage_path: mock_s3/staged/returns_events.csv
raw_table: raw_returns_events
load_sql: sql/load_raw.sql
build_sql: sql/build_table.sql
report_sql: sql/report_view.sql
final_view: returns_daily_report
""",
                "sql/load_raw.sql": """create or replace view staged_returns_events as
select
  return_id,
  customer_id,
  cast(return_date as date) as return_date,
  return_reason,
  refund_amount
from raw_returns_events;
""",
                "sql/build_table.sql": """create or replace table returns_daily_summary as
select
  return_reason,
  returned_at,
  count(*) as return_count,
  sum(refund_amount) as refund_total_usd
from staged_returns_events
group by 1, 2;
""",
                "sql/report_view.sql": """create or replace view returns_daily_report as
select
  return_reason,
  return_date,
  return_count,
  refund_total_usd
from returns_daily_summary;
""",
            },
            expected_raw_preview=[
                {"return_id": "R001", "customer_id": "C300", "return_reason": "damaged"},
                {"return_id": "R002", "customer_id": "C301", "return_reason": "wrong_item"},
            ],
            expected_derived_preview=[
                {"return_reason": "damaged", "return_date": "2026-04-11", "return_count": 1},
            ],
            expected_final_preview=[
                {"return_reason": "wrong_item", "return_date": "2026-04-11", "refund_total_usd": 30.0},
            ],
        ),
    ],
}

SCENARIO_REGISTRY: dict[str, dict] = {
    scenario["scenario_id"]: scenario
    for scenario_group in SCENARIOS.values()
    for scenario in scenario_group
}


def _replace_text(value: str, mapping: dict[str, str]) -> str:
    rendered = value
    for old, new in mapping.items():
        rendered = rendered.replace(old, new)
    return rendered


def _replace_groups(groups: list[list[str]], mapping: dict[str, str]) -> list[list[str]]:
    return [[_replace_text(option, mapping) for option in group] for group in groups]


def _apply_variant_mapping(value: object, mapping: dict[str, str]) -> object:
    if isinstance(value, str):
        return _replace_text(value, mapping)
    if isinstance(value, list):
        return [_apply_variant_mapping(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _apply_variant_mapping(item, mapping) for key, item in value.items()}
    return value


def _seed_variant_index(seed: int, count: int, salt: int) -> int:
    if count <= 0:
        return 0
    return abs(seed + salt) % count


def _apply_seeded_variant(scenario: dict, seed: int | None) -> dict:
    if seed is None:
        return scenario

    scenario_id = scenario.get("scenario_id", "")

    if scenario_id == "PR-005":
        variants = [
            {
                "env_key": "TARGET_WAREHOUSE_ROLE",
                "stale_key": "WAREHOUSE_ROLE",
                "role": "analytics_writer",
            },
            {
                "env_key": "TARGET_REPORTING_ROLE",
                "stale_key": "REPORTING_ROLE",
                "role": "reporting_writer",
            },
        ]
        variant = variants[_seed_variant_index(seed, len(variants), 5)]
        mapping = {
            "TARGET_WAREHOUSE_ROLE": variant["env_key"],
            "WAREHOUSE_ROLE": variant["stale_key"],
            "analytics_writer": variant["role"],
        }
        scenario = _apply_variant_mapping(scenario, mapping)
        scenario["developer_request"] = (
            f"{scenario['developer_request']} The packaged runner now expects `{variant['env_key']}` "
            f"with the `{variant['role']}` role."
        )
        return scenario

    if scenario_id == "PR-006":
        variants = [
            {"id_col": "customer_code"},
            {"id_col": "member_code"},
            {"id_col": "profile_code"},
        ]
        variant = variants[_seed_variant_index(seed, len(variants), 6)]
        mapping = {"customer_code": variant["id_col"]}
        scenario = _apply_variant_mapping(scenario, mapping)
        scenario["developer_request"] = (
            f"{scenario['developer_request']} The failing identifier column is `{variant['id_col']}`."
        )
        return scenario

    if scenario_id == "LR-003":
        variants = [
            {"metric": "mrr_usd", "replacement": "monthly_recurring_revenue_usd"},
            {"metric": "arr_usd", "replacement": "annual_recurring_revenue_usd"},
            {"metric": "net_revenue_usd", "replacement": "net_revenue_amount_usd"},
        ]
        variant = variants[_seed_variant_index(seed, len(variants), 3)]
        mapping = {
            "mrr_usd": variant["metric"],
            "monthly_recurring_revenue_usd": variant["replacement"],
        }
        scenario = _apply_variant_mapping(scenario, mapping)
        scenario["developer_request"] = (
            f"{scenario['developer_request']} The published metric under review is `{variant['metric']}`."
        )
        return scenario

    if scenario_id == "WS-004":
        variants = [
            {"metric": "margin_delta_pct", "threshold": "-0.15"},
            {"metric": "net_revenue_delta_pct", "threshold": "-0.12"},
        ]
        variant = variants[_seed_variant_index(seed, len(variants), 4)]
        mapping = {
            "margin_delta_pct": variant["metric"],
            "-0.15": variant["threshold"],
        }
        scenario = _apply_variant_mapping(scenario, mapping)
        scenario["developer_request"] = (
            f"{scenario['developer_request']} The alert should monitor `{variant['metric']}` "
            f"and fire below `{variant['threshold']}`."
        )
        return scenario

    return scenario


def scenario_count(task_id: str) -> int:
    task = TASK_DEFINITIONS.get(task_id)
    if not task:
        raise ValueError(f"Unknown task_id '{task_id}'")
    return len(task.get("scenario_ids", []))


def get_task(task_id: str) -> dict:
    if task_id not in TASK_DEFINITIONS:
        raise ValueError(f"Unknown task_id '{task_id}'")
    task = deepcopy(TASK_DEFINITIONS[task_id])
    task["scenarios"] = scenario_count(task_id)
    return task


def get_scenario(task_id: str, index: int = 0, seed: int | None = None) -> dict:
    task = TASK_DEFINITIONS.get(task_id)
    if not task:
        raise ValueError(f"Unknown task_id '{task_id}'")
    scenario_ids = task.get("scenario_ids", [])
    if index < 0 or index >= len(scenario_ids):
        raise ValueError(
            f"Scenario index {index} out of range for task '{task_id}' "
            f"(available: 0-{len(scenario_ids) - 1})"
        )
    scenario_id = scenario_ids[index]
    scenario = deepcopy(SCENARIO_REGISTRY[scenario_id])
    return _apply_seeded_variant(scenario, seed)


def get_grader_family(task_id: str) -> str:
    task = TASK_DEFINITIONS.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task_id '{task_id}'")
    return str(task.get("grader_family", ""))


def list_tasks() -> list[dict]:
    return [get_task(task_id) for task_id in TASK_DEFINITIONS]
