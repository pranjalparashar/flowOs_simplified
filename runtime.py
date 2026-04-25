"""Local execution runtime for CSV-to-report simulation scenarios."""

from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only when dependency missing
    duckdb = None


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if current_list_key is None:
                continue
            data.setdefault(current_list_key, []).append(stripped[2:].strip())
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            current_list_key = key
            data[key] = []
        else:
            current_list_key = None
            data[key] = value.strip("\"'")
    return data


def _preview_rows(rows: list[tuple[Any, ...]], columns: list[str], limit: int = 3) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for row in rows[:limit]:
        preview.append({column: value for column, value in zip(columns, row)})
    return preview


def execute_csv_report_runtime(scenario: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    files = dict(scenario.get("files", {}))
    files.update(state.get("edited_files", {}))

    result: dict[str, Any] = {
        "ran": True,
        "succeeded": False,
        "checks": {
            "storage_stage_check": False,
            "duckdb_load_check": False,
            "report_view_check": False,
            "output_schema_check": False,
        },
        "logs": [],
        "errors": [],
        "output_schema": [],
        "report_preview": [],
        "row_count": 0,
        "db_path": "",
        "staged_csv_path": "",
        "run_dir": "",
        "materialized_artifacts": {},
    }

    if duckdb is None:
        result["errors"].append("duckdb dependency is not installed.")
        return result

    target = scenario.get("simulation_target", {}) or scenario.get("llm_draft", {}).get("simulation_target", {})
    source_csv = str(target.get("source_csv", ""))
    pipeline_path = str(target.get("pipeline_path", ""))
    load_sql_path = str(target.get("load_sql_path", ""))
    build_sql_path = str(target.get("build_sql_path", ""))
    report_sql_path = str(target.get("report_sql_path", ""))

    csv_content = files.get(source_csv, "")
    pipeline_text = files.get(pipeline_path, "")
    load_sql = files.get(load_sql_path, "")
    build_sql = files.get(build_sql_path, "")
    report_sql = files.get(report_sql_path, "")
    if not csv_content.strip():
        result["errors"].append(f"Missing source CSV: {source_csv}")
        return result
    if not pipeline_text.strip():
        result["errors"].append(f"Missing pipeline file: {pipeline_path}")
        return result
    if not load_sql.strip():
        result["errors"].append(f"Missing SQL file: {load_sql_path}")
        return result
    if not build_sql.strip():
        result["errors"].append(f"Missing SQL file: {build_sql_path}")
        return result
    if not report_sql.strip():
        result["errors"].append(f"Missing SQL file: {report_sql_path}")
        return result

    pipeline = _parse_simple_yaml(pipeline_text)
    required_keys = ["storage_path", "raw_table", "load_sql", "build_sql", "report_sql", "final_view"]
    missing_keys = [key for key in required_keys if not pipeline.get(key)]
    if missing_keys:
        result["errors"].append(f"Pipeline missing keys: {', '.join(missing_keys)}")
        return result

    episode_id = str(state.get("episode_id") or "unknown-episode")
    workdir = Path(tempfile.mkdtemp(prefix=f"flowos-runtime-{episode_id}-"))
    staged_csv = workdir / pipeline["storage_path"]
    db_path = workdir / "runtime.duckdb"
    workdir.mkdir(parents=True, exist_ok=True)
    result["db_path"] = str(db_path)
    result["staged_csv_path"] = str(staged_csv)
    result["run_dir"] = str(workdir)

    try:
        staged_csv.parent.mkdir(parents=True, exist_ok=True)
        staged_csv.write_text(csv_content, encoding="utf-8")
        result["materialized_artifacts"] = {
            pipeline_path: pipeline_text,
            load_sql_path: load_sql,
            build_sql_path: build_sql,
            report_sql_path: report_sql,
            source_csv: csv_content,
        }
        result["checks"]["storage_stage_check"] = True
        result["logs"].append(f"staged csv to {staged_csv}")

        con = duckdb.connect(str(db_path))
        raw_table = str(pipeline["raw_table"])
        quoted_csv_path = str(staged_csv).replace("'", "''")
        con.execute(
            f"create or replace table {raw_table} as "
            f"select * from read_csv_auto('{quoted_csv_path}', header=true)"
        )
        result["checks"]["duckdb_load_check"] = True
        result["logs"].append(f"loaded raw table {raw_table}")

        con.execute(load_sql)
        result["logs"].append(f"executed {load_sql_path}")

        con.execute(build_sql)
        result["logs"].append(f"executed {build_sql_path}")

        con.execute(report_sql)
        final_view = str(pipeline["final_view"])
        columns = [row[1] for row in con.execute(f"pragma table_info('{final_view}')").fetchall()]
        result["output_schema"] = columns
        result["checks"]["report_view_check"] = bool(columns)

        expected_columns = list(target.get("required_output_columns", []))
        if columns == expected_columns:
            result["checks"]["output_schema_check"] = True

        rows = con.execute(f"select * from {final_view} order by 1, 2 limit 3").fetchall()
        count = con.execute(f"select count(*) from {final_view}").fetchone()
        result["report_preview"] = _preview_rows(rows, columns)
        result["row_count"] = int(count[0]) if count else 0
        result["succeeded"] = all(result["checks"].values())
        result["logs"].append(f"queried {final_view} rows={result['row_count']}")
        con.close()
        result["db_path"] = ""
        result["staged_csv_path"] = ""
        result["run_dir"] = ""
        return result
    except Exception as exc:
        result["errors"].append(str(exc))
        result["db_path"] = ""
        result["staged_csv_path"] = ""
        result["run_dir"] = ""
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
